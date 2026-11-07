# -*- coding: utf-8 -*-
"""
第二步：微调 BERT 做 12 国语言识别（文本分类）。

做法：在预训练好的 bert-base-multilingual-cased 顶层加一个 12 类的分类头，
用 [CLS] 向量接一层全连接 + softmax，用带标签的语言数据做有监督微调。

用法：
    python src/train.py                      # 默认 3 轮、batch 16、lr 2e-5
    python src/train.py --epochs 5 --batch_size 32
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根目录（config.py）
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))  # 同目录（train.py）
import config

try:  # Windows 控制台默认 GBK，避免中文/emoji 打印报错
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 数据集：文本先 tokenize 成 id，拼 batch 时再做动态 padding（比统一 pad 到 128 省算力）
# --------------------------------------------------------------------------
class LangDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.encodings = tokenizer(
            df["text"].tolist(), truncation=True, max_length=max_len, padding=False
        )
        self.labels = df["label_id"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encodings["input_ids"][idx], self.labels[idx]


def make_collate(pad_id: int):
    def collate(batch):
        ids = [torch.tensor(b[0], dtype=torch.long) for b in batch]
        input_ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        attention_mask = (input_ids != pad_id).long()
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool):
    model.eval()
    losses, preds, golds = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", enabled=use_amp):
            outputs = model(**batch)
        losses.append(outputs.loss.item())
        preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        golds.extend(batch["labels"].cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(golds, preds),
        "macro_f1": f1_score(golds, preds, average="macro"),
    }, preds, golds


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="微调 BERT 做 12 国语言识别")
    parser.add_argument("--model", default=str(config.PRETRAINED_DIR), help="预训练模型路径或 HF 名称")
    parser.add_argument("--save_dir", default=str(config.SAVE_DIR), help="最佳模型保存目录")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_len", type=int, default=config.MAX_LEN)
    parser.add_argument("--warmup_ratio", type=float, default=config.WARMUP_RATIO)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no_amp", action="store_true", help="关闭混合精度（默认 GPU 上开启）")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    config.ensure_dirs()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    use_amp = (device.type == "cuda") and not args.no_amp
    print("=" * 66)
    print(f"设备      : {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"预训练模型: {args.model}")
    print(f"类别数    : {config.NUM_LABELS}  ({' / '.join(config.LABELS)})")
    print(f"超参      : epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"max_len={args.max_len} amp={use_amp}")
    print("=" * 66)

    # ---------- 数据 ----------
    train_df = pd.read_csv(config.DATA_PROCESSED / "train.csv", encoding="utf-8-sig")
    dev_df = pd.read_csv(config.DATA_PROCESSED / "dev.csv", encoding="utf-8-sig")
    print(f"训练集 {len(train_df)} 条 | 验证集 {len(dev_df)} 条")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id or 0

    train_ds = LangDataset(train_df, tokenizer, args.max_len)
    dev_ds = LangDataset(dev_df, tokenizer, args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=make_collate(pad_id), num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=make_collate(pad_id), num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---------- 模型 ----------
    id2label = {int(i): c for i, c in config.ID2LABEL.items()}      # {0: "zho", ...}
    label2id = {c: int(i) for c, i in config.LABEL2ID.items()}      # {"zho": 0, ...}
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        num_labels=config.NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr, eps=1e-8)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------- 训练 ----------
    log_path = config.OUTPUT_DIR / "train_log.csv"
    log_rows, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            bsz = batch["labels"].size(0)
            running += loss.item() * bsz
            seen += bsz
            pbar.set_postfix(loss=f"{running / max(seen, 1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running / max(seen, 1)
        dev_metrics, _, _ = evaluate(model, dev_loader, device, use_amp)
        lr_now = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} | dev_loss={dev_metrics['loss']:.4f} "
              f"| dev_acc={dev_metrics['acc']:.4f} | dev_macro_f1={dev_metrics['macro_f1']:.4f} "
              f"| lr={lr_now:.2e}")
        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "dev_loss": round(dev_metrics["loss"], 6), "dev_acc": round(dev_metrics["acc"], 6),
            "dev_macro_f1": round(dev_metrics["macro_f1"], 6), "lr": lr_now,
        })

        if dev_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = dev_metrics["macro_f1"], epoch, 0
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    ↑ 验证集 macro-F1 提升，保存模型到 {save_dir}")
        else:
            bad_epochs += 1
            print(f"    → 未提升（{bad_epochs}/{config.EARLY_STOP_PATIENCE}）")
            if bad_epochs >= config.EARLY_STOP_PATIENCE:
                print("早停：验证集指标不再提升。")
                break

    elapsed = time.time() - t_start

    # ---------- 收尾：日志 / 曲线 / 说明 ----------
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    info = {
        "base_model": args.model,
        "num_labels": config.NUM_LABELS,
        "labels": config.LABELS,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": round(best_f1, 6),
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_len": args.max_len, "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay, "amp": use_amp, "seed": args.seed,
        },
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "train_seconds": round(elapsed, 1),
        "device": str(device),
    }
    with open(save_dir / "train_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        eps = [r["epoch"] for r in log_rows]
        axes[0].plot(eps, [r["train_loss"] for r in log_rows], "o-", label="train loss")
        axes[0].plot(eps, [r["dev_loss"] for r in log_rows], "s-", label="dev loss")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[0].set_xticks(eps)
        axes[1].plot(eps, [r["dev_acc"] for r in log_rows], "o-", label="dev acc")
        axes[1].plot(eps, [r["dev_macro_f1"] for r in log_rows], "s-", label="dev macro-F1")
        axes[1].set_xlabel("epoch"); axes[1].set_ylim(0.9, 1.005); axes[1].legend(); axes[1].grid(alpha=.3)
        axes[1].set_xticks(eps)
        fig.suptitle("BERT language-ID fine-tuning")
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / "curves.png", dpi=150)
        plt.close(fig)
        print(f"[写出] {config.OUTPUT_DIR / 'curves.png'}")
    except Exception as e:  # 画图失败不影响训练结果
        print(f"[警告] 训练曲线绘制失败：{e}")

    print("=" * 66)
    print(f"训练完成，用时 {elapsed / 60:.1f} 分钟；最佳 epoch={best_epoch}，dev macro-F1={best_f1:.4f}")
    print(f"模型已保存到 {save_dir}")
    print("下一步：python src/evaluate.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
