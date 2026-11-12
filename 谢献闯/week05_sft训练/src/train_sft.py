# -*- coding: utf-8 -*-
"""
第 2 步：SFT 训练（LoRA）。

和课件里说的一致：结构不动，只训练 LoRA 适配器（约 0.7% 参数），
损失函数还是 next-token 交叉熵，只不过 prompt 段的 label 是 -100，不参与计算。

用法：
    python src/train_sft.py                     # 默认 3 epoch
    python src/train_sft.py --epochs 1 --max_steps 50   # 冒烟测试
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SFTDataset(Dataset):
    def __init__(self, path: Path, limit: int = 0):
        self.data = torch.load(path, weights_only=False)
        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集 loss（只在回答段算）+ 回答 token 的预测准确率。

    注意：因果 LM 里位置 i 的 logits 预测的是第 i+1 个 token，
    所以算准确率时 logits 要和 labels 错开一位（和内部 loss 的 shift 一致）。
    """
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
        pred = out.logits[:, :-1].argmax(dim=-1)
        target = batch["labels"][:, 1:]
        mask = target != -100
        correct += (pred[mask] == target[mask]).sum().item()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "ppl": math.exp(min(total_loss / max(total_tokens, 1), 20)),
        "token_acc": correct / max(total_tokens, 1),
    }


def char_f1(pred: str, gold: str) -> float:
    """字符级 F1：开放问答里完全匹配太苛刻，用它衡量「答得像不像」。"""
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    p, r = overlap / len(pred), overlap / len(gold)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


@torch.no_grad()
def generation_score(model, tokenizer, rows, device, max_new_tokens=48, batch_size=8):
    """每轮抽几条验证样本真实生成一遍，算字符级 F1 / 完全匹配率。"""
    model.eval()
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    im_end_id = tokenizer.convert_tokens_to_ids(config.IM_END)
    gens = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        enc = tokenizer([config.build_prompt(r["instruction"]) for r in chunk],
                        return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=im_end_id)
        for i, seq in enumerate(out):
            gens.append(tokenizer.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    tokenizer.padding_side = prev_side
    golds = [r["output"] for r in rows]
    f1 = sum(char_f1(g, t) for g, t in zip(gens, golds)) / max(len(golds), 1)
    em = sum(1 for g, t in zip(gens, golds) if g.strip() == t.strip()) / max(len(golds), 1)
    return {"gen_char_f1": round(f1, 4), "gen_em": round(em, 4)}


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B LoRA SFT")
    parser.add_argument("--model", default=str(config.MODEL_DIR))
    parser.add_argument("--save_dir", default=str(config.ADAPTER_DIR))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=config.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=config.LORA_ALPHA)
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条训练（调试用）")
    parser.add_argument("--max_steps", type=int, default=0, help="最多训练多少步（调试用）")
    parser.add_argument("--grad_ckpt", action="store_true", help="开启梯度检查点（省显存、降速）")
    parser.add_argument("--patience", type=int, default=2, help="连续几轮无提升就早停")
    parser.add_argument("--gen_samples", type=int, default=20, help="每轮抽多少条验证样本做真实生成打分（0=不做）")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    config.ensure_dirs()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print("=" * 70)
    print(f"设备      : {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"基座模型  : {args.model}")
    print(f"数据类型  : {dtype}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    train_ds = SFTDataset(config.PROCESSED_DIR / "train.pt", args.limit)
    val_ds = SFTDataset(config.PROCESSED_DIR / "val.pt")
    collate = make_collate(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collate, num_workers=0)
    print(f"训练 {len(train_ds)} 条 / 验证 {len(val_ds)} 条 | 每轮 {len(train_loader)} 个 batch")

    # 每轮用来做「真实生成」打分的验证样本
    all_val_rows = [json.loads(l) for l in
                    (config.DATA_DIR / "sft_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gen_rows = all_val_rows[:args.gen_samples] if args.gen_samples else []

    # ---------- 模型 + LoRA ----------
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    if args.grad_ckpt:                       # 显存不够时再开，速度会慢一半左右
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGETS, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.WARMUP_RATIO), total_steps
    )

    # ---------- 训练 ----------
    log_rows, best_loss, bad_epochs, best_epoch = [], float("inf"), 0, 0
    best_score, best_gen = (-1.0, float("-inf")), {}
    global_step, t0 = 0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(trainable, config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            n = (batch["labels"] != -100).sum().item()
            running += loss.item() * args.grad_accum * n
            seen += n
            pbar.set_postfix(loss=f"{running / max(seen, 1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if args.max_steps and global_step >= args.max_steps:
                break

        train_loss = running / max(seen, 1)
        val = evaluate(model, val_loader, device)
        gen = generation_score(model, tokenizer, gen_rows, device) if gen_rows else {"gen_char_f1": None, "gen_em": None}
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} "
              f"| ppl={val['ppl']:.2f} | token_acc={val['token_acc']:.4f} "
              f"| gen_f1={gen['gen_char_f1']} gen_em={gen['gen_em']}")
        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val["loss"], 6),
                         "val_ppl": round(val["ppl"], 4), "val_token_acc": round(val["token_acc"], 6),
                         "gen_char_f1": gen["gen_char_f1"], "gen_em": gen["gen_em"],
                         "steps": global_step})

        # 模型选择：优先「答得像」（生成 F1），同分再比 loss —— 这类开放回答任务上，
        # 光看 teacher-forcing 的 val_loss 容易选到「会背不会答」的那一轮
        score = (gen["gen_char_f1"] if gen["gen_char_f1"] is not None else 0.0, -val["loss"])
        if score > best_score:
            best_score, best_loss, best_epoch, bad_epochs = score, val["loss"], epoch, 0
            best_gen = dict(gen)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    ↑ 生成 F1 提升，保存 LoRA 到 {save_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"早停：连续 {args.patience} 轮没有提升。")
                break
        if args.max_steps and global_step >= args.max_steps:
            print(f"达到 --max_steps={args.max_steps}，停止。")
            break

    elapsed = time.time() - t0

    with open(config.OUTPUT_DIR / "train_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    info = {
        "base_model": Path(args.model).name,
        "method": f"LoRA r={args.lora_r} alpha={args.lora_alpha}",
        "trainable_params": n_train,
        "total_params": n_all,
        "trainable_ratio": round(n_train / n_all, 6),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "best_gen_char_f1": best_gen.get("gen_char_f1"),
        "best_gen_em": best_gen.get("gen_em"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "train_seconds": round(elapsed, 1),
        "device": str(device),
        "train_size": len(train_ds),
    }
    (save_dir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps = [r["epoch"] for r in log_rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(eps, [r["train_loss"] for r in log_rows], "o-", label="train loss")
        axes[0].plot(eps, [r["val_loss"] for r in log_rows], "s-", label="val loss")
        axes[0].set_xticks(eps); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
        axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(eps, [r["val_token_acc"] for r in log_rows], "o-", label="val token acc")
        axes[1].set_xticks(eps); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1)
        axes[1].legend(); axes[1].grid(alpha=.3)
        fig.suptitle("Qwen3-0.6B LoRA SFT")
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / "curves.png", dpi=150)
        print(f"[写出] {config.OUTPUT_DIR / 'curves.png'}")
    except Exception as e:
        print(f"[警告] 画图失败：{e}")

    print("=" * 70)
    print(f"训练完成，用时 {elapsed / 60:.1f} 分钟 | 最佳 epoch={best_epoch} | best val_loss={best_loss:.4f}")
    print(f"LoRA 适配器已保存到 {save_dir}（只有适配器，几十 MB）")
    print("下一步：python src/evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
