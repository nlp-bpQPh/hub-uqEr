"""
换预训练模型实验：StructBERT-tiny（4 层 / 256 维 / 8.8M 参数）在 TNEWS 15 类上微调。

为什么选它：
  - 本机无 GPU，BERT-base（12 层 / 102M）全量微调约 3–5 天，跑不动；
  - StructBERT-tiny 只有 4 层 256 维，单步约 1.3s，3 epoch 约 1 小时；
  - 同一份 train（53,360）、同一批 500 条干净验证样本（seed=42），
    与项目里 BERT-base / TF-IDF / 规则的结果可直接横向比较。

用法：
  python experiments/train_small_bert.py --epochs 3
  python experiments/train_small_bert.py --epochs 1 --max_steps 3 --no_save   # 冒烟
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

# 沙箱环境下系统临时目录不可写，统一指到工作区
_TMP = ROOT / "_tmp"
_TMP.mkdir(exist_ok=True)
os.environ["TMP"] = str(_TMP)
os.environ["TEMP"] = str(_TMP)
os.environ["TMPDIR"] = str(_TMP)
tempfile.tempdir = str(_TMP)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common.data import load_clean_val, load_splits  # noqa: E402
from common.metrics import (  # noqa: E402
    classification_report,
    expected_calibration_error,
    pred_distribution,
)

MODEL_DIR = ROOT / "_models" / "structbert-tiny"
OUT_DIR = ROOT / "experiments" / "outputs"
NUM_LABELS = 15


class TinyClassifier(nn.Module):
    """BERT 编码器 + [CLS] 池化 + 线性头（与项目里的 BertClassifier 同构，只是换成本地小模型）"""

    def __init__(self, bert, hidden=256, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        vec = out.last_hidden_state[:, 0]  # [CLS]
        return self.classifier(self.dropout(vec))


def load_tiny_model():
    from transformers import BertConfig, BertModel

    cd = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    cfg = BertConfig(
        vocab_size=cd["vocab_size"],
        hidden_size=cd["hidden_size"],
        num_hidden_layers=cd["num_hidden_layers"],
        num_attention_heads=cd["num_attention_heads"],
        intermediate_size=cd["intermediate_size"],
        hidden_act=cd.get("hidden_act", "gelu"),
        hidden_dropout_prob=cd.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=cd.get("attention_probs_dropout_prob", 0.1),
        max_position_embeddings=cd["max_position_embeddings"],
        type_vocab_size=cd["type_vocab_size"],
        initializer_range=cd.get("initializer_range", 0.02),
    )
    model = BertModel(cfg)
    sd = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=False)
    # 预训练权重是 BertForMaskedLM 格式，前缀是 bert.；cls.* 是 MLM 头，丢掉
    new_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        raise RuntimeError(f"权重缺失: {missing[:5]}")
    print(f"[model] StructBERT-tiny 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"| 忽略的键: {unexpected}", flush=True)
    return model, cfg


def build_tokenizer():
    from transformers import BertTokenizerFast

    # 该模型目录只有 vocab.txt，没有 tokenizer.json；用 vocab 构建 fast tokenizer
    return BertTokenizerFast(vocab_file=str(MODEL_DIR / "vocab.txt"), do_lower_case=False)


def encode(tok, texts, max_length, batch=1000):
    ids, masks = [], []
    for i in range(0, len(texts), batch):
        enc = tok(
            texts[i:i + batch],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
    return torch.cat(ids), torch.cat(masks)


class TnewsTensors(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids, self.masks, self.labels = ids, masks, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.masks[i], self.labels[i]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for ids, masks, y in loader:
        logits = model(ids.to(device), masks.to(device))
        p = torch.softmax(logits.float(), dim=-1).cpu()
        probs.append(p)
        preds.extend(p.argmax(-1).tolist())
        ys.extend(y.tolist())
    return np.array(ys), np.array(preds), torch.cat(probs).numpy()


def evaluate(model, loader, device):
    y_true, y_pred, probs = predict(model, loader, device)
    report = classification_report(y_true, y_pred)
    report["ece"] = expected_calibration_error(y_true, probs)
    report["pred_distribution"] = pred_distribution(y_pred)
    return report, y_true, y_pred, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5, help="小模型用 5e-5，分类头 ×2")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时每个 epoch 只跑这么多步（冒烟）")
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    device = "cpu"

    tok = build_tokenizer()
    train, _ = load_splits()
    eval_samples, meta = load_clean_val(500)

    t0 = time.time()
    train_ids, train_masks = encode(tok, [d["sentence"] for d in train], args.max_length)
    train_labels = torch.tensor([d["label"] for d in train], dtype=torch.long)
    eval_ids, eval_masks = encode(tok, [d["sentence"] for d in eval_samples], args.max_length)
    eval_labels = torch.tensor([d["label"] for d in eval_samples], dtype=torch.long)
    print(f"[data] train={len(train)} eval={len(eval_samples)} | 编码耗时 {time.time()-t0:.1f}s", flush=True)

    train_ds = TnewsTensors(train_ids, train_masks, train_labels)
    eval_ds = TnewsTensors(eval_ids, eval_masks, eval_labels)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    bert, cfg = load_tiny_model()
    model = TinyClassifier(bert, hidden=cfg.hidden_size).to(device)

    # 类别加权：和 BERT-base 实验保持同一口径
    from sklearn.utils.class_weight import compute_class_weight

    cw = compute_class_weight(
        "balanced", classes=np.arange(NUM_LABELS), y=[d["label"] for d in train]
    )
    weight = torch.tensor(cw, dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert_params = [p for n, p in model.named_parameters()
                   if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 2},
        ],
        weight_decay=0.01,
    )

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: step / max(1, warmup)
        if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)),
    )

    print(f"[cfg] model=StructBERT-tiny(4L256, 8.8M) epochs={args.epochs} batch={args.batch_size} "
          f"lr={args.lr} max_length={args.max_length} steps/epoch={steps_per_epoch} "
          f"eval_n={len(eval_samples)} save={'off' if args.no_save else 'on'}", flush=True)

    best_f1, best_state, best_epoch, best_report = 0.0, None, 0, None
    log = []
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running, n_step = 0.0, 0
        for i, (ids, masks, y) in enumerate(train_dl):
            opt.zero_grad(set_to_none=True)
            logits = model(ids.to(device), masks.to(device))
            loss = crit(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        report, _, _, _ = evaluate(model, eval_dl, device)
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(1, n_step), 4),
            "val_acc": report["acc"],
            "val_macro_f1": report["macro_f1"],
            "elapsed_s": round(time.time() - t_epoch, 1),
        }
        log.append(row)
        print(f"epoch {epoch}: val_acc={report['acc']} val_f1={report['macro_f1']} "
              f"({row['elapsed_s']}s)", flush=True)

        if report["macro_f1"] > best_f1:
            best_f1 = report["macro_f1"]
            best_epoch = epoch
            best_report = report
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    train_seconds = round(time.time() - t_train, 1)
    # 用最佳 epoch 的权重做最终评测
    assert best_state is not None
    model.load_state_dict(best_state)
    final_report, y_true, y_pred, probs = evaluate(model, eval_dl, device)
    final_report.update({
        "method": "structbert_tiny_4l256",
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "pool": "cls",
    })

    print("\n" + str(final_report))
    print(f"\n最佳 epoch {best_epoch} | Macro-F1 {best_f1} | 训练 {train_seconds}s")

    if args.no_save:
        print("--no_save：不写文件")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta.update({
        "model": "iic/nlp_structbert_backbone_tiny_std",
        "params_m": final_report["params_m"],
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "seed": 42,
    })
    preds = [
        {"text": d["sentence"], "true": int(t), "pred": int(p), "conf": round(float(c), 4)}
        for d, t, p, c in zip(eval_samples, y_true, y_pred, probs.max(axis=1))
    ]
    (OUT_DIR / "small_bert_pred.json").write_text(
        json.dumps({"meta": meta, "report": final_report, "preds": preds},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "small_bert_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"state_dict": best_state, "config": cfg.to_dict(),
                "report": final_report, "meta": meta},
               OUT_DIR / "small_bert_ckpt.pt")
    print(f"→ {OUT_DIR / 'small_bert_pred.json'}")
    print(f"→ {OUT_DIR / 'small_bert_log.json'}")


if __name__ == "__main__":
    main()
