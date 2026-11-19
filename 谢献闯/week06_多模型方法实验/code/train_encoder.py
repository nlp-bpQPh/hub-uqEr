"""通用编码器微调脚本：同一套训练配方，只换预训练权重或数据集。

对照实验的三条纪律：
  1) 同一数据集上，所有编码器用**完全相同**的超参（epoch / lr / batch / 池化 / 类别加权）；
  2) dev 只用于选 epoch，test 只在最后评一次；
  3) 记录训练耗时与显存峰值——「涨几个点」必须和成本一起看。

用法：
  python code/train_encoder.py --model bert-base --dataset tnews
  python code/train_encoder.py --model roberta-wwm-ext --dataset thucnews
  python code/train_encoder.py --model macbert-base --dataset chnsenticorp --epochs 3
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg import DATASET_SPECS, MODEL_REGISTRY, OUT_DIR
from data_lib import load_data
from metrics_lib import classification_report, expected_calibration_error


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, data, tok, max_length):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["text"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


class EncoderClassifier(nn.Module):
    """Encoder + 池化 + 线性头（与课程 bert/train_bert.py 同结构，换模型时只换 encoder）。"""

    def __init__(self, model_path, num_labels, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(str(model_path))
        self.pool = pool
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


@torch.no_grad()
def evaluate(model, loader, device, num_labels, return_probs=False):
    model.eval()
    ys, ps, probs = [], [], []
    t0 = time.time()
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        p = torch.softmax(logits.float(), dim=-1)
        ps.extend(p.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
        if return_probs:
            probs.extend(p.cpu().tolist())
    report = classification_report(ys, ps, labels=list(range(num_labels)))
    report["ms_per_sample"] = round((time.time() - t0) * 1000 / max(1, len(ys)), 1)
    if return_probs:
        report["ece"] = expected_calibration_error(ys, probs)
        report["probs"] = probs
    report["_y_true"] = ys
    report["_y_pred"] = ps
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    ap.add_argument("--max_length", type=int, default=0, help="0=按数据集默认")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="冒烟用：每 epoch 最多步数")
    ap.add_argument("--max_train", type=int, default=0, help="冒烟用：限制训练条数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = DATASET_SPECS[args.dataset]
    num_labels = spec["num_labels"]
    max_length = args.max_length or spec["max_length"]
    model_path = MODEL_REGISTRY[args.model]["path"]

    data = load_data(args.dataset)
    if args.max_train:
        data["train"] = data["train"][: args.max_train]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path))

    train_ds = TextDataset(data["train"], tok, max_length)
    dev_ds = TextDataset(data["dev"], tok, max_length)
    test_ds = TextDataset(data["test"], tok, max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size * 2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2)

    model = EncoderClassifier(model_path, num_labels, pool=args.pool).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_encoder_params = sum(p.numel() for p in model.encoder.parameters())

    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    enc_params = [p for n, p in model.named_parameters()
                  if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    cw = compute_class_weight("balanced", classes=np.arange(num_labels),
                              y=[d["label"] for d in data["train"]])
    weight = torch.tensor(cw, dtype=torch.float).to(device)
    crit = nn.CrossEntropyLoss(weight=weight)

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"[cfg] model={MODEL_REGISTRY[args.model]['display']} dataset={args.dataset} "
          f"train={len(data['train'])} dev={len(data['dev'])} test={len(data['test'])} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} maxlen={max_length} "
          f"params={n_params/1e6:.1f}M", flush=True)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_f1, best_state, best_epoch = [], 0.0, None, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 100 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        dev_rep = evaluate(model, dev_dl, device, num_labels)
        history.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                        "dev_acc": dev_rep["acc"], "dev_macro_f1": dev_rep["macro_f1"],
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: dev_acc={dev_rep['acc']} dev_f1={dev_rep['macro_f1']} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # 用 dev 选模型：只有 dev macro-F1 提升才覆盖（test 全程不可见）
        if dev_rep["macro_f1"] > best_f1:
            best_f1, best_epoch = dev_rep["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = round(time.time() - t_start, 1)
    peak_mem = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 0) if device == "cuda" else 0

    # 用 dev 最优 epoch 的权重评 test（只评一次）
    model.load_state_dict(best_state)
    model.to(device)
    test_rep = evaluate(model, test_dl, device, num_labels, return_probs=True)

    # 去重测试集：剔除与训练集**完全同句**的样本，避免词面泄漏抬高所有方法
    train_texts = {d["text"] for d in data["train"]}
    keep = [i for i, d in enumerate(data["test"]) if d["text"] not in train_texts]
    clean_rep = classification_report(
        [data["test"][i]["label"] for i in keep],
        [test_rep["_y_pred"][i] for i in keep], labels=list(range(num_labels)))
    clean_rep["n_removed"] = len(data["test"]) - len(keep)
    clean_rep["removed_rate"] = round((len(data["test"]) - len(keep)) / len(data["test"]), 4)

    report = {
        "method": "encoder_finetune",
        "model_key": args.model,
        "model_display": MODEL_REGISTRY[args.model]["display"],
        "dataset": args.dataset,
        "dataset_display": spec["display"],
        "args": vars(args),
        "n_params": n_params,
        "n_encoder_params": n_encoder_params,
        "train_seconds": train_seconds,
        "peak_gpu_mb": peak_mem,
        "best_epoch": best_epoch,
        "dev_best_macro_f1": best_f1,
        "history": history,
        "data_meta": data["meta"],
        "report": {k: v for k, v in test_rep.items() if not k.startswith("_")},
        "report_clean": clean_rep,
        "preds": [{"text": d["text"], "true": d["label"], "pred": p}
                  for d, p in zip(data["test"], test_rep["_y_pred"])],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"enc_{args.model}_{args.dataset}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    r = report["report"]
    print(f"[test] acc={r['acc']} macro_f1={r['macro_f1']} ece={r.get('ece')} "
          f"best_epoch={best_epoch} train={train_seconds}s peak={peak_mem}MB", flush=True)
    print(f"[test-clean] n={clean_rep['n']} acc={clean_rep['acc']} macro_f1={clean_rep['macro_f1']} "
          f"(removed {clean_rep['n_removed']})", flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
