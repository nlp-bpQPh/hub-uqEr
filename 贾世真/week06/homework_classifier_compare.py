"""
Week6 作业：更换预训练模型后的中文新闻分类方法对比。

原课程：bert-base-chinese；本作业：hfl/rbt3（3 层轻量中文 BERT）。
固定：TNEWS、训练/验证样本、随机种子、Accuracy/Macro-F1 与逐条预测口径。

运行：
  python3 homework_classifier_compare.py --download-only
  python3 homework_classifier_compare.py --train-size 3000 --eval-size 500 --epochs 2
  python3 homework_classifier_compare.py --train-size 150 --eval-size 30 --epochs 1 \
      --max-steps 2 --output-dir results_smoke

数据目录：
  data/train.json、data/val.json，元素为 {"sentence": str, "label": int}。
"""

import argparse
import csv
import gc
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LABELS = [
    "故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
    "科技", "军事", "旅游", "国际", "证券", "农业", "电竞",
]
NUM_CLASSES = len(LABELS)


def parse_args():
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=base / "data")
    ap.add_argument("--model-dir", type=Path, default=base / "pretrain_models" / "rbt3")
    ap.add_argument("--output-dir", type=Path, default=base / "results")
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="每轮最多训练步数；0 为完整训练")
    ap.add_argument("--methods", default="rule,tfidf,rbt3_frozen,rbt3_finetuned")
    ap.add_argument("--download-only", action="store_true")
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path):
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    assert rows and all("sentence" in r and "label" in r for r in rows)
    assert all(0 <= int(r["label"]) < NUM_CLASSES for r in rows)
    return rows


def build_split(args):
    """固定训练子集；验证集先去掉与训练集整句重复的数据，再抽样。"""
    train = load_rows(args.data_dir / "train.json")
    val = load_rows(args.data_dir / "val.json")
    train_sentences = {r["sentence"] for r in train}
    clean_val = [r for r in val if r["sentence"] not in train_sentences]
    n_dup = len(val) - len(clean_val)
    if not 0 < args.train_size <= len(train):
        raise ValueError(f"train-size 应在 1 到 {len(train)} 之间")
    if not 0 < args.eval_size <= len(clean_val):
        raise ValueError(f"eval-size 应在 1 到 {len(clean_val)} 之间")

    rng = random.Random(args.seed)
    train_positions = sorted(rng.sample(range(len(train)), args.train_size))
    eval_positions = sorted(rng.sample(range(len(clean_val)), args.eval_size))
    train_rows = [train[i] for i in train_positions]
    eval_rows = [clean_val[i] for i in eval_positions]
    meta = {
        "dataset": "TNEWS",
        "original_train": len(train),
        "original_val": len(val),
        "duplicates_removed": n_dup,
        "clean_val": len(clean_val),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "seed": args.seed,
        "train_label_counts": {LABELS[k]: v for k, v in sorted(Counter(r["label"] for r in train_rows).items())},
        "eval_label_counts": {LABELS[k]: v for k, v in sorted(Counter(r["label"] for r in eval_rows).items())},
        "train_positions": train_positions,
        "eval_positions_in_clean_val": eval_positions,
    }
    return train_rows, eval_rows, meta


def ensure_model(model_dir):
    weights_exist = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    if (model_dir / "config.json").exists() and weights_exist:
        print(f"[model] 使用本地模型 {model_dir}", flush=True)
        return model_dir
    from huggingface_hub import snapshot_download

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print("[model] 下载 hfl/rbt3（仅保留 PyTorch 与 tokenizer 必要文件）", flush=True)
    snapshot_download(
        repo_id="hfl/rbt3",
        local_dir=str(model_dir),
        allow_patterns=["*.json", "*.txt", "model.safetensors", "pytorch_model.bin"],
    )
    return model_dir


def classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    labels = list(range(NUM_CLASSES))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class_recall": {LABELS[i]: round(float(recalls[i]), 4) for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "used_classes": len(set(y_pred)),
        "prediction_counts": {LABELS[k]: v for k, v in sorted(Counter(y_pred).items())},
    }


def evaluate_method(name, train_rows, eval_rows, fit_seconds=0.0, eval_seconds=0.0, extra=None):
    y_true = [int(r["label"]) for r in eval_rows]
    y_pred = name(y_true) if callable(name) else name
    result = classification_metrics(y_true, y_pred)
    result.update({"fit_seconds": round(fit_seconds, 2), "eval_seconds": round(eval_seconds, 2)})
    if extra:
        result.update(extra)
    return result, y_pred


def run_rule(train_rows, eval_rows):
    t0 = time.time()
    fallback = Counter(int(r["label"]) for r in train_rows).most_common(1)[0][0]
    fit_seconds = time.time() - t0
    t0 = time.time()
    preds = []
    hits = 0
    for row in eval_rows:
        found = next((i for i, label in enumerate(LABELS) if label in row["sentence"]), None)
        hits += int(found is not None)
        preds.append(fallback if found is None else found)
    extra = {"keyword_hit_rate": round(hits / len(eval_rows), 4), "majority_fallback": LABELS[fallback]}
    return evaluate_method(preds, train_rows, eval_rows, fit_seconds, time.time() - t0, extra)


def run_tfidf(train_rows, eval_rows, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=300000)
    x_train = vectorizer.fit_transform([r["sentence"] for r in train_rows])
    x_eval = vectorizer.transform([r["sentence"] for r in eval_rows])
    classifier = LinearSVC(C=1.0, random_state=seed)
    classifier.fit(x_train, [int(r["label"]) for r in train_rows])
    fit_seconds = time.time() - t0
    t0 = time.time()
    preds = classifier.predict(x_eval).tolist()
    extra = {"features": int(x_train.shape[1]), "classifier": "LinearSVC(C=1.0)", "ngram": "char 1-2"}
    return evaluate_method(preds, train_rows, eval_rows, fit_seconds, time.time() - t0, extra)


def load_tokenizer_model(model_dir, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoder = AutoModel.from_pretrained(str(model_dir)).to(device)
    return tokenizer, encoder


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def extract_embeddings(rows, tokenizer, encoder, device, batch_size, max_length):
    encoder.eval()
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        enc = tokenizer(
            [r["sentence"] for r in batch], truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(**enc, return_dict=True)
        chunks.append(mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_frozen(train_rows, eval_rows, model_dir, device, args):
    tokenizer, encoder = load_tokenizer_model(model_dir, device)
    t0 = time.time()
    x_train = extract_embeddings(train_rows, tokenizer, encoder, device, args.batch_size, args.max_length)
    x_eval = extract_embeddings(eval_rows, tokenizer, encoder, device, args.batch_size, args.max_length)
    encode_seconds = time.time() - t0
    del encoder
    gc.collect()

    from sklearn.svm import LinearSVC

    t0 = time.time()
    classifier = LinearSVC(C=1.0, random_state=args.seed)
    classifier.fit(x_train, [int(r["label"]) for r in train_rows])
    fit_seconds = encode_seconds + time.time() - t0
    t0 = time.time()
    preds = classifier.predict(x_eval).tolist()
    extra = {
        "pretrained_model": "hfl/rbt3", "encoder_trainable": False,
        "pooling": "mean_without_pad", "embedding_dim": int(x_train.shape[1]),
        "encode_seconds_in_fit": round(encode_seconds, 2), "classifier": "LinearSVC(C=1.0)",
    }
    return evaluate_method(preds, train_rows, eval_rows, fit_seconds, time.time() - t0, extra)


class EncodedDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.enc = tokenizer(
            [r["sentence"] for r in rows], truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        self.labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.enc["input_ids"][index],
            "attention_mask": self.enc["attention_mask"][index],
            "labels": self.labels[index],
        }


class Rbt3Classifier(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(str(model_dir))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))


@torch.no_grad()
def predict_torch(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def run_finetuned(train_rows, eval_rows, model_dir, device, args):
    from transformers import AutoTokenizer

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    train_ds = EncodedDataset(train_rows, tokenizer, args.max_length)
    eval_ds = EncodedDataset(eval_rows, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    model = Rbt3Classifier(model_dir).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * 5},
    ], weight_decay=0.01)
    counts = np.bincount([int(r["label"]) for r in train_rows], minlength=NUM_CLASSES)
    weights = len(train_rows) / (NUM_CLASSES * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    steps_per_epoch = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / warmup if step < warmup else max(
            0.0, (total_steps - step) / max(1, total_steps - warmup)
        ),
    )

    train_log = []
    best_f1 = -1.0
    best_preds = None
    best_epoch = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        t_epoch = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            steps += 1
            if steps == 1 or steps % 25 == 0:
                print(f"  [finetune] epoch={epoch} step={steps}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        epoch_preds = predict_torch(model, eval_loader, device)
        epoch_metrics = classification_metrics([int(r["label"]) for r in eval_rows], epoch_preds)
        train_log.append({
            "epoch": epoch, "train_loss": round(running_loss / max(1, steps), 4),
            "val_accuracy": epoch_metrics["accuracy"], "val_macro_f1": epoch_metrics["macro_f1"],
            "seconds": round(time.time() - t_epoch, 2),
        })
        print(f"  [finetune] epoch={epoch} val_acc={epoch_metrics['accuracy']:.4f} "
              f"val_f1={epoch_metrics['macro_f1']:.4f}", flush=True)
        if epoch_metrics["macro_f1"] > best_f1:
            best_f1 = epoch_metrics["macro_f1"]
            best_preds = epoch_preds
            best_epoch = epoch
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(), "model_name": "hfl/rbt3",
                "pooling": "mean_without_pad", "best_epoch": best_epoch,
                "val_macro_f1": best_f1, "max_length": args.max_length,
            }, args.output_dir / "rbt3_finetuned.pt")
    fit_seconds = time.time() - t_start
    extra = {
        "pretrained_model": "hfl/rbt3", "encoder_trainable": True,
        "pooling": "mean_without_pad", "loss": "balanced_cross_entropy",
        "lr": args.lr, "head_lr": args.lr * 5, "epochs": args.epochs,
        "best_epoch": best_epoch, "train_log": train_log,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return evaluate_method(best_preds, train_rows, eval_rows, fit_seconds, 0.0, extra)


def save_results(args, split_meta, results, predictions):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in run_config.items()}
    split_without_positions = {k: v for k, v in split_meta.items() if not k.endswith("positions") and k != "eval_positions_in_clean_val"}
    payload = {"run_config": run_config, "dataset": split_without_positions, "methods": results}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "split.json").write_text(json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["method", "accuracy", "macro_f1", "weighted_f1", "used_classes", "fit_seconds", "eval_seconds"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({key: metrics.get(key, "") for key in fields} | {"method": name})
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for method_name, (rows, preds) in predictions.items():
            for row, pred in zip(rows, preds):
                f.write(json.dumps({
                    "method": method_name, "text": row["sentence"],
                    "true_id": int(row["label"]), "true": LABELS[int(row["label"])],
                    "pred_id": int(pred), "pred": LABELS[int(pred)],
                }, ensure_ascii=False) + "\n")
    (out / "train_log.json").write_text(json.dumps(
        results.get("rbt3_finetuned", {}).get("train_log", []), ensure_ascii=False, indent=2
    ), encoding="utf-8")


def save_plot(results, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 不可用，跳过图片", flush=True)
        return
    names = list(results)
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [results[n]["accuracy"] for n in names], width, label="Accuracy", color="#85B7EB")
    ax.bar(x + width / 2, [results[n]["macro_f1"] for n in names], width, label="Macro-F1", color="#1D9E75")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("TNEWS small-sample comparison (same train/eval split)")
    ax.set_xticks(x, ["Rule", "TF-IDF", "RBT3 frozen", "RBT3 fine-tuned"][:len(names)])
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_model(args.model_dir)
    if args.download_only:
        print("[done] 模型已就绪", flush=True)
        return

    train_rows, eval_rows, split_meta = build_split(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid_methods = {"rule", "tfidf", "rbt3_frozen", "rbt3_finetuned"}
    unknown = set(methods) - valid_methods
    if unknown:
        raise ValueError(f"未知方法：{sorted(unknown)}")
    print(f"[data] train={len(train_rows)} eval={len(eval_rows)} "
          f"removed_duplicates={split_meta['duplicates_removed']} device={device}", flush=True)

    results = {}
    predictions = {}
    for method in methods:
        print(f"\n===== {method} =====", flush=True)
        if method == "rule":
            metrics, preds = run_rule(train_rows, eval_rows)
        elif method == "tfidf":
            metrics, preds = run_tfidf(train_rows, eval_rows, args.seed)
        elif method == "rbt3_frozen":
            metrics, preds = run_frozen(train_rows, eval_rows, args.model_dir, device, args)
        else:
            metrics, preds = run_finetuned(train_rows, eval_rows, args.model_dir, device, args)
        metrics["method"] = method
        results[method] = metrics
        predictions[method] = (eval_rows, preds)
        save_results(args, split_meta, results, predictions)
        print(f"[result] {method}: acc={metrics['accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f} fit={metrics['fit_seconds']:.1f}s", flush=True)

    save_plot(results, args.output_dir)
    print("\n===== 汇总 =====", flush=True)
    print(json.dumps({name: {k: r[k] for k in ["accuracy", "macro_f1", "weighted_f1", "fit_seconds"]}
                      for name, r in results.items()}, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] 结果目录：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
