"""
THUCNews 10 类的 BERT-base 线性探针：冻结 BERT，抽 [CLS]，逻辑回归。

本机无 GPU，BERT-base 全量微调 4,000×3 约 3–4 小时，跑不动；
  - 训练特征：从 train 4,000 里按类分层抽 2,000（200/类）
  - 测试：test 1,500 全量
  - max_length=128（长新闻会截断）
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

_TMP = ROOT / "_tmp"
_TMP.mkdir(exist_ok=True)
os.environ["TMP"] = str(_TMP)
os.environ["TEMP"] = str(_TMP)
os.environ["TMPDIR"] = str(_TMP)
tempfile.tempdir = str(_TMP)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from common.metrics import (  # noqa: E402
    classification_report,
    expected_calibration_error,
    pred_distribution,
)
from common.model_paths import bert_path  # noqa: E402

DATA = ROOT / "_data" / "thucnews"
OUT = ROOT / "experiments_thucnews" / "outputs"
NUM_LABELS = 10


def load(split):
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


def stratified_sample(rows, n, seed=42):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    rng = random.Random(seed)
    per = max(1, n // len(by))
    out = []
    for lab in sorted(by):
        items = by[lab][:]
        rng.shuffle(items)
        out.extend(items[:min(per, len(items))])
    rng.shuffle(out)
    return out


@torch.no_grad()
def extract(model, tok, texts, max_length, batch):
    feats = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt")
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    return_dict=True)
        feats.append(out.last_hidden_state[:, 0].cpu())
        if i % (batch * 8) == 0:
            print(f"  extract {i}/{len(texts)}", flush=True)
    return torch.cat(feats).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    torch.set_num_threads(8)
    from transformers import AutoModel, AutoTokenizer

    train, test = load("train"), load("test")
    subset = stratified_sample(train, args.n_train)
    print(f"[data] BERT 特征训练集 {len(subset)}，测试 {len(test)}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(bert_path()))
    model = AutoModel.from_pretrained(str(bert_path())).eval()
    Xtr = extract(model, tok, [d["text"] for d in subset], args.max_length, args.batch_size)
    Xte = extract(model, tok, [d["text"] for d in test], args.max_length, args.batch_size)
    print(f"[feature] {time.time()-t0:.1f}s | {Xtr.shape} / {Xte.shape}", flush=True)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             solver="liblinear")
    clf.fit(Xtr, [d["label"] for d in subset])
    probs = clf.predict_proba(Xte)
    preds = probs.argmax(axis=1).tolist()
    y_true = [d["label"] for d in test]
    report = classification_report(y_true, preds, num_classes=NUM_LABELS)
    report.update({
        "method": "bert_base_linear_probe",
        "params_m": 102.3,
        "n_train_features": len(subset),
        "max_length": args.max_length,
        "train_seconds": round(time.time() - t0, 1),
        "ece": expected_calibration_error(y_true, probs),
        "pred_distribution": pred_distribution(preds, num_classes=NUM_LABELS),
    })
    print(json.dumps({k: report[k] for k in ("acc", "macro_f1", "n_train_features")},
                     ensure_ascii=False))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"dataset": "THUCNews", "n_train_total": len(train),
                 "n_train_features": len(subset), "n_test": len(test), "seed": 42},
        "report": report,
        "preds": [{"text": d["text"], "true": int(t), "pred": int(p),
                   "conf": round(float(c), 4)}
                  for d, t, p, c in zip(test, y_true, preds, probs.max(axis=1))],
    }
    (OUT / "bert_probe_pred.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→", OUT / "bert_probe_pred.json")


if __name__ == "__main__":
    main()
