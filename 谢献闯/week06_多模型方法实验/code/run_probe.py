"""方法三的对照：预训练模型**不微调**，只当特征提取器 + 线性分类器。

回答的问题：「换预训练模型」的收益，有多少来自预训练表示本身、有多少来自微调？
做法：冻结 encoder，取 mean-pooling 句向量，训练 LinearSVC（与 TF-IDF 基线同一分类器，
      差别只在特征来自词面 n-gram 还是上下文向量），在去重 test 上评一次。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg import DATASET_SPECS, MODEL_REGISTRY, OUT_DIR
from data_lib import load_data
from metrics_lib import classification_report


@torch.no_grad()
def extract(model_path, texts, tok, device, max_length, batch_size=128):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(model_path)).to(device).eval()
    feats = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt")
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device), return_dict=True)
        hs = out.last_hidden_state
        m = enc["attention_mask"].to(device).unsqueeze(-1).float()
        feats.append(((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from sklearn.svm import LinearSVC
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = DATASET_SPECS[args.dataset]
    model_path = MODEL_REGISTRY[args.model]["path"]
    tok = AutoTokenizer.from_pretrained(str(model_path))
    data = load_data(args.dataset)

    t0 = time.time()
    Xtr = extract(model_path, [d["text"] for d in data["train"]], tok, device, spec["max_length"])
    Xte = extract(model_path, [d["text"] for d in data["test"]], tok, device, spec["max_length"])
    extract_s = round(time.time() - t0, 1)

    ytr = [d["label"] for d in data["train"]]
    yte = [d["label"] for d in data["test"]]

    t1 = time.time()
    clf = LinearSVC(C=1.0).fit(Xtr, ytr)
    y_pred = clf.predict(Xte).tolist()
    fit_s = round(time.time() - t1, 1)

    report = classification_report(yte, y_pred, labels=list(range(spec["num_labels"])))

    train_texts = {d["text"] for d in data["train"]}
    keep = [i for i, d in enumerate(data["test"]) if d["text"] not in train_texts]
    clean_rep = classification_report(
        [yte[i] for i in keep], [y_pred[i] for i in keep],
        labels=list(range(spec["num_labels"])))
    clean_rep["n_removed"] = len(data["test"]) - len(keep)
    clean_rep["removed_rate"] = round((len(data["test"]) - len(keep)) / len(data["test"]), 4)

    out = Path(args.out) if args.out else OUT_DIR / f"probe_{args.model}_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "frozen_probe",
        "model_key": args.model,
        "model_display": MODEL_REGISTRY[args.model]["display"] + "（冻结特征 + LinearSVC）",
        "dataset": args.dataset, "dataset_display": spec["display"],
        "extract_seconds": extract_s, "lr_fit_seconds": fit_s,
        "train_seconds": round(extract_s + fit_s, 1),
        "data_meta": data["meta"], "report": report, "report_clean": clean_rep,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[probe/{args.model}/{args.dataset}] extract={extract_s}s "
          f"acc={report['acc']} f1={report['macro_f1']} | clean f1={clean_rep['macro_f1']}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
