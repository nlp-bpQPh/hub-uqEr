"""方法一：规则 / 关键词（L0 基线，只适用于 TNEWS 15 类）。

规则：标题里出现哪个类别名就判哪类，都不命中则退回训练集众数类。
它的作用不是刷分，而是给出「不训练模型能拿到多少」的下限。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg import DATASET_SPECS, OUT_DIR
from data_lib import load_data
from metrics_lib import classification_report


def classify(text, labels, fallback):
    for i, name in enumerate(labels):
        if name in text:
            return i
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tnews", choices=["tnews"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = load_data(args.dataset)
    labels = DATASET_SPECS[args.dataset]["labels"]
    fallback = Counter(d["label"] for d in data["train"]).most_common(1)[0][0]

    y_true = [d["label"] for d in data["test"]]
    y_pred = [classify(d["text"], labels, fallback) for d in data["test"]]
    hit = sum(1 for d in data["test"] if any(n in d["text"] for n in labels))

    report = classification_report(y_true, y_pred)
    report["hit_rate"] = round(hit / len(data["test"]), 4)

    # 去重测试集口径
    train_texts = {d["text"] for d in data["train"]}
    keep = [i for i, d in enumerate(data["test"]) if d["text"] not in train_texts]
    clean_rep = classification_report([y_true[i] for i in keep], [y_pred[i] for i in keep])
    clean_rep["n_removed"] = len(data["test"]) - len(keep)
    clean_rep["removed_rate"] = round((len(data["test"]) - len(keep)) / len(data["test"]), 4)

    out = Path(args.out) if args.out else OUT_DIR / f"rule_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "rule_keyword", "model_display": "规则 / 关键词",
        "dataset": args.dataset, "dataset_display": DATASET_SPECS[args.dataset]["display"],
        "fit_seconds": 0, "data_meta": data["meta"], "report": report,
        "report_clean": clean_rep,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[rule] hit={report['hit_rate']:.2%} acc={report['acc']} f1={report['macro_f1']} "
          f"| clean f1={clean_rep['macro_f1']}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
