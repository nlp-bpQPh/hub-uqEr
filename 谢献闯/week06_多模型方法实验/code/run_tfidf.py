"""方法二：TF-IDF + LinearSVC（无预训练权重的传统基线），三个数据集通用。

同一套字 1-2gram + C=1.0，不调参——它的定位就是「20 秒能拿到多少」的下限参照。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg import DATASET_SPECS, OUT_DIR
from data_lib import load_data
from metrics_lib import classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tnews", choices=list(DATASET_SPECS))
    ap.add_argument("--ngram", default="1,2")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    data = load_data(args.dataset)
    spec = DATASET_SPECS[args.dataset]
    lo, hi = (int(x) for x in args.ngram.split(","))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), max_features=300000)

    t0 = time.time()
    Xtr = vec.fit_transform([d["text"] for d in data["train"]])
    clf = LinearSVC(C=args.C).fit(Xtr, [d["label"] for d in data["train"]])
    Xte = vec.transform([d["text"] for d in data["test"]])
    t_infer = time.time()
    y_pred = clf.predict(Xte).tolist()
    fit_seconds = round(time.time() - t0, 1)
    infer_ms = round((time.time() - t_infer) * 1000 / len(data["test"]), 2)

    report = classification_report([d["label"] for d in data["test"]], y_pred,
                                   labels=list(range(spec["num_labels"])))

    # 去重测试集口径：剔除与训练集完全同句的样本
    train_texts = {d["text"] for d in data["train"]}
    keep = [i for i, d in enumerate(data["test"]) if d["text"] not in train_texts]
    clean_rep = classification_report(
        [data["test"][i]["label"] for i in keep], [y_pred[i] for i in keep],
        labels=list(range(spec["num_labels"])))
    clean_rep["n_removed"] = len(data["test"]) - len(keep)
    clean_rep["removed_rate"] = round((len(data["test"]) - len(keep)) / len(data["test"]), 4)

    out = Path(args.out) if args.out else OUT_DIR / f"tfidf_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "tfidf_svm", "model_display": f"TF-IDF + LinearSVC (char{args.ngram}, C={args.C})",
        "dataset": args.dataset, "dataset_display": spec["display"],
        "fit_seconds": fit_seconds, "ms_per_sample": infer_ms,
        "data_meta": data["meta"], "report": report, "report_clean": clean_rep,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[tfidf/{args.dataset}] fit={fit_seconds}s infer={infer_ms}ms/条 "
          f"acc={report['acc']} f1={report['macro_f1']} | clean acc={clean_rep['acc']} "
          f"f1={clean_rep['macro_f1']} (n={clean_rep['n']})")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
