"""
补充评测：在 dev 1,000 上算规则 / TF-IDF 的指标，并读取 tiny 的 dev 日志。

目的：说明 dev 和 test 分布不同——dev 上的分数明显更高，不能拿 dev 选模型后直接当 test 估计。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from experiments_thucnews.run_baselines import rule_predict  # noqa: E402

DATA = ROOT / "_data" / "thucnews"
OUT = ROOT / "experiments_thucnews" / "outputs"


def load(split):
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


def main():
    train, dev = load("train"), load("dev")
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    y = [d["label"] for d in dev]
    fb = Counter(d["label"] for d in train).most_common(1)[0][0]

    rule_pred = [rule_predict(d["text"], labels, fb) for d in dev]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=300000,
                          min_df=2, sublinear_tf=True)
    X = vec.fit_transform([d["text"] for d in train])
    clf = LinearSVC(C=1.0).fit(X, [d["label"] for d in train])
    svc_pred = clf.predict(vec.transform([d["text"] for d in dev])).tolist()

    log = json.loads((OUT / "small_bert_log.json").read_text(encoding="utf-8"))
    tiny_best = max(log, key=lambda x: x["dev_macro_f1"])

    out = {
        "rule": {"acc": round(float(accuracy_score(y, rule_pred)), 4),
                 "macro_f1": round(float(f1_score(y, rule_pred, average="macro")), 4)},
        "tfidf_svc": {"acc": round(float(accuracy_score(y, svc_pred)), 4),
                      "macro_f1": round(float(f1_score(y, svc_pred, average="macro")), 4)},
        "tiny": {"macro_f1": tiny_best["dev_macro_f1"],
                 "best_epoch": tiny_best["epoch"]},
        "n_dev": len(dev),
    }
    (OUT / "dev_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
