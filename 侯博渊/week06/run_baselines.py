"""
同一批 500 条干净验证样本上的轻量基线：规则 / TF-IDF+LinearSVC / TF-IDF+SGD(balanced)

目的：给「换小模型」提供同口径的低成本对照。
  - 规则：项目里的 L0 基线（类别名命中，否则猜多数类）
  - TF-IDF + LinearSVC：项目里的 L1 基线（char 1-2gram，C=1.0）
  - TF-IDF + SGD(log_loss, class_weight=balanced)：服务里实际用的后端
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from common.data import load_clean_val, load_splits  # noqa: E402
from common.labels import LABELS  # noqa: E402
from common.metrics import classification_report, pred_distribution  # noqa: E402

OUT_DIR = ROOT / "experiments" / "outputs"


def save(name, meta, report, samples, preds):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["pred_distribution"] = pred_distribution(preds)
    payload = {
        "meta": meta,
        "report": report,
        "preds": [
            {"text": d["sentence"], "true": int(d["label"]), "pred": int(p)}
            for d, p in zip(samples, preds)
        ],
    }
    (OUT_DIR / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT_DIR / name}")


def main():
    train, _ = load_splits()
    samples, meta = load_clean_val(500)
    y_true = [d["label"] for d in samples]
    texts = [d["sentence"] for d in samples]

    # ---------- 1) 规则 ----------
    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]
    rule_preds = []
    for t in texts:
        pred = fallback
        for i, name in enumerate(LABELS):
            if name in t:
                pred = i
                break
        rule_preds.append(pred)
    hit = sum(1 for t in texts if any(name in t for name in LABELS)) / len(texts)
    rule_report = classification_report(y_true, rule_preds)
    rule_report.update({"method": "rule_keyword", "hit_rate": round(hit, 4)})
    save("rule_pred.json", dict(meta), rule_report, samples, rule_preds)

    # ---------- 2) TF-IDF + LinearSVC（项目 L1 基线） ----------
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=300000)
    Xtr = vec.fit_transform([d["sentence"] for d in train])
    clf = LinearSVC(C=1.0).fit(Xtr, [d["label"] for d in train])
    svm_preds = clf.predict(vec.transform(texts)).tolist()
    svm_seconds = round(time.time() - t0, 1)
    svm_report = classification_report(y_true, svm_preds)
    svm_report.update({"method": "tfidf_svm", "fit_seconds": svm_seconds})
    save("tfidf_pred.json", dict(meta), svm_report, samples, svm_preds)

    # ---------- 3) TF-IDF + SGD(balanced)（服务后端同款） ----------
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(train))
    n_cal = max(1000, len(train) // 10)
    fit_set = [train[i] for i in idx[n_cal:]]
    t0 = time.time()
    vec2 = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=300000,
                           min_df=2, sublinear_tf=True)
    X2 = vec2.fit_transform([d["sentence"] for d in fit_set])
    sgd = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                        class_weight="balanced", random_state=42)
    sgd.fit(X2, [d["label"] for d in fit_set])
    sgd_preds = sgd.predict(vec2.transform(texts)).tolist()
    sgd_seconds = round(time.time() - t0, 1)
    sgd_report = classification_report(y_true, sgd_preds)
    sgd_report.update({"method": "tfidf_sgd_balanced", "fit_seconds": sgd_seconds,
                       "n_fit": len(fit_set)})
    save("tfidf_sgd_pred.json", dict(meta), sgd_report, samples, sgd_preds)

    print("\n规则      : acc=%.4f macro_f1=%.4f hit=%.2f%%" % (
        rule_report["acc"], rule_report["macro_f1"], hit * 100))
    print("TF-IDF SVC: acc=%.4f macro_f1=%.4f (%.1fs)" % (
        svm_report["acc"], svm_report["macro_f1"], svm_seconds))
    print("TF-IDF SGD: acc=%.4f macro_f1=%.4f (%.1fs)" % (
        sgd_report["acc"], sgd_report["macro_f1"], sgd_seconds))


if __name__ == "__main__":
    main()
