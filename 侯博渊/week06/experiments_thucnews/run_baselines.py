"""
THUCNews 10 类新闻分类的轻量基线：规则关键词 / TF-IDF+LinearSVC / TF-IDF+LogReg。

数据：train 4,000 / dev 1,000 / test 1,500（10 类平衡，val/test 来自官方 split）
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from common.metrics import (  # noqa: E402
    classification_report,
    expected_calibration_error,
    pred_distribution,
)

DATA = ROOT / "_data" / "thucnews"
OUT = ROOT / "experiments_thucnews" / "outputs"
NUM_LABELS = 10

KEYWORDS = {
    "体育": ["球", "比赛", "冠军", "奥运", "NBA", "足球", "篮球", "联赛", "队员", "教练"],
    "娱乐": ["明星", "电影", "演唱会", "综艺", "演员", "导演", "电视剧", "票房", "粉丝"],
    "家居": ["装修", "家具", "户型", "客厅", "卧室", "厨房", "建材", "家居"],
    "房产": ["楼市", "房价", "地产", "商品房", "开发商", "限购", "二手房", "房贷"],
    "教育": ["高考", "学生", "学校", "老师", "考研", "大学", "招生", "课程", "留学"],
    "时尚": ["时尚", "穿搭", "服装", "品牌", "美妆", "发型", "潮流", "奢侈品"],
    "时政": ["政府", "国家", "主席", "会议", "政策", "国务院", "人大", "政协", "外交"],
    "游戏": ["游戏", "玩家", "手游", "电竞", "副本", "版本", "英雄", "主机"],
    "科技": ["手机", "科技", "互联网", "AI", "芯片", "华为", "苹果", "5G", "智能"],
    "财经": ["股票", "基金", "经济", "市场", "银行", "投资", "股价", "上市", "财报"],
}


def load(split):
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


def save(name, report, rows, preds, probs=None):
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["pred_distribution"] = pred_distribution(preds, num_classes=NUM_LABELS)
    payload = {
        "meta": {"dataset": "THUCNews", "n_train": 4000, "n_dev": 1000,
                 "n_test": len(rows), "labels": NUM_LABELS},
        "report": report,
        "preds": [
            {"text": d["text"], "true": d["label"], "pred": int(p),
             **({"conf": round(float(c), 4)} if probs is not None else {})}
            for d, p, c in zip(rows, preds, probs if probs is not None else [None] * len(rows))
        ],
    }
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→", OUT / name)


def rule_predict(text, label2id, fallback):
    best, best_label = 0, fallback
    for name, words in KEYWORDS.items():
        cnt = sum(text.count(w) for w in words)
        if cnt > best:
            best, best_label = cnt, label2id[name]
    return best_label


def main():
    train, test = load("train"), load("test")
    label2id = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    x_train = [d["text"] for d in train]
    y_train = [d["label"] for d in train]
    x_test = [d["text"] for d in test]
    y_test = [d["label"] for d in test]
    fallback = Counter(y_train).most_common(1)[0][0]

    # 1) 规则
    t0 = time.time()
    rule_preds = [rule_predict(t, label2id, fallback) for t in x_test]
    rule_report = classification_report(y_test, rule_preds, num_classes=NUM_LABELS)
    rule_report.update({"method": "rule_keyword", "fit_seconds": round(time.time() - t0, 2)})
    save("rule_pred.json", rule_report, test, rule_preds)

    # 2) TF-IDF + LinearSVC
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=300000,
                          min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(x_train)
    Xte = vec.transform(x_test)
    svc = LinearSVC(C=1.0).fit(Xtr, y_train)
    svc_preds = svc.predict(Xte).tolist()
    svc_report = classification_report(y_test, svc_preds, num_classes=NUM_LABELS)
    svc_report.update({"method": "tfidf_svm", "fit_seconds": round(time.time() - t0, 1)})
    save("tfidf_svc_pred.json", svc_report, test, svc_preds)

    # 3) TF-IDF + LogReg
    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                            solver="liblinear").fit(Xtr, y_train)
    probs = lr.predict_proba(Xte)
    lr_preds = probs.argmax(axis=1).tolist()
    lr_report = classification_report(y_test, lr_preds, num_classes=NUM_LABELS)
    lr_report.update({"method": "tfidf_lr", "fit_seconds": round(time.time() - t0, 1),
                      "ece": expected_calibration_error(y_test, probs)})
    save("tfidf_lr_pred.json", lr_report, test, lr_preds, probs=probs.max(axis=1))

    for name, r in [("规则", rule_report), ("TF-IDF+SVC", svc_report), ("TF-IDF+LR", lr_report)]:
        print(f"{name:<12} acc={r['acc']:.4f} macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
