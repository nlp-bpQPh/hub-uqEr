"""统一数据加载与切分（三个数据集，同一套 {"text","label"} 结构）。

评测协议（三个数据集一致）：
  - train：用于训练；
  - dev：只用于选 epoch（早停），不参与训练；
  - test：只在最后评一次，用于报告。

TNEWS 的特殊处理：
  - 课程原始验证集有 11% 与训练集**完全重复**，去重后剩 8,901 条作为 test；
  - dev 从 train 里按标签分层抽 3,000 条（原数据没有干净 dev），
    因此训练集为 53,360 - 3,000 = 50,360 条。
  - 另外保留课程口径的 500 条抽样（seed=42），便于与课件数字对照。
"""

import json
import random

import pandas as pd
import pyarrow.ipc as ipc
from sklearn.model_selection import train_test_split

from cfg import COURSE_TNEWS_TRAIN, COURSE_TNEWS_VAL, DATA_DIR, DATASET_SPECS


def _norm(items, text_key="text", label_key="label"):
    return [{"text": str(d[text_key]), "label": int(d[label_key])} for d in items]


def _read_arrow(path):
    table = ipc.open_stream(str(path)).read_all()
    return table.to_pylist()


def load_tnews(dev_size=3000, seed=42):
    train = json.loads(COURSE_TNEWS_TRAIN.read_text(encoding="utf-8"))
    val = json.loads(COURSE_TNEWS_VAL.read_text(encoding="utf-8"))
    train = _norm(train, "sentence", "label")
    val = _norm(val, "sentence", "label")

    seen = {d["text"] for d in train}
    clean = [d for d in val if d["text"] not in seen]
    n_dup = len(val) - len(clean)

    labels = [d["label"] for d in train]
    tr, dev = train_test_split(train, test_size=dev_size, stratify=labels, random_state=seed)
    rng = random.Random(seed)
    test500 = rng.sample(clean, 500)

    return {
        "name": "tnews",
        "train": tr, "dev": dev, "test": clean, "test500": test500,
        "meta": {"n_train_raw": len(train), "n_val_raw": len(val), "n_dup": n_dup,
                 "dup_rate": round(n_dup / len(val), 4), "n_test": len(clean),
                 "dev_from": "train 分层抽样", "seed": seed},
    }


def load_thucnews():
    tr = _read_arrow(DATA_DIR / "thucnews" / "thuc_news_title-train.arrow")
    dev = _read_arrow(DATA_DIR / "thucnews" / "thuc_news_title-validation.arrow")
    te = _read_arrow(DATA_DIR / "thucnews" / "thuc_news_title-test.arrow")
    train, dev, test = _norm(tr), _norm(dev), _norm(te)
    # 检查训练/测试是否存在完全重复（同一标题被转载）
    seen = {d["text"] for d in train}
    n_dup = sum(1 for d in test if d["text"] in seen)
    return {
        "name": "thucnews",
        "train": train, "dev": dev, "test": test, "test500": None,
        "meta": {"n_train_raw": len(train), "n_dup": n_dup,
                 "dup_rate": round(n_dup / len(test), 4), "n_test": len(test),
                 "dev_from": "官方 validation", "seed": 42},
    }


def load_chnsenticorp():
    base = DATA_DIR / "chnsenticorp"
    train = pd.read_parquet(base / "train-00000-of-00001-02f200ca5f2a7868.parquet").to_dict("records")
    dev = pd.read_parquet(base / "validation-00000-of-00001-405befbaa3bcf1a2.parquet").to_dict("records")
    test = pd.read_parquet(base / "test-00000-of-00001-5372924f059fe767.parquet").to_dict("records")
    train, dev, test = _norm(train), _norm(dev), _norm(test)
    seen = {d["text"] for d in train}
    n_dup = sum(1 for d in test if d["text"] in seen)
    return {
        "name": "chnsenticorp",
        "train": train, "dev": dev, "test": test, "test500": None,
        "meta": {"n_train_raw": len(train), "n_dup": n_dup,
                 "dup_rate": round(n_dup / len(test), 4), "n_test": len(test),
                 "dev_from": "官方 validation", "seed": 42},
    }


LOADERS = {
    "tnews": load_tnews,
    "thucnews": load_thucnews,
    "chnsenticorp": load_chnsenticorp,
}


def load_data(name):
    return LOADERS[name]()


def dataset_stats(data):
    """给报告用的数据概况：样本量、类别数、平均长度、不均衡比。"""
    import collections
    out = {"name": data["name"], "splits": {}, "meta": data["meta"]}
    for split in ("train", "dev", "test"):
        items = data[split]
        cnt = collections.Counter(d["label"] for d in items)
        lengths = [len(d["text"]) for d in items]
        out["splits"][split] = {
            "n": len(items),
            "n_classes": len(cnt),
            "avg_len": round(sum(lengths) / len(lengths), 1),
            "max_len": max(lengths),
            "min_class": min(cnt.values()),
            "max_class": max(cnt.values()),
            "imbalance": round(max(cnt.values()) / min(cnt.values()), 1),
            "counts": dict(sorted(cnt.items())),
        }
    return out


if __name__ == "__main__":
    for name in DATASET_SPECS:
        d = load_data(name)
        print(json.dumps(dataset_stats(d), ensure_ascii=False, indent=2))
