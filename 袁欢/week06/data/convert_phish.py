"""
数据转换：malicious_phish.csv → 内部契约 train.json / val.json

把 (url, type) 转成 {"idx","sentence","label"}，sentence 键继续承载 URL，
下游五路脚本的 d["sentence"] 全部零改动。

用法：
  python data/convert_phish.py                              # 数据集已在 data/ 下，无参即可
  python data/convert_phish.py --train_n 0                 # 全量训练（GPU 档）
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
from common.labels import LABELS, LABEL2ID   # 转换与五路方法共用同一张标签表


def normalize(url: str) -> str:
    """统一小写 + 截 200 字符。截断的必要性见 MIGRATION_PLAN.md 第 8 节：
    SFT 的 max_length=256，超长 URL 会把 chat 模板/答案 token 一起切掉，
    造成 loss 全被 mask 的废样本。实测仅 1.6%（10,501 条）超长 URL 受影响。"""
    return url.strip().lower()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parent / "malicious_phish.csv"))
    ap.add_argument("--train_n", type=int, default=50000, help="训练集条数，0=全量")
    ap.add_argument("--val_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    df = df.dropna(subset=["url", "type"])

    # 实测有 3 条脏行（乱码/空 type）：过滤丢弃并打印，别让个别脏数据挡住整批转换
    bad = ~df["type"].isin(LABELS)
    if bad.any():
        print(f"[warn] 丢弃 {int(bad.sum())} 条 LABELS 外的脏行:", df.loc[bad, "type"].tolist()[:5])
        df = df[~bad]

    df["url"] = df["url"].map(normalize)
    df = df.drop_duplicates(subset=["url", "type"])          # 实测 10,132 条重复（1.6%）：train/val 泄漏从源头掐掉
    # 可选更干净的口径：按 url 单键去重（实测仅 6 条同 URL 不同标签的矛盾样本）
    # df = df.drop_duplicates(subset=["url"])

    total = len(df)
    train_rows, val_rows = [], []
    for label in LABELS:                                      # 分层切分：每类内部独立按比例分
        rows = df[df["type"] == label].sample(frac=1.0, random_state=args.seed).to_dict("records")
        n_val = max(1, round(len(rows) * args.val_n / total))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed).shuffle(val_rows)

    if 0 < args.train_n < len(train_rows):                    # GPU 不在身边时的抽样档
        by_type = {}
        for r in train_rows:
            by_type.setdefault(r["type"], []).append(r)
        keep = []
        for rs in by_type.values():
            keep.extend(rs[:round(len(rs) * args.train_n / len(train_rows))])
        train_rows = keep

    def to_records(rows, start):
        return [{"idx": start + i, "sentence": r["url"], "label": LABEL2ID[r["type"]]}
                for i, r in enumerate(rows)]

    tr, va = to_records(train_rows, 0), to_records(val_rows, len(train_rows))
    (CODE_ROOT / "data" / "train.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    (CODE_ROOT / "data" / "val.json").write_text(json.dumps(va, ensure_ascii=False), encoding="utf-8")
    print(f"train={len(tr)} val={len(va)}")
    print("train 各类:", Counter(LABELS[r["label"]] for r in tr))
    print("val   各类:", Counter(LABELS[r["label"]] for r in va))


if __name__ == "__main__":
    main()
