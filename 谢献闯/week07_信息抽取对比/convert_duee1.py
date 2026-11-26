# -*- coding: utf-8 -*-
"""
把 DuEE 1.0（百度通用领域事件抽取，65 类）转换成本周项目的数据契约，
用于「换数据集」对照实验：同一套 BERT-CRF / 规则 / LLM 代码，换一个 schema 更大的领域。

源：HF dataset `nlhappy/DuEE`（train/validation 两个 parquet；许可证 MIT，见源 README）
字段：text, events[{label, trigger{text,indices}, args[{label,text,indices,is_continuous}]}]

转换规则（与 DuEE-Fin 的处理口径保持一致，坑都写在 data/README.md 里）：
  1. 一行 = 一句话 + **恰好 1 个事件**：DuEE 1.0 有 13% 的句子是多事件，
     本项目所有判别式脚本的 BioDataset 只吃单事件（row_event 取第一个），
     直接喂多事件句会静默丢事件、指标失真 → 这里显式丢弃多事件句，并在 stats 里报数。
  2. 索引换算：DuEE 的 indices 是「字符下标列表」，项目契约是 start/end（左闭右开）。
  3. 丢空触发词/空论元的样本（DuEE 1.0 实测为 0，保留检查以防换版本）。
  4. 输出 schema：{事件类型: [角色...]}，供 build_joint_schema 自动生成标签空间。

用法：
  python w7_ext/convert_duee1.py \
      --src E:/大模型学习/week7_data/duee1/data \
      --out data/duee1
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

TRAIN_FILE = "train-00000-of-00001-0c572268f3739da6.parquet"
DEV_FILE = "validation-00000-of-00001-9e6bf93485a41d0b.parquet"


def to_span(text: str, item: dict) -> dict | None:
    """{'text','indices'} → {'text','start','end'}；索引对不上原文就返回 None。"""
    raw_idx = item.get("indices")
    idx = [int(i) for i in list(raw_idx)] if raw_idx is not None else []
    s = (item.get("text") or "").strip()
    if len(idx) == 0 or not s:
        return None
    start, end = idx[0], idx[-1] + 1
    # 防御：原文切片必须与标注文本一致（不一致说明索引语义变了，宁可丢样本也不写脏数据）
    if text[start:end] != item.get("text"):
        return None
    return {"text": item["text"], "start": start, "end": end}


def convert(src: Path, split: str, fn: str, stats: Counter):
    df = pd.read_parquet(src / fn)
    rows = []
    for i, r in df.iterrows():
        text = (r["text"] or "").strip()
        evs = list(r["events"]) if r["events"] is not None else []
        stats["raw_rows"] += 1
        if not text:
            stats["drop_empty_text"] += 1
            continue
        if len(evs) == 0:
            stats["drop_no_event"] += 1
            continue
        if len(evs) > 1:
            stats["drop_multi_event"] += 1
            continue
        e = evs[0]
        trig = to_span(text, e["trigger"])
        if trig is None:
            stats["drop_bad_trigger"] += 1
            continue
        args = []
        for a in e["args"]:
            sp = to_span(text, a)
            if sp is None:
                stats["drop_bad_arg"] += 1
                continue
            args.append({"role": a["label"], **sp})
        if not args:
            stats["drop_no_arg"] += 1
            continue
        rows.append({
            "id": f"duee1_{split}_{i}",
            "sent_id": f"duee1_{split}_{i}",
            "text": text,
            "event": {"type": e["label"], "trigger": trig, "args": args},
        })
    return rows, df


def build_schema(rows: list[dict]) -> dict:
    schema: dict[str, set] = {}
    for r in rows:
        ev = r["event"]
        schema.setdefault(ev["type"], set()).update(a["role"] for a in ev["args"])
    return {k: sorted(v) for k, v in sorted(schema.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"E:\大模型学习\week7_data\duee1\data")
    ap.add_argument("--out", default="data/duee1")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    train_rows, _ = convert(src, "train", TRAIN_FILE, stats)
    stats_dev = Counter()
    dev_rows, _ = convert(src, "dev", DEV_FILE, stats_dev)

    schema = build_schema(train_rows + dev_rows)

    for name, rows in (("train_extract", train_rows), ("dev_extract", dev_rows)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "event_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    n_roles = {r for v in schema.values() for r in v}
    n_slots = sum(len(v) for v in schema.values())
    lens = sorted(len(r["text"]) for r in train_rows + dev_rows)
    info = {
        "source": "HF nlhappy/DuEE (DuEE 1.0, paper: 百度通用领域事件抽取)",
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "schema": {"types": len(schema), "slots": n_slots, "uniq_roles": len(n_roles)},
        "text_len": {"p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)],
                     "max": lens[-1]},
        "filter_stats_train": dict(stats, kept=len(train_rows)),
        "filter_stats_dev": dict(stats_dev, kept=len(dev_rows)),
    }
    (out / "convert_stats.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
