# -*- coding: utf-8 -*-
"""
把 DuEE 1.0（2020 千言事件抽取竞赛，65 类）转换成本项目的 JSONL 契约。

输入（zip 内）：
  conf/DuEE1.0/duee_event_schema.json   65 个事件类型 + 角色表（JSONL）
  data/DuEE1.0/duee_train.json          11,908 句 / 13,860 事件（JSONL）
  data/DuEE1.0/duee_dev.json             1,492 句 /  1,783 事件（JSONL）

输出（code/data/processed/）：
  event_schema.json     {"事件类型": [角色, ...], "__meta__": {...}}
  train_extract.jsonl   单事件句（与课程 DuEE-Fin 口径一致：一行 = 一句 = 恰好 1 个事件）
  dev_extract.jsonl     单事件句（主评测集）
  dev_multi.jsonl       多事件句（只做案例分析，不参与主指标）
  stats/duee10_stats.json  原始/保留/修复/分布统计

为什么只保留「单事件句」做主实验：
  课程与评测脚本（common/evaluate_events.py 的 compare_events）按「一句一事件」设计；
  DuEE 1.0 有 13.1% 的句子含 2+ 个事件，直接混入会让「一句抽多个事件」的方法吃亏。
  所以主评测集对齐课程口径（单事件句），多事件句单独存一份做案例分析 —— 这一点会在报告里写明。

offset 修复：
  原始标注自带 trigger_start_index / argument_start_index。实测 99.99% 可用，
  但仍有少量错位（本项目全量校验：trigger 0 错、argument 1 错）。转换时统一
  用「先信任标注下标，回读不一致则在附近/全句回找」的策略，并把修复次数写进 stats。
"""

import argparse
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
CODE_DIR = HERE.parents[2]          # .../code
PROJECT_DIR = CODE_DIR.parent      # .../hw_duee10
ZIP_PATH = PROJECT_DIR / "data" / "raw" / "DuEE_pytorch.zip"
OUT_DIR = CODE_DIR / "data" / "processed"
STATS_DIR = CODE_DIR / "data" / "stats"
ZIP_BASE = "DuEE_pytorch-master/1-DuEE_BERT/"

# 主实验保留的事件类型数：0 = 全部 65 类；15 = 按 train 事件数取前 15 类（本项目主实验）
MAX_TYPES = 0


def read_jsonl_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    text = zf.read(ZIP_BASE + name).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def locate(text: str, surface: str, hint, used: list[tuple[int, int]] | None = None) -> tuple[int, int] | None:
    """
    在原文中定位一个 span：先信任标注下标，失败再就近回找。

    论元允许重叠（同一段文本可以同时是两个角色；评测按 (角色,文本) 多元组比，不看 offset），
    所以这里**不**强制避开已占用位置 —— 强行避开会把「同一个『中国』同时是地点和主体」
    这类合法标注丢掉（本项目实测丢掉 2% 的论元，全是这么来的）。
    """
    if not surface:
        return None
    n = len(text)

    def ok(s):
        return 0 <= s <= n - len(surface) and text[s:s + len(surface)] == surface

    if isinstance(hint, int) and ok(hint):
        return hint, hint + len(surface)

    positions = []
    start = 0
    while True:
        pos = text.find(surface, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + 1
    if not positions:
        return None
    ref = hint if isinstance(hint, int) and hint >= 0 else 0
    positions.sort(key=lambda p: (abs(p - ref), p))
    pos = positions[0]
    return pos, pos + len(surface)


def convert_event(text: str, ev: dict, stats: Counter) -> dict | None:
    etype = ev.get("event_type", "")
    trig = ev.get("trigger", "") or ""
    trig_span = locate(text, trig, ev.get("trigger_start_index"), [])
    if trig_span is None:
        stats["drop_event_no_trigger"] += 1
        return None
    if trig_span[0] != ev.get("trigger_start_index"):
        stats["repair_trigger_index"] += 1

    used = [trig_span]
    args = []
    for a in ev.get("arguments") or []:
        role = a.get("role", "") or ""
        surface = a.get("argument", "") or ""
        span = locate(text, surface, a.get("argument_start_index"), used)
        if span is None:
            stats["drop_argument_no_span"] += 1
            continue
        if span[0] != a.get("argument_start_index"):
            stats["repair_argument_index"] += 1
        used.append(span)
        args.append({"role": role, "text": surface, "start": span[0], "end": span[1]})

    return {
        "type": etype,
        "trigger": {"text": trig, "start": trig_span[0], "end": trig_span[1]},
        "args": args,
    }


def build_split(raw_rows: list[dict], split: str, stats: Counter):
    single, multi = [], []
    for idx, row in enumerate(raw_rows):
        text = row.get("text", "")
        if not text:
            stats[f"{split}_empty_text"] += 1
            continue
        events = []
        for ev in row.get("event_list") or []:
            conv = convert_event(text, ev, stats)
            if conv is not None:
                events.append(conv)
        if not events:
            stats[f"{split}_drop_sentence"] += 1
            continue
        record = {"id": row.get("id") or f"{split}-{idx}", "sent_id": f"s{idx}",
                  "text": text, "events": events}
        if len(events) == 1:
            single.append({"id": record["id"], "sent_id": record["sent_id"], "text": text,
                           "event": events[0]})
        else:
            multi.append(record)
    return single, multi


def percentile(values, p):
    vals = sorted(values)
    return vals[min(len(vals) - 1, int(len(vals) * p))] if vals else 0


def main():
    global MAX_TYPES, OUT_DIR
    ap = argparse.ArgumentParser(description="DuEE 1.0 → 项目 JSONL 契约")
    ap.add_argument("--max_types", type=int, default=0,
                    help="只保留样本量前 N 的事件类型；0=全部 65 类")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 code/data/processed）")
    ap.add_argument("--stats_name", default="duee10_stats.json")
    args = ap.parse_args()
    MAX_TYPES = args.max_types
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    assert ZIP_PATH.exists(), f"缺少原始数据：{ZIP_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        schema_lines = read_jsonl_from_zip(zf, "conf/DuEE1.0/duee_event_schema.json")
        raw_train = read_jsonl_from_zip(zf, "data/DuEE1.0/duee_train.json")
        raw_dev = read_jsonl_from_zip(zf, "data/DuEE1.0/duee_dev.json")

    schema = {}
    for item in schema_lines:
        etype = item["event_type"]
        roles = [r["role"] for r in item.get("role_list") or []]
        schema[etype] = list(dict.fromkeys(roles))

    stats = Counter()
    train_single, train_multi = build_split(raw_train, "train", stats)
    dev_single, dev_multi = build_split(raw_dev, "dev", stats)

    # 类型选择（默认全量；MAX_TYPES 可用来做小规模冒烟）
    if MAX_TYPES:
        order = [t for t, _ in Counter(ev["type"] for r in train_single for ev in [r["event"]]).most_common()]
        keep = set(order[:MAX_TYPES])
        schema = {k: v for k, v in schema.items() if k in keep}
        train_single = [r for r in train_single if r["event"]["type"] in keep]
        dev_single = [r for r in dev_single if r["event"]["type"] in keep]
        dev_multi = [r for r in dev_multi if any(ev["type"] in keep for ev in r["events"])]
        stats["max_types"] = MAX_TYPES

    # 只保留 schema 里定义过的类型/角色（防止脏标注带出未定义标签）
    def clean(rows):
        out = []
        for r in rows:
            ev = r.get("event") or (r.get("events") or [None])[0]
            if not ev or ev["type"] not in schema:
                stats["drop_unknown_type"] += 1
                continue
            roles = set(schema[ev["type"]])
            kept = []
            for a in ev.get("args") or []:
                if a["role"] in roles:
                    kept.append(a)
                else:
                    stats["drop_role_not_in_schema"] += 1
            ev["args"] = kept
            out.append(r)
        return out

    train_single, dev_single = clean(train_single), clean(dev_single)

    total_raw_events = sum(len(r.get("event_list") or []) for r in raw_train) + \
                       sum(len(r.get("event_list") or []) for r in raw_dev)
    kept_events = len(train_single) + len(dev_single)
    coverage = kept_events / max(total_raw_events, 1)

    # 顺序固定（评测/抽样可复现）：train 按原顺序，dev 按原顺序
    for split, rows in (("train", train_single), ("dev", dev_single)):
        path = OUT_DIR / f"{split}_extract.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "dev_multi.jsonl", "w", encoding="utf-8") as f:
        for r in dev_multi:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    enums = {}
    meta = {
        "name": "DuEE 1.0",
        "source": "P100080/DuEE_pytorch (github) / 2020 千言中文事件抽取竞赛",
        "note": "一行 = 一句 = 恰好 1 个事件（与课程 DuEE-Fin 口径一致）；多事件句见 dev_multi.jsonl",
        "num_types": len(schema),
        "num_type_role_slots": sum(len(v) for v in schema.values()),
        "num_union_roles": len({r for v in schema.values() for r in v}),
        "max_types": MAX_TYPES or 65,
        "event_coverage": round(coverage, 4),
        "kept_events": kept_events,
        "all_events": total_raw_events,
    }
    with open(OUT_DIR / "event_schema.json", "w", encoding="utf-8") as f:
        json.dump({**schema, "__meta__": meta, "__enums__": enums}, f, ensure_ascii=False, indent=2)

    # 统计
    def split_stats(rows, raw_rows):
        lenses = [len(r["text"]) for r in rows]
        tc = Counter(r["event"]["type"] for r in rows)
        role_c = Counter(a["role"] for r in rows for a in r["event"]["args"])
        n_args = [len(r["event"]["args"]) for r in rows]
        return {
            "raw_sentences": len(raw_rows),
            "raw_events": sum(len(r.get("event_list") or []) for r in raw_rows),
            "kept_sentences_single_event": len(rows),
            "kept_events": len(rows),
            "mean_len": round(sum(lenses) / max(len(lenses), 1), 1),
            "p50_len": percentile(lenses, 0.5), "p95_len": percentile(lenses, 0.95),
            "max_len": max(lenses) if lenses else 0,
            "mean_args_per_event": round(sum(n_args) / max(len(n_args), 1), 2),
            "num_types_present": len(tc),
            "type_counts": dict(tc.most_common()),
            "top_roles": dict(role_c.most_common(20)),
        }

    train_texts = {r["text"] for r in train_single}
    dev_texts = [r["text"] for r in dev_single]
    overlap = sum(1 for t in dev_texts if t in train_texts)
    all_stats = {
        "dataset": meta,
        "raw": {"train_sentences": len(raw_train), "dev_sentences": len(raw_dev),
                "train_events": sum(len(r.get("event_list") or []) for r in raw_train),
                "dev_events": sum(len(r.get("event_list") or []) for r in raw_dev),
                "train_multi_event_sentences": sum(1 for r in raw_train if len(r.get("event_list") or []) > 1),
                "dev_multi_event_sentences": sum(1 for r in raw_dev if len(r.get("event_list") or []) > 1)},
        "train": split_stats(train_single, raw_train),
        "dev": split_stats(dev_single, raw_dev),
        "dev_multi_sentences": len(dev_multi),
        "dev_text_overlap_train": overlap,
        "dev_text_overlap_ratio": round(overlap / max(len(dev_single), 1), 4),
        "drop_argument_no_span": stats["drop_argument_no_span"],
        "drop_role_not_in_schema": stats["drop_role_not_in_schema"],
        "repair_trigger_index": stats["repair_trigger_index"],
        "repair_argument_index": stats["repair_argument_index"],
        "drop_event_no_trigger": stats["drop_event_no_trigger"],
        "drop_sentence": stats["train_drop_sentence"] + stats["dev_drop_sentence"],
        "drop_unknown_type": stats["drop_unknown_type"],
        "label_space_joint": 1 + 2 * (len(schema) + len({r for v in schema.values() for r in v})),
    }
    with open(STATS_DIR / args.stats_name, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # 转换后硬校验：每个 span 都必须能回原文取到同样的字
    bad = 0
    for split in ("train", "dev"):
        for r in (train_single if split == "train" else dev_single):
            t = r["text"]
            ev = r["event"]
            for sp in [ev["trigger"]] + ev["args"]:
                if t[sp["start"]:sp["end"]] != sp["text"]:
                    bad += 1
    all_stats["span_check_failures"] = bad
    with open(STATS_DIR / args.stats_name, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in all_stats.items() if k not in ("train", "dev", "dataset")},
                     ensure_ascii=False, indent=2))
    print("train:", {k: all_stats["train"][k] for k in
                     ("kept_sentences_single_event", "mean_len", "p95_len", "mean_args_per_event",
                      "num_types_present")})
    print("dev  :", {k: all_stats["dev"][k] for k in
                     ("kept_sentences_single_event", "mean_len", "p95_len", "mean_args_per_event",
                      "num_types_present")})
    print("span check failures:", bad)
    print("schema:", meta)


if __name__ == "__main__":
    main()
