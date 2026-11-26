# -*- coding: utf-8 -*-
"""两个数据集的对比统计（写报告「数据」一节用）：规模 / 事件密度 / 文本长度 / 类型分布。"""
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "w7_compare" / "dataset_stats.json"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def stats(name, train_p, dev_p, schema_p):
    tr, dv = load(train_p), load(dev_p)
    sch = json.loads(Path(schema_p).read_text(encoding="utf-8"))
    sch = {k: v for k, v in sch.items() if not k.startswith("__")}

    def one(rows):
        n_args = sum(len(r["event"]["args"]) for r in rows)
        lens = [len(r["text"]) for r in rows]
        return {
            "samples": len(rows),
            "args": n_args,
            "args_per_event": round(n_args / len(rows), 2),
            "text_len_mean": round(statistics.mean(lens), 1),
            "text_len_p95": sorted(lens)[int(len(lens) * 0.95)],
            "types_present": len({r["event"]["type"] for r in rows}),
        }

    dev_stats = one(dv)
    type_counts = Counter(r["event"]["type"] for r in dv)
    return {
        "dataset": name,
        "schema": {"types": len(sch),
                   "slots": sum(len(v) for v in sch.values()),
                   "uniq_roles": len({r for v in sch.values() for r in v})},
        "train": one(tr),
        "dev": dev_stats,
        "dev_type_top5": type_counts.most_common(5),
        "dev_type_bottom5": type_counts.most_common()[-5:],
    }


def main():
    out = [
        stats("DuEE-Fin",
              ROOT / "data/processed/train_extract.jsonl",
              ROOT / "data/processed/dev_extract.jsonl",
              ROOT / "data/processed/event_schema.json"),
        stats("DuEE1.0",
              ROOT / "data/duee1/train_extract.jsonl",
              ROOT / "data/duee1/dev_extract.jsonl",
              ROOT / "data/duee1/event_schema.json"),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in out:
        print(r["dataset"], r["schema"], "train", r["train"], "dev", r["dev"])


if __name__ == "__main__":
    main()
