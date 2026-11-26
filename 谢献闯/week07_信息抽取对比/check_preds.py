# -*- coding: utf-8 -*-
"""预测文件体检：行数 / 唯一键 / 是否覆盖金标子集 / 事件与论元规模。
用于发现「两个进程同时写同一文件」这类污染（重复键、行数不对）。

用法：python w7_ext/check_preds.py outputs/llm/pred_w7_qwen06_zero.jsonl [...]
"""
import json
import sys
from collections import Counter
from pathlib import Path


def check(path: Path, gold: Path | None = None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    keys = [f"{r.get('id')}|{r.get('sent_id')}" for r in rows]
    c = Counter(keys)
    dups = {k: v for k, v in c.items() if v > 1}
    n_events = sum(len(r.get("events") or ([r["event"]] if r.get("event") else [])) for r in rows)
    out = {
        "file": path.name,
        "lines": len(rows),
        "unique_keys": len(c),
        "duplicate_keys": len(dups),
        "events": n_events,
    }
    if gold and gold.exists():
        g = [json.loads(l) for l in open(gold, encoding="utf-8") if l.strip()]
        gk = {f"{r.get('id')}|{r.get('sent_id')}" for r in g}
        out["gold_lines"] = len(g)
        out["missing_vs_gold"] = len(gk - set(keys))
        out["extra_vs_gold"] = len(set(keys) - gk)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"[missing] {p}")
            continue
        gold = None
        if "duee1" in str(p):
            gold = Path("outputs/llm_duee1") / f"gold_n{200}.jsonl"
        else:
            gold = Path("outputs/llm") / f"gold_n{200}.jsonl"
        print(json.dumps(check(p, gold), ensure_ascii=False))
