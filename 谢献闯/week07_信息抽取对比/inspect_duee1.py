# -*- coding: utf-8 -*-
"""DuEE 1.0 parquet 结构体检（只读，不改数据）。输出写到 outputs/runlogs/duee1_stats.json"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

SRC = Path(r"E:\大模型学习\week7_data\duee1\data")
OUT = Path(__file__).resolve().parents[1] / "outputs" / "runlogs" / "duee1_stats.json"


def load(split_file):
    df = pd.read_parquet(SRC / split_file)
    rows = []
    for _, r in df.iterrows():
        evs = list(r["events"]) if r["events"] is not None else []
        rows.append({"text": r["text"], "events": evs})
    return rows


def analyze(rows, name):
    n = len(rows)
    ev_per_row = Counter(len(r["events"]) for r in rows)
    types, roles = Counter(), Counter()
    n_args = 0
    n_disc = 0           # 不连续论元
    bad_align = 0        # 索引对不上
    bad_trig = 0
    lens = []
    empty_text = 0
    for r in rows:
        t = r["text"] or ""
        lens.append(len(t))
        if not t.strip():
            empty_text += 1
        for e in r["events"]:
            types[e["label"]] += 1
            tr = e["trigger"]
            if not tr["text"]:
                bad_trig += 1
            else:
                idx = list(tr["indices"])
                if idx and t[idx[0]:idx[-1] + 1] != tr["text"]:
                    bad_trig += 1
            for a in e["args"]:
                n_args += 1
                roles[a["label"]] += 1
                idx = list(a["indices"])
                if not a.get("is_continuous", True):
                    n_disc += 1
                elif idx and t[idx[0]:idx[-1] + 1] != a["text"]:
                    bad_align += 1
    single = sum(v for k, v in ev_per_row.items() if k <= 1)
    return {
        "split": name,
        "rows": n,
        "empty_text": empty_text,
        "events_per_row": dict(sorted(ev_per_row.items())),
        "single_event_rows": single,
        "single_event_ratio": round(single / n, 4) if n else 0,
        "num_types": len(types),
        "num_roles": len(roles),
        "num_args": n_args,
        "discontinuous_args": n_disc,
        "bad_align_args": bad_align,
        "bad_trigger": bad_trig,
        "text_len": {
            "mean": round(sum(lens) / n, 1),
            "p50": int(pd.Series(lens).quantile(0.5)),
            "p90": int(pd.Series(lens).quantile(0.9)),
            "p99": int(pd.Series(lens).quantile(0.99)),
            "max": int(max(lens)),
        },
        "top_types": types.most_common(10),
    }


def main():
    train = load("train-00000-of-00001-0c572268f3739da6.parquet")
    dev = load("validation-00000-of-00001-9e6bf93485a41d0b.parquet")
    out = {"train": analyze(train, "train"), "dev": analyze(dev, "dev")}
    # schema（从 train+dev 合并）
    schema = {}
    for rows in (train, dev):
        for r in rows:
            for e in r["events"]:
                schema.setdefault(e["label"], set()).update(a["label"] for a in e["args"])
    out["schema"] = {k: sorted(v) for k, v in sorted(schema.items())}
    out["schema_stats"] = {"types": len(schema), "slots": sum(len(v) for v in schema.values()),
                           "uniq_roles": len({r for v in schema.values() for r in v})}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("schema_stats",)}, ensure_ascii=False))
    for sp in ("train", "dev"):
        d = out[sp]
        print(sp, {k: d[k] for k in ("rows", "single_event_rows", "single_event_ratio",
                                     "num_types", "num_roles", "num_args",
                                     "discontinuous_args", "bad_align_args", "bad_trigger")})
        print("  events_per_row:", d["events_per_row"])
        print("  text_len:", d["text_len"])


if __name__ == "__main__":
    main()
