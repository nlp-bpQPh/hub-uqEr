# -*- coding: utf-8 -*-
"""
把训练集按类型分层抽到固定规模（CPU 预算对齐用），并备份原全量训练集。

背景：DuEE 1.0 单事件训练集 10,354 句；在 8 核 CPU 上 4 层 BERT 一个 epoch 约 43 分钟，
      线性 + CRF 两个模型就要 1.5 小时。为了让「所有学习方法」在**同一训练预算**下比较，
      统一抽 6,000 句（按事件类型分层，seed=42）：

        python data/scripts/subsample_train.py --n 6000

产出：
  train_extract_full.jsonl   原全量（10,354，保留备查）
  train_extract.jsonl        分层抽样后的 6,000（训练脚本默认读它）
  stats/train_subsample.json 抽样元信息（每类型抽了多少）
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA
    path = data_dir / "train_extract.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    full_path = data_dir / "train_extract_full.jsonl"
    if not full_path.exists():
        full_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                             encoding="utf-8")
        print(f"已备份全量训练集 → {full_path.name}（{len(rows)} 句）")
    else:
        rows = [json.loads(line) for line in open(full_path, encoding="utf-8") if line.strip()]

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["event"]["type"]].append(r)

    rng = random.Random(args.seed)
    for t in by_type:
        rng.shuffle(by_type[t])

    # 按各类型样本数比例分配名额（最大余数法），再按轮转补齐/截断到恰好 n
    total = len(rows)
    quota = {t: args.n * len(rs) / total for t, rs in by_type.items()}
    alloc = {t: int(q) for t, q in quota.items()}
    rest = args.n - sum(alloc.values())
    for t, _ in sorted(quota.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:rest]:
        alloc[t] += 1

    picked = []
    for t, rs in by_type.items():
        picked.extend(rs[: min(alloc[t], len(rs))])
    # 名额可能因类型样本不足而少，用剩余样本补齐
    if len(picked) < args.n:
        used = {id(r) for r in picked}
        extra = [r for r in rows if id(r) not in used]
        rng.shuffle(extra)
        picked.extend(extra[: args.n - len(picked)])
    rng.shuffle(picked)
    picked = picked[: args.n]

    with open(path, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "n": len(picked), "seed": args.seed, "full_n": total,
        "by_type": dict(Counter(r["event"]["type"] for r in picked).most_common()),
        "note": "按类型分层抽样；所有学习方法共用这一份训练集（CPU 预算对齐）",
    }
    (data_dir.parent / "stats").mkdir(parents=True, exist_ok=True)
    with open(data_dir.parent / "stats" / "train_subsample.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"训练集：{total} → {len(picked)} 句（{len(stats['by_type'])} 类）")
    print(json.dumps(stats["by_type"], ensure_ascii=False, indent=1)[:1200])


if __name__ == "__main__":
    main()
