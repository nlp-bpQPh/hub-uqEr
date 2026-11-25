"""
生成学生课件包用的小样本数据（data/sample/），让整套流程在没下载全量数据时也能跑通

为什么需要：
  CCED 全量 processed 在 data/processed/，CCED-main 原始仓库在 data/CCED-main/。
  学生拿到课件包后如果第一步就是「去找数据」，很容易卡在环境/网络上直接放弃。
  所以课件包只发一份**确定性抽样**的小样本（几个 MB），跑通全流程没问题；
  想复现 PPT 上的完整数字，直接用 data/processed/ 即可（脚本默认读它）。

抽样是**确定性**的（固定 seed + 先按 (id, sent_id) 排序再洗牌）：
  同样的 seed 在任何机器上都得到同一批样本，PPT 上的数字才可复现、学生和老师的结果才对得上。

口径说明（2026-11-24 换装 CCED 起）：无论元/无负样本，不再产出 detect 抽样文件。

使用方式：
  python tools/make_sample_data.py                  # 从 data/processed 生成 data/sample
  python tools/make_sample_data.py --from_sample    # 不重新抽样，只校验已有的 sample 并打印统计
  python tools/make_sample_data.py --scale 0.5      # 只跑通流程时可以用更小的规模
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from event_data import load_jsonl, save_jsonl, row_event

PROCESSED = ROOT / "data" / "processed"
SAMPLE = ROOT / "data" / "sample"
SEED = 42

# 每个文件抽多少条（学生包默认规模；换装 CCED 后无 detect 文件）
DEFAULT_SIZES = {
    "train_extract": 800,
    "dev_extract": 300,
    "dev_docs": 50,
}

def stable_shuffle(rows: list[dict], seed: int = SEED) -> list[dict]:
    """
    确定性洗牌：先按 (id, sent_id) 排序，再 shuffle。
    不排序直接 shuffle 的结果依赖原始文件行序 —— 换个版本的数据文件就抽到别的样本了。
    """
    rows = sorted(rows, key=lambda r: (str(r.get("id", "")), str(r.get("sent_id", ""))))
    random.Random(seed).shuffle(rows)
    return rows

def type_dist(rows: list[dict]) -> Counter:
    return Counter((row_event(r) or {}).get("type", "") for r in rows if row_event(r))

def make_sample(sizes: dict[str, int], seed: int = SEED) -> None:
    SAMPLE.mkdir(parents=True, exist_ok=True)
    print(f"从 {PROCESSED} 抽样 → {SAMPLE}\n")

    # schema 原样复制（标签体系必须和全量完全一致，否则标签 id 对不上）
    schema = json.loads((PROCESSED / "event_schema.json").read_text(encoding="utf-8"))
    (SAMPLE / "event_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'event_schema.json':<24} 原样复制（9 类事件定义）")

    stats = {}
    for name, n in sizes.items():
        src = PROCESSED / f"{name}.jsonl"
        if not src.exists():
            print(f"{name:<24} [跳过] 全量文件不存在")
            continue
        rows = load_jsonl(src)
        picked = stable_shuffle(rows, seed)[:n]
        picked = sorted(picked, key=lambda r: (str(r.get("id", "")),
                                               str(r.get("sent_id", ""))))  # 写盘也保持稳定顺序
        save_jsonl(picked, SAMPLE / f"{name}.jsonl")
        stats[name] = {"full": len(rows), "sample": len(picked)}
        print(f"{name:<24} 全量 {len(rows):>6} → 抽样 {len(picked):>5}")

    # ── 打印统计，确认抽样没改变数据分布 ──
    print("\n【抽样后分布校验】")
    full_tr = type_dist(load_jsonl(PROCESSED / "train_extract.jsonl"))
    samp_tr = type_dist(load_jsonl(SAMPLE / "train_extract.jsonl"))
    print("  事件类型分布（全量% → 抽样%）：")
    for t in sorted(full_tr, key=lambda x: -full_tr[x]):
        fp = full_tr[t] / max(sum(full_tr.values()), 1)
        sp = samp_tr.get(t, 0) / max(sum(samp_tr.values()), 1)
        print(f"    {t:<8} {fp:>6.1%} → {sp:>6.1%}")

    print(f"\n  seed={seed}（固定）；样本合计 "
          f"{sum(v['sample'] for v in stats.values())} 条")

    (SAMPLE / "sample_stats.json").write_text(
        json.dumps({"seed": seed, "sizes": sizes, "stats": stats},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  统计已写入 {(SAMPLE / 'sample_stats.json').relative_to(ROOT)}")

def check_sample() -> None:
    """--from_sample：不重新抽样，只校验 sample 完整并打印「下一步怎么跑」。"""
    if not SAMPLE.exists():
        print(f"没有找到 {SAMPLE}，请先运行：python tools/make_sample_data.py")
        return
    print(f"样本目录：{SAMPLE}\n")
    ok = True
    for name in list(DEFAULT_SIZES) + ["event_schema.json"]:
        p = SAMPLE / name if name.endswith(".json") else SAMPLE / f"{name}.jsonl"
        if not p.exists():
            print(f"  [缺失] {p.name}")
            ok = False
            continue
        if p.suffix == ".jsonl":
            rows = load_jsonl(p)
            print(f"  {p.name:<24} {len(rows):>5} 条")
        else:
            schema = json.loads(p.read_text(encoding="utf-8"))
            n_types = len([k for k in schema if not k.startswith("__")])
            print(f"  {p.name:<24} {n_types} 个事件类型")
    print("\n用样本数据跑通全流程（不需要全量数据）：")
    print("  python bert/train.py --data_dir data/sample --device cpu --epochs 1 --bert_layers 4")
    print("  python eval/compare_methods.py --data_dir data/sample   # 汇总对比表")
    print("\n想复现 PPT 上的完整数字：把全量 processed 放回 data/processed/ 即可（脚本自动优先用全量）。")
    if not ok:
        print("\n[注意] 上面有缺失文件，流程可能跑不通")

def main():
    ap = argparse.ArgumentParser(description="生成学生包用的小样本数据")
    ap.add_argument("--from_sample", action="store_true",
                    help="不重新抽样，只校验已有 sample 并打印统计")
    ap.add_argument("--scale", type=float, default=1.0, help="抽样规模缩放（0.5 = 减半）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.from_sample:
        check_sample()
        return
    if not (PROCESSED / "train_extract.jsonl").exists():
        print(f"找不到全量数据 {PROCESSED / 'train_extract.jsonl'}\n"
              f"请先把 data/processed/ 准备好，或直接用 --from_sample 校验已有样本。")
        return
    sizes = {k: max(1, int(v * args.scale)) for k, v in DEFAULT_SIZES.items()}
    make_sample(sizes, args.seed)

if __name__ == "__main__":
    main()
