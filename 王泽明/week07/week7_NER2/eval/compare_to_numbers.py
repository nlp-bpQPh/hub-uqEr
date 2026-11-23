# -*- coding: utf-8 -*-
"""
把当前 outputs/metrics 里的数字与「上一轮实测值」逐项对比（回归基线检查）。

为什么需要这个脚本（而不是人肉对表）：
  重构 + 重跑之后，最需要回答的问题是「**数字有没有变**」。
  人肉对 20 多行 × 4 个指标非常容易看漏，所以把基线值固化成一个字典、逐项算差，
  并把超过阈值（默认 0.5pp）的项显式标红 —— 这正是「重构不能改变数值行为」的验收依据。

基线的来源：2026-11-23 把编码器换成 chinese-roberta-wwm-ext 后的全量重训结果
（`outputs/metrics/disc/summary.md`）。以后如果又重训并确认新数字是准的，
把下面的 OLD 字典同步改掉即可；只要预测文件没重跑、指标没变，本脚本应全绿。

用法：
  python eval/compare_to_numbers.py                # 打印对比表
  python eval/compare_to_numbers.py --threshold 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")   # GBK 控制台打印 ✅/⚠ 不崩
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
METRIC_DISC = ROOT / "outputs" / "metrics" / "disc"
METRIC_LLM = ROOT / "outputs" / "metrics" / "llm"

# 基线值（2026-11-23 RoBERTa 重训实测）：method -> (trigger, argument, event)
OLD = {
    "rule":                 (0.6168, 0.1740, 0.0161),
    "bilstm_crf":           (0.8165, 0.4634, 0.0885),
    "roberta_linear_256":   (0.8192, 0.5620, 0.1704),
    "roberta_crf_256":      (0.8105, 0.5755, 0.1720),
    "roberta_linear_128":   (0.8177, 0.5234, 0.1440),
    "roberta_crf_256_hard": (0.8356, 0.6620, 0.2210),
    "pipeline":             (0.6685, 0.4542, 0.1273),
    "api_zero_n200":        (0.8212, 0.6249, 0.1008),
    "api_few_n200":         (0.8283, 0.6599, 0.1010),
    "sft_lora_n200":        (0.8800, 0.6367, 0.1200),   # LoRA SFT 的 n=200 结果文件就叫这个名字
}

# 基线里这些是 n=200 口径；其余是 dev 930 全量口径
N200 = {"api_zero_n200", "api_few_n200", "sft_lora_n200"}


def load_pred(method: str):
    for d in (METRIC_DISC, METRIC_LLM):
        p = d / f"{method}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    # LoRA 评测脚本输出的 tag 可能带后缀
    for p in sorted(METRIC_LLM.glob("*.json")):
        if method in p.stem:
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    ap = argparse.ArgumentParser(description="当前指标 vs RoBERTa 重训基线值")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="超过多少个百分点标 ⚠（默认 0.5pp）")
    args = ap.parse_args()

    print(f"{'方法':<22}{'口径':<8}{'trigger':>18}{'argument':>18}{'event':>18}")
    print("-" * 86)
    flagged = []
    for method, (ot, oa, oe) in OLD.items():
        r = load_pred(method)
        if r is None:
            print(f"{method:<22}{'(未跑)':<8}")
            continue
        scope = "n=200" if method in N200 else "dev930"
        cur = (r["trigger"]["f1"], r["argument"]["f1"], r["event_exact"]["f1"])
        cells = []
        for o, c in zip((ot, oa, oe), cur):
            d = (c - o) * 100
            mark = " ⚠" if abs(d) > args.threshold else ""
            cells.append(f"{o:.4f}→{c:.4f} ({d:+.2f}pp){mark}")
            if abs(d) > args.threshold:
                flagged.append((method, scope, d))
        print(f"{method:<22}{scope:<8}" + "".join(f"{c:>18}" for c in cells))

    print("-" * 86)
    if flagged:
        print(f"\n⚠ 超出 ±{args.threshold}pp 的项（需人工确认是否是 n=200 的采样噪声）：")
        for m, s, d in flagged:
            print(f"  - {m}（{s}）：{d:+.2f}pp")
    else:
        print(f"\n✅ 没有任何一项偏差超过 ±{args.threshold}pp —— 重构未改变数值行为。")
    print("\n注：n=200 口径下 1 条样本 ≈ 0.5pp，所以该口径的 ±0.5pp 波动属采样噪声。")


if __name__ == "__main__":
    main()
