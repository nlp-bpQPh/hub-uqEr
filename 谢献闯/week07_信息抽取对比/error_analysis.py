# -*- coding: utf-8 -*-
"""
错误分析（CPU-only，读已落盘的预测文件）：把"事件级失败"拆成互斥的几类，
并给出类型混淆 Top-N —— 报告里用来说明"不同方法的错误形态不一样"。

口径与 common/evaluate_events.py 完全一致（直接 import 它的 as_event_list/row_key）。
产物：outputs/w7_compare/error_analysis.md / .json

用法：python w7_ext/error_analysis.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from evaluate_events import as_event_list, load_jsonl, row_key  # noqa: E402

OUT = ROOT / "outputs" / "w7_compare"

CASES = [
    # (名称, gold, pred, 数据集)
    ("BERT-CRF (DuEE-Fin)", ROOT / "data/processed/dev_extract.jsonl",
     ROOT / "outputs/disc/pred_w7_bert_crf.jsonl"),
    ("RoBERTa-CRF (DuEE-Fin)", ROOT / "data/processed/dev_extract.jsonl",
     ROOT / "outputs/disc/pred_w7_roberta_crf.jsonl"),
    ("Qwen3-0.6B SFT (DuEE-Fin)", ROOT / "data/processed/dev_extract.jsonl",
     ROOT / "outputs/llm/pred_w7_qwen06_sft.jsonl"),
    ("Qwen3-0.6B zero-shot (DuEE-Fin)", ROOT / "data/processed/dev_extract.jsonl",
     ROOT / "outputs/llm/pred_w7_qwen06_zero.jsonl"),
    ("BERT-CRF (DuEE1.0)", ROOT / "data/duee1/dev_extract.jsonl",
     ROOT / "outputs/disc/pred_w7_bert_crf_duee1.jsonl"),
]


def classify(gold_events, pred_events):
    """一条样本的事件级诊断（互斥分类，用于统计）。优先取 gold 的第一个事件。"""
    if not gold_events:
        return "gold_empty"
    if not pred_events:
        return "no_event"
    g = gold_events[0]
    # 同类型优先匹配
    same = [p for p in pred_events if p["type"] == g["type"]]
    if not same:
        return "wrong_type"
    p = same[0]
    if p["trigger"] != g["trigger"]:
        return "wrong_trigger"
    gc = Counter(g["args"])
    pc = Counter(p["args"])
    if gc == pc:
        return "exact"
    extra = sum((pc - gc).values())
    miss = sum((gc - pc).values())
    if miss and not extra:
        return "arg_missing"
    if extra and not miss:
        return "arg_spurious"
    return "arg_both"


def analyze(name, gold_path, pred_path):
    if not Path(pred_path).exists():
        return None
    gold = load_jsonl(str(gold_path))
    pred = {row_key(r): r for r in load_jsonl(str(pred_path))}
    bucket = Counter()
    confusion = Counter()
    role_miss, role_extra = Counter(), Counter()
    type_fail = Counter()
    type_total = Counter()
    for g in gold:
        ge = as_event_list(g)
        pe = as_event_list(pred.get(row_key(g), {}))
        for e in ge:
            type_total[e["type"]] += 1
        label = classify(ge, pe)
        bucket[label] += 1
        if label not in ("exact",) and ge:
            type_fail[ge[0]["type"]] += 1
        if label == "wrong_type" and pe:
            confusion[(ge[0]["type"], pe[0]["type"])] += 1
        if label in ("arg_missing", "arg_both", "arg_spurious") and ge and pe:
            gc = Counter(ge[0]["args"])
            pc = Counter(pe[0]["args"])
            for (role, txt), c in (gc - pc).items():
                role_miss[role] += c
            for (role, txt), c in (pc - gc).items():
                role_extra[role] += c
    total = sum(bucket.values())
    worst_types = sorted(((t, type_fail[t] / type_total[t], type_total[t])
                          for t in type_total if type_total[t] >= 10),
                         key=lambda x: -x[1])[:5]
    return {
        "name": name,
        "samples": total,
        "buckets": {k: [v, round(v / total, 4)] for k, v in bucket.most_common()},
        "confusions": [[list(k), v] for k, v in confusion.most_common(5)],
        "top_missing_roles": [[r, c] for r, c in role_miss.most_common(6)],
        "top_spurious_roles": [[r, c] for r, c in role_extra.most_common(6)],
        "worst_types": [[t, round(r, 3), n] for t, r, n in worst_types],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [r for r in (analyze(*c) for c in CASES) if r]
    (OUT / "error_analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["## 事件级错误画像（互斥分类，dev 全量）", ""]
    md.append("| 方法 | 完全正确 | 未检出事件 | 类型错 | 触发词错 | 论元缺失 | 论元多余 | 两者都有 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        b = r["buckets"]
        def pct(k):
            return f'{b[k][1]:.1%}' if k in b else "0.0%"
        md.append(f"| {r['name']} | {pct('exact')} | {pct('no_event')} | "
                  f"{pct('wrong_type')} | {pct('wrong_trigger')} | "
                  f"{pct('arg_missing')} | {pct('arg_spurious')} | {pct('arg_both')} |")
    md.append("")
    for r in results:
        md.append(f"### {r['name']}")
        md.append("")
        if r["confusions"]:
            md.append("- 类型混淆 Top：" + "；".join(
                f"{a} → {b}（{c} 次）" for (a, b), c in r["confusions"]))
        if r["top_missing_roles"]:
            md.append("- 漏得最多的角色：" + "；".join(
                f"{a}（{c}）" for a, c in r["top_missing_roles"]))
        if r["top_spurious_roles"]:
            md.append("- 多抽最多的角色：" + "；".join(
                f"{a}（{c}）" for a, c in r["top_spurious_roles"]))
        if r["worst_types"]:
            md.append("- 失败率最高的类型：" + "；".join(
                f"{a}（{b:.0%}，n={n}）" for a, b, n in r["worst_types"]))
        md.append("")
    (OUT / "error_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:20]))
    print(f"\n→ {OUT/'error_analysis.md'}")


if __name__ == "__main__":
    main()
