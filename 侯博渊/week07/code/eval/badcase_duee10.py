# -*- coding: utf-8 -*-
"""
Bad case 归类：把某个方法的 dev 预测与金标逐条对比，输出 10 条可写进报告的案例。

归类口径（与 rule/run_rule.py 的失败分析一致，互斥）：
  no_trigger    没抽到触发词 → 整句漏抽
  wrong_type    触发词命中但事件类型判错
  wrong_trigger 类型对、触发词文本不对
  arg_missing   类型/触发词对，但论元漏抽
  arg_spurious  类型/触发词对，但论元多抽
  exact         完全正确

用法：
  python eval/badcase_duee10.py --pred outputs/disc/pred_bert4L_crf_hard.jsonl \
      --gold data/processed/dev_extract.jsonl --out outputs/report/badcases_bert4L_crf_hard.md
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from evaluate_events import row_key  # noqa: E402


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def classify(g, p):
    ge = g.get("event") or {}
    pe = (p.get("events") or [{}])[0] if p.get("events") else {}
    g_trig = (ge.get("trigger") or {}).get("text", "")
    p_trig = (pe.get("trigger") or {}).get("text", "")
    g_args = {(a["role"], a["text"]) for a in ge.get("args") or []}
    p_args = {(a["role"], a["text"]) for a in pe.get("args") or []}
    if not p_trig:
        return "no_trigger", g_trig, p_trig, g_args, p_args
    if pe.get("type") != ge.get("type"):
        return "wrong_type", g_trig, p_trig, g_args, p_args
    if p_trig != g_trig:
        return "wrong_trigger", g_trig, p_trig, g_args, p_args
    if p_args == g_args:
        return "exact", g_trig, p_trig, g_args, p_args
    if g_args - p_args and not (p_args - g_args):
        return "arg_missing", g_trig, p_trig, g_args, p_args
    return ("arg_spurious" if p_args - g_args else "arg_missing"), g_trig, p_trig, g_args, p_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default=str(ROOT / "data" / "processed" / "dev_extract.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report" / "badcases.md"))
    ap.add_argument("--top_n", type=int, default=3, help="每类留几个例子")
    args = ap.parse_args()

    gold = load(args.gold)
    preds = {row_key(p): p for p in load(args.pred)}
    cats = Counter()
    examples = defaultdict(list)
    for g in gold:
        p = preds.get(row_key(g), {"events": []})
        cat, gt, pt, ga, pa = classify(g, p)
        cats[cat] += 1
        if cat != "exact" and len(examples[cat]) < args.top_n:
            examples[cat].append({
                "text": g["text"],
                "gold": {"type": (g.get("event") or {}).get("type"), "trigger": gt,
                         "args": sorted([list(x) for x in ga])},
                "pred": {"type": (p.get("events") or [{}])[0].get("type", ""), "trigger": pt,
                         "args": sorted([list(x) for x in pa])},
            })

    total = max(len(gold), 1)
    lines = [f"# Bad case 分析：{Path(args.pred).name}", "",
             f"评测样本 {len(gold)} 条；预测文件 {Path(args.pred).name}", "",
             "| 错误类型 | 条数 | 占比 |", "|---|---|---|"]
    for k, v in cats.most_common():
        lines.append(f"| {k} | {v} | {v / total:.1%} |")
    lines.append("")
    for cat in ("no_trigger", "wrong_type", "wrong_trigger", "arg_missing", "arg_spurious"):
        if not examples.get(cat):
            continue
        lines += [f"## {cat}（{cats[cat]} 条）", ""]
        for i, ex in enumerate(examples[cat], 1):
            lines += [f"**例 {i}**：{ex['text']}", "",
                      f"- 金标：`{json.dumps(ex['gold'], ensure_ascii=False)}`",
                      f"- 预测：`{json.dumps(ex['pred'], ensure_ascii=False)}`", ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"badcase → {args.out}")
    print({k: v for k, v in cats.most_common()})


if __name__ == "__main__":
    main()
