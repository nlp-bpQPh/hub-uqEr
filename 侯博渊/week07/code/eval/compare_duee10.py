# -*- coding: utf-8 -*-
"""
DuEE 1.0 换数据集实验：统一汇总表 + 图（所有方法同一口径）

只读 outputs/ 下已经跑出来的预测与指标，不重新训练。做四件事：
  1. 汇总每个方法的三级 F1（trigger / argument / event）、type_macro、成本；
  2. 把**所有方法投影到同一批 LLM 抽样样本**（n=100）上再评一次，保证跨链路可比；
  3. 画 4 张图（方法对比 / 成本-效果 / per-type / 非法 BIO）；
  4. 生成 outputs/report/compare.md + compare.csv（报告与 PPT 直接引用）。

用法：
  python eval/compare_duee10.py                      # 默认 15 类主实验
  python eval/compare_duee10.py --tag_set full65     # 全 65 类补充实验
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from evaluate_events import evaluate, load_jsonl, row_key

# 中文字体（Windows 自带）；找不到时退回 DejaVu（会显示方块，但不影响数字）
for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
    if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORT = OUT / "report"
FIG.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

METRIC_DIR = OUT / "metrics" / "disc"
LLM_METRIC_DIR = OUT / "metrics" / "llm"
LLM_DIR = OUT / "llm"

# (key, 显示名, 指标文件, 类别, 输出 tag)
METHODS = [
    ("rule", "规则（词典+正则+词性）", METRIC_DIR / "rule.json", "rule", ""),
    ("bilstm_crf", "BiLSTM-CRF（字符级）", METRIC_DIR / "bilstm_crf.json", "disc", ""),
    ("berttiny_linear", "BERT-tiny+Linear", METRIC_DIR / "berttiny_linear.json", "disc", ""),
    ("berttiny_crf", "BERT-tiny+CRF", METRIC_DIR / "berttiny_crf.json", "disc", ""),
    ("berttiny_crf_hard", "BERT-tiny+CRF+硬约束解码", METRIC_DIR / "berttiny_crf_hard.json", "disc", ""),
]

LLM_METHODS = [
    ("llm_zero", "本地 Qwen3-0.6B zero-shot", "local_zero"),
    ("llm_few", "本地 Qwen3-0.6B few-shot", "local_few"),
]


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def collect(tag_set: str):
    methods = list(METHODS)
    if tag_set == "full65":
        methods = [
            ("rule_full65", "规则（65 类）", METRIC_DIR / "rule_full65.json", "rule", ""),
            ("berttiny_linear_full65", "BERT(4L)+Linear（65 类）",
             METRIC_DIR / "berttiny_linear_full65.json", "disc", ""),
        ]
    for key, disp, prefix in LLM_METHODS:
        methods.append((key, disp, LLM_METRIC_DIR / f"{prefix}.json", "llm", prefix))
    rows = []
    for key, disp, path, kind, tag in methods:
        res = load_json(path)
        if not res:
            print(f"[skip] 缺指标：{path.name}")
            continue
        meta = res.get("meta") or {}
        row = {
            "key": key, "name": disp, "kind": kind,
            "trigger": res["trigger"]["f1"],
            "argument": res["argument"]["f1"],
            "event": res["event_exact"]["f1"],
            "type_macro": res.get("type_macro_f1", 0.0),
            "train_s": meta.get("train_s", 0),
            "params_m": meta.get("params_m", 0),
            "ms": meta.get("ms_per_sentence", 0),
            "n": res.get("num_samples", 0),
            "illegal": (meta.get("illegal_dev") or {}).get("total_illegal"),
            "illegal_seq_rate": (meta.get("illegal_dev") or {}).get("illegal_seq_rate"),
            "pred": res.get("pred", ""),
            "metrics_path": str(path),
        }
        if kind == "llm":
            rep = load_json(LLM_DIR / f"report_{tag}.json")
            row["pred"] = str(LLM_DIR / f"pred_{tag}.jsonl")
            if rep:
                row["train_s"] = 0
                row["ms"] = rep.get("latency_mean_s", 0)
                row["json_valid"] = rep.get("json_valid_rate")
                row["tokens"] = rep.get("total_tokens")
                row["n"] = rep.get("n", row["n"])
        rows.append(row)
    return rows


def _tag_of(key: str, prefix: str) -> str:
    return prefix


def subset_eval(rows, subset_n: int):
    """把所有方法投影到 LLM 固定抽样的 n 条上（seed=42，嵌套子集）。"""
    gold_path = LLM_DIR / f"gold_n{subset_n}.jsonl"
    if not gold_path.exists():
        # 只认正式抽样（n>=20），跳过 --dry-run 留下的 gold_n3 之类小文件
        cands = [p for p in LLM_DIR.glob("gold_n*.jsonl")
                 if p.stem.replace("gold_n", "").isdigit() and int(p.stem.replace("gold_n", "")) >= 20]
        if not cands:
            print("[subset] 没有 LLM 抽样金标，跳过同批对比")
            return {}
        gold_path = max(cands, key=lambda p: int(p.stem.replace("gold_n", "")))
    gold = load_jsonl(str(gold_path))
    keys = {row_key(g) for g in gold}
    print(f"[subset] {gold_path.name}：{len(gold)} 条")
    out = {}
    for row in rows:
        pred_path = Path(row["pred"]) if row["pred"] else None
        if not pred_path or not pred_path.exists():
            continue
        preds = [p for p in load_jsonl(str(pred_path)) if row_key(p) in keys]
        res = evaluate(gold, preds, verbose=False)
        out[row["key"]] = {"trigger": res["trigger"]["f1"], "argument": res["argument"]["f1"],
                           "event": res["event_exact"]["f1"], "n": res["num_samples"],
                           "missing": res["missing_predictions"]}
    return out


def draw(rows, sub):
    # 图 1：三级 F1 分组柱状
    names = [r["name"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.26
    for i, (metric, label) in enumerate((("trigger", "Trigger F1"), ("argument", "Argument F1"),
                                        ("event", "Event F1"))):
        vals = [r[metric] for r in rows]
        ax.bar([xi + (i - 1) * w for xi in x], vals, width=w, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("DuEE 1.0（15 类子集）方法对比：三级 F1", fontsize=12)
    ax.legend(frameon=False)
    for xi, r in zip(x, rows):
        ax.text(xi, r["argument"] + 0.02, f'{r["argument"]:.3f}', ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_methods.png", dpi=150)
    plt.close(fig)

    # 图 2：成本 vs 论元 F1（训练时长对数轴；规则/LLM 单独标）
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for r in rows:
        xv = max(r["train_s"], 1)
        ax.scatter(xv, r["argument"], s=60)
        ax.annotate(r["name"].split("（")[0], (xv, r["argument"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("训练耗时（秒，log；规则=0 按 1 画；LLM 零样本=0）")
    ax.set_ylabel("Argument F1")
    ax.grid(alpha=0.3)
    ax.set_title("成本-效果：训练时间 vs 论元 F1", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cost.png", dpi=150)
    plt.close(fig)

    # 图 3：per-type argument/event F1（取事件句数 top 12 的类型）
    for key in ("berttiny_crf_hard", "berttiny_crf", "rule"):
        res = load_json(METRIC_DIR / f"{key}.json")
        if not res:
            continue
        per_type = res.get("per_type") or {}
        if not per_type:
            continue
        items = sorted(per_type.items(), key=lambda kv: -kv[1]["tp"])[:12]
        labels = [k.replace("-", "-\n", 1) if len(k) > 8 else k for k, _ in items]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar(range(len(items)), [v["f1"] for _, v in items], width=0.6)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([k for k, _ in items], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Event F1")
        ax.set_title(f"per-type 事件级 F1（{res['method']}）", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_per_type_{key}.png", dpi=150)
        plt.close(fig)
        break

    # 图 4：非法 BIO 数量（BiLSTM-CRF vs BERT-Linear；CRF 跑完后再补）
    il = []
    for key, label in (("bilstm_crf", "BiLSTM-CRF"), ("berttiny_linear", "BERT+Linear"),
                       ("berttiny_crf", "BERT+CRF"), ("berttiny_crf_hard", "BERT+CRF+硬约束")):
        res = load_json(METRIC_DIR / f"{key}.json")
        if res and (res.get("meta") or {}).get("illegal_dev"):
            d = res["meta"]["illegal_dev"]
            il.append((label, d.get("total_illegal", 0), d.get("illegal_seqs", 0)))
    if len(il) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.bar([x[0] for x in il], [x[1] for x in il], width=0.5)
        for i, (_, total, seqs) in enumerate(il):
            ax.text(i, total, f"{total}\n({seqs} 句)", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("非法 BIO 转移数（dev）")
        ax.set_title("非法序列：Linear 喷、CRF 少、硬约束归零", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "fig_illegal.png", dpi=150)
        plt.close(fig)


def write_tables(rows, sub, tag_set):
    fields = ["key", "name", "trigger", "argument", "event", "type_macro",
              "train_s", "params_m", "ms", "n", "illegal"]
    with open(REPORT / f"compare_{tag_set}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = [f"# DuEE 1.0 方法对比（{tag_set}）", "",
             "| 方法 | Trigger F1 | Argument F1 | Event F1 | type_macro | 训练(s) | 参数量(M) | 延迟(ms/句) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['trigger']:.4f} | {r['argument']:.4f} | "
                     f"{r['event']:.4f} | {r['type_macro']:.4f} | {r['train_s']} | "
                     f"{r['params_m']} | {r['ms']} |")
    if sub:
        lines += ["", f"## 同批 n={next(iter(sub.values()))['n']} 子集（seed=42，所有方法投影）", "",
                  "| 方法 | Trigger F1 | Argument F1 | Event F1 |", "|---|---|---|---|"]
        name = {r["key"]: r["name"] for r in rows}
        for k, v in sub.items():
            lines.append(f"| {name.get(k, k)} | {v['trigger']:.4f} | {v['argument']:.4f} | "
                         f"{v['event']:.4f} |")
    (REPORT / f"compare_{tag_set}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(REPORT / f"summary_{tag_set}.json", "w", encoding="utf-8") as f:
        json.dump({"methods": rows, "subset": sub}, f, ensure_ascii=False, indent=2)
    print(f"表已写入 {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_set", default="main", choices=["main", "full65"])
    ap.add_argument("--subset_n", type=int, default=100)
    args = ap.parse_args()
    rows = collect(args.tag_set)
    if not rows:
        print("没有任何结果可汇总")
        return
    sub = subset_eval(rows, args.subset_n)
    print("\n【汇总】")
    for r in rows:
        print(f"  {r['name']:28s} trig={r['trigger']:.4f} arg={r['argument']:.4f} "
              f"event={r['event']:.4f} macro={r['type_macro']:.4f} train={r['train_s']}s "
              f"ms={r['ms']} n={r['n']}")
    if sub:
        print("\n【同批子集】")
        for k, v in sub.items():
            print(f"  {k:20s} trig={v['trigger']:.4f} arg={v['argument']:.4f} event={v['event']:.4f}")
    draw(rows, sub)
    write_tables(rows, sub, args.tag_set)


if __name__ == "__main__":
    main()
