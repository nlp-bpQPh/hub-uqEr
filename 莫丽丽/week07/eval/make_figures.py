# -*- coding: utf-8 -*-
"""生成 BERT / RoBERTa 金融事件抽取项目图表。

图表均从 outputs/metrics/disc/*.json、outputs/logs/*_train.json 与
outputs/disc/pred_*.jsonl 读取；不会在绘图脚本中重新推理，也不会手写实验指标。

推荐流程（先保证 BERT、RoBERTa 的 clean dev 结果均已生成）：
  python eval/make_figures.py

可选：
  python eval/make_figures.py --only method_comparison paired_comparison per_role_compare
  python eval/make_figures.py --curve_runs bert_crf_256_clean roberta_crf_256_clean
"""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = ROOT / "outputs" / "metrics" / "disc"
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"
DATA_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "outputs" / "disc"

CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "WenQuanYi Zen Hei", "Arial Unicode MS"]
HAS_CJK_FONT = False
ZH_SHORT_LABELS = {
    "规则基线": "规则基线",
    "BiLSTM-CRF": "BiLSTM-CRF",
    "BERT+Linear": "BERT+Linear",
    "BERT+CRF": "BERT+CRF",
    "两段式流水线": "两段式\n流水线",
    "BERT+CRF（硬约束，旧dev）": "BERT硬约束\n（旧dev）",
    "BERT+CRF（clean dev）": "BERT+CRF\n（clean）",
    "BERT+CRF硬约束（clean dev）": "BERT硬约束\n（clean）",
    "RoBERTa+CRF（clean dev）": "RoBERTa+CRF\n（clean）",
    "RoBERTa+CRF硬约束（clean dev）": "RoBERTa硬约束\n（clean）",
    "RoBERTa+CRF硬约束（控制台摘要）": "RoBERTa硬约束\n（控制台摘要）",
}
EN_LABELS = {
    "规则基线": "Rule\nbaseline",
    "BiLSTM-CRF": "BiLSTM\nCRF",
    "BERT+Linear": "BERT\nLinear",
    "BERT+CRF": "BERT\nCRF",
    "两段式流水线": "Two-stage\npipeline",
    "BERT+CRF（硬约束，旧dev）": "BERT hard\n(old dev)",
    "BERT+CRF（clean dev）": "BERT CRF\n(clean)",
    "BERT+CRF硬约束（clean dev）": "BERT hard\n(clean)",
    "RoBERTa+CRF（clean dev）": "RoBERTa CRF\n(clean)",
    "RoBERTa+CRF硬约束（clean dev）": "RoBERTa hard\n(clean)",
    "RoBERTa+CRF硬约束（控制台摘要）": "RoBERTa hard\n(summary)",
    "触发词 F1": "Trigger F1",
    "论元 F1": "Argument F1",
    "事件级 F1": "Event-level F1",
    "触发词F1": "Trigger F1",
    "论元F1": "Argument F1",
    "事件级F1": "Event-level F1",
    "BERT+CRF（soft）": "BERT CRF (soft)",
    "BERT+CRF（hard）": "BERT CRF (hard)",
    "RoBERTa+CRF（soft）": "RoBERTa CRF (soft)",
    "RoBERTa+CRF（hard）": "RoBERTa CRF (hard)",
    "软约束：BERT vs RoBERTa": "Soft constraints: BERT vs RoBERTa",
    "硬约束：BERT vs RoBERTa": "Hard constraints: BERT vs RoBERTa",
    "当前结果参考对照（非同口径）": "Reference comparison\n(different dev sets)",
}

def display_text(value: str) -> str:
    if HAS_CJK_FONT:
        return ZH_SHORT_LABELS.get(value, value)
    return EN_LABELS.get(value, value)

# key 必须与 outputs/metrics/disc/<key>.json 的文件名一致。
METHODS = [
    ("rule", "规则基线"),
    ("bilstm_crf", "BiLSTM-CRF"),
    ("bert_linear_256", "BERT+Linear"),
    ("bert_crf_256", "BERT+CRF"),
    ("pipeline", "两段式流水线"),
    ("bert_crf_256_hard", "BERT+CRF（硬约束，旧dev）"),
    ("bert_crf_256_clean", "BERT+CRF（clean dev）"),
    ("bert_crf_256_clean_hard", "BERT+CRF硬约束（clean dev）"),
    ("roberta_crf_256_clean", "RoBERTa+CRF（clean dev）"),
    ("roberta_crf_256_clean_hard", "RoBERTa+CRF硬约束（clean dev）"),
    ("roberta_crf_256_clean_hard_console_summary", "RoBERTa+CRF硬约束（控制台摘要）"),
]

PAIR_METHODS = [
    ("bert_crf_256_clean", "roberta_crf_256_clean", "软约束：BERT vs RoBERTa"),
    ("bert_crf_256_clean_hard", "roberta_crf_256_clean_hard", "硬约束：BERT vs RoBERTa"),
]


def setup_style() -> None:
    global HAS_CJK_FONT
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in CJK_CANDIDATES if f in available]
    HAS_CJK_FONT = bool(picked)
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 180,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10.5, "axes.titlesize": 13, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "axes.grid": True, "grid.alpha": 0.22,
        "grid.linestyle": "--", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    if not picked:
        print("[warn] 未检测到常见中文字体；图表使用英文标签以避免中文乱码。Windows 安装微软雅黑后会自动恢复中文标签。")


def load_metric(name: str) -> dict | None:
    path = METRIC_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[skip] 指标文件无法读取 {path.name}: {exc}")
        return None


def meta_of(metric: dict) -> dict:
    return metric.get("meta") or {}


def dev_file_of(metric: dict) -> str:
    meta = meta_of(metric)
    value = meta.get("data_file") or meta.get("dev_file")
    if value:
        return Path(str(value).replace("\\", "/")).name
    gold = metric.get("gold")
    if gold:
        return Path(str(gold).replace("\\", "/")).name
    return "dev_extract.jsonl"


def sample_count(metric: dict) -> int:
    return int(metric.get("num_samples") or 0)


def same_eval_split(a: dict, b: dict) -> bool:
    return sample_count(a) == sample_count(b) and dev_file_of(a) == dev_file_of(b)


def save(fig, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


def _label_with_n(label: str, metric: dict) -> str:
    n = sample_count(metric)
    label = display_text(label)
    return f"{label}\n(n={n})" if n else label


# 1. 所有可用实验的三级 F1 主图；不同 dev 会明确提示“仅供参考”。
def fig_method_comparison(methods=METHODS) -> None:
    rows = []
    for key, label in methods:
        # 如果用户本机已有完整评测 JSON，不再把从控制台整理的摘要重复画一遍。
        if key == "roberta_crf_256_clean_hard_console_summary" and load_metric("roberta_crf_256_clean_hard"):
            continue
        metric = load_metric(key)
        if not metric or not all(k in metric for k in ("trigger", "argument", "event_exact")):
            continue
        rows.append((key, label, metric))
    if not rows:
        print("  [skip] method_comparison：没有可用指标文件")
        return

    labels = [_label_with_n(label, metric) for _, label, metric in rows]
    x = np.arange(len(rows))
    width = 0.25
    series = [
        (display_text("触发词 F1"), [r[2]["trigger"]["f1"] for r in rows], "#4C78A8"),
        (display_text("论元 F1"), [r[2]["argument"]["f1"] for r in rows], "#F58518"),
        (display_text("事件级 F1"), [r[2]["event_exact"]["f1"] for r in rows], "#54A24B"),
    ]
    fig, ax = plt.subplots(figsize=(max(14.0, len(rows) * 1.85), 6.0))
    for idx, (name, values, color) in enumerate(series):
        bars = ax.bar(x + (idx - 1) * width, values, width, label=name,
                      color=color, edgecolor="white", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.012, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8.5, rotation=0)
    splits = sorted({f"{dev_file_of(m)} / n={sample_count(m)}" for _, _, m in rows})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0, rotation=10, ha="right")
    ax.set_ylabel("F1 score" if not HAS_CJK_FONT else "F1 分数")
    ax.set_ylim(0, 1.0)
    if not HAS_CJK_FONT:
        title = ("Event extraction methods (mixed dev splits; overview only)"
                 if len(splits) > 1 else f"Event extraction methods ({splits[0]})")
    elif len(splits) == 1:
        title = f"事件抽取方法对比（统一评测集：{splits[0]}）"
    else:
        title = "事件抽取方法对比（包含不同验证集口径，仅作概览；严格比较请看同一 clean dev 对照图）"
    ax.set_title(title, pad=14)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_method_compare.png")


# 2. 当前能生成的 BERT / RoBERTa 参考对照图。
# 旧 BERT 是 930 条 dev_extract，当前 RoBERTa 是 920 条 dev_extract_clean，必须显式标注非同口径。
def fig_reference_comparison(
    bert_key: str = "bert_crf_256_hard",
    roberta_key: str = "roberta_crf_256_clean_hard_console_summary",
) -> None:
    bert = load_metric(bert_key)
    # 完整硬约束评测 JSON 一旦存在，优先用完整结果；否则用本次控制台摘要。
    rob = load_metric("roberta_crf_256_clean_hard") or load_metric(roberta_key)
    if not bert or not rob:
        print("  [skip] reference_comparison：缺少 BERT 历史基线或 RoBERTa 当前结果")
        return

    metric_keys = [("trigger", "Trigger F1"), ("argument", "Argument F1"),
                   ("event_exact", "Event-level F1")]
    bvals = [float(bert[k]["f1"]) for k, _ in metric_keys]
    rvals = [float(rob[k]["f1"]) for k, _ in metric_keys]
    x = np.arange(len(metric_keys))
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    b = ax.bar(x - w/2, bvals, w, label="BERT+CRF hard (historical)",
               color="#4C78A8", edgecolor="white")
    r = ax.bar(x + w/2, rvals, w, label="RoBERTa+CRF hard (current)",
               color="#F58518", edgecolor="white")
    for bars, vals in ((b, bvals), (r, rvals)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.012, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9)
    for i, (bv, rv) in enumerate(zip(bvals, rvals)):
        delta = (rv - bv) * 100
        ax.text(i, max(bv, rv) + 0.075, f"Δ {delta:+.2f} pp",
                ha="center", va="bottom", fontsize=8.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_keys])
    ax.set_ylim(0, min(1.0, max(max(bvals), max(rvals)) + 0.18))
    ax.set_ylabel("F1 score")
    b_split = f"{dev_file_of(bert)} (n={sample_count(bert)})"
    r_split = f"{dev_file_of(rob)} (n={sample_count(rob)})"
    ax.set_title("BERT vs RoBERTa — reference only, NOT a controlled comparison\n"
                 f"BERT: {b_split} | RoBERTa: {r_split}", pad=14)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.015,
             "Warning: evaluation sets differ (930 vs 920 samples); differences are descriptive only. "
             "Re-evaluate both checkpoints on dev_extract_clean.jsonl for a fair comparison.",
             ha="center", va="bottom", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "fig_bert_roberta_reference_compare.png")

    # 同步输出 CSV，防止从图片人工抄数时抄错。
    import csv
    out_csv = METRIC_DIR / "bert_roberta_reference_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dev_file", "num_samples", "trigger_f1",
                         "argument_f1", "event_exact_f1", "comparison_note"])
        writer.writerow(["BERT+CRF hard", dev_file_of(bert), sample_count(bert),
                         *bvals, "historical baseline; not same dev as RoBERTa"])
        writer.writerow(["RoBERTa+CRF hard", dev_file_of(rob), sample_count(rob),
                         *rvals, "current result; not same dev as BERT"])
    print(f"  ✓ {out_csv.relative_to(ROOT)}")


# 2. 专门的 BERT/RoBERTa 同验证集对比图。若验证集不一致，明确跳过。
def fig_paired_comparison(pairs=PAIR_METHODS) -> None:
    rows = []
    for bert_key, roberta_key, label in pairs:
        bert = load_metric(bert_key)
        rob = load_metric(roberta_key)
        if not bert or not rob:
            print(f"  [skip] {label}：缺少 {bert_key} 或 {roberta_key} 的指标文件")
            continue
        if not same_eval_split(bert, rob):
            print(f"  [skip] {label}：验证集不一致："
                  f"{dev_file_of(bert)}/n={sample_count(bert)} vs "
                  f"{dev_file_of(rob)}/n={sample_count(rob)}；请统一验证集后重评。")
            continue
        rows.append((label, bert, rob))
    if not rows:
        return

    metrics = [("触发词 F1", "trigger"), ("论元 F1", "argument"), ("事件级 F1", "event_exact")]
    metric_labels = [display_text(label) for label, _ in metrics]
    fig, axes = plt.subplots(1, len(rows), figsize=(6.2 * len(rows), 5.0), squeeze=False)
    for ax, (title, bert, rob) in zip(axes[0], rows):
        x = np.arange(len(metrics))
        w = 0.34
        bvals = [bert[k]["f1"] for _, k in metrics]
        rvals = [rob[k]["f1"] for _, k in metrics]
        b = ax.bar(x - w/2, bvals, w, label="BERT", color="#4C78A8")
        r = ax.bar(x + w/2, rvals, w, label="RoBERTa", color="#F58518")
        for bars, vals in ((b, bvals), (r, rvals)):
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.012, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        pair_title = title if HAS_CJK_FONT else ("Soft constraints: BERT vs RoBERTa" if "软" in title else "Hard constraints: BERT vs RoBERTa")
        ax.set_title(f"{pair_title}\n{dev_file_of(bert)} · n={sample_count(bert)}", pad=12)
        ax.legend(frameon=False)
    fig.suptitle("BERT 与 RoBERTa 同口径对照（验证数据完全一致）" if HAS_CJK_FONT else
                 "BERT vs RoBERTa (same evaluation split)", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, "fig_bert_roberta_same_dev.png")


# 3. Token 级非法 BIO；字符级空格映射伪转移不混入这张图。
def fig_illegal_bio() -> None:
    keys = [
        ("bert_crf_256_clean", "BERT+CRF（soft）"),
        ("bert_crf_256_clean_hard", "BERT+CRF（hard）"),
        ("roberta_crf_256_clean", "RoBERTa+CRF（soft）"),
        ("roberta_crf_256_clean_hard", "RoBERTa+CRF（hard）"),
    ]
    items = []
    for key, label in keys:
        metric = load_metric(key)
        if not metric:
            continue
        stats = meta_of(metric).get("illegal_token_dev") or {}
        if not stats:
            print(f"  [skip] {label}：指标文件还没有 token 级统计，请用新版 evaluate 脚本复评。")
            continue
        items.append((label, stats, metric))
    if not items:
        print("  [skip] illegal_bio：没有 token 级非法 BIO 统计")
        return

    labels = [f"{display_text(label)}\n(n={sample_count(metric)})" for label, _, metric in items]
    counts = [int(st.get("illegal_seqs", 0)) for _, st, _ in items]
    rates = [float(st.get("illegal_seq_rate", 0)) for _, st, _ in items]
    x = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(max(8.5, len(items)*2.1), 5.2))
    bars = ax.bar(x, counts, color=["#4C78A8", "#54A24B", "#F58518", "#59A14F"][:len(items)],
                  edgecolor="white", width=0.62)
    for bar, count, rate in zip(bars, counts, rates):
        ax.text(bar.get_x()+bar.get_width()/2, count + max(max(counts), 1)*0.025,
                f"{count} 句\n({rate:.2%})", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Invalid BIO sequences" if not HAS_CJK_FONT else "含非法 BIO 的句子数")
    ax.set_title("Token-level invalid BIO sequences (hard constraint should reach 0)" if not HAS_CJK_FONT else
                 "Token 级非法 BIO 序列（真实模型标签序列；硬约束应压至 0）", pad=14)
    ax.set_ylim(0, max(max(counts), 1)*1.28)
    fig.tight_layout()
    save(fig, "fig_illegal_bio.png")


# 4. 单模型角色论元 F1（默认优先画 RoBERTa hard 结果）。
def fig_per_role(method_key: str, method_label: str, min_support: int = 20) -> None:
    metric = load_metric(method_key)
    if not metric or not metric.get("per_role"):
        print(f"  [skip] per_role_f1：缺少 {method_key} 的 per_role")
        return
    if not HAS_CJK_FONT:
        print("  [skip] per_role_f1：当前环境没有中文字体；在 Windows 安装/启用微软雅黑后可绘制角色中文标签。")
        return
    items = [(role, val["f1"], val["tp"] + val["fn"])
             for role, val in metric["per_role"].items()]
    items = [item for item in items if item[2] >= min_support]
    items.sort(key=lambda x: x[1])
    top = items[:18]
    if not top:
        print(f"  [skip] per_role_f1：{method_key} 没有支持数 >= {min_support} 的角色")
        return
    labels = [f"{role}（{n}）" for role, _, n in top]
    vals = [f1 for _, f1, _ in top]
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    colors = ["#D9534F" if v < 0.2 else ("#F0AD4E" if v < 0.5 else "#5CB85C") for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    for bar, value in zip(bars, vals):
        ax.text(value + 0.008, bar.get_y()+bar.get_height()/2, f"{value:.3f}",
                va="center", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("论元 F1（括号内为金标支持数）")
    ax.set_title(f"各角色论元 F1（最难的 18 个）— {method_label} / {dev_file_of(metric)}", pad=14)
    fig.tight_layout()
    save(fig, "fig_per_role_f1.png")


# 5. BERT / RoBERTa 逐角色论元 F1 同口径对比。
def fig_per_role_compare(bert_key="bert_crf_256_clean_hard",
                         roberta_key="roberta_crf_256_clean_hard", min_support=20) -> None:
    bert = load_metric(bert_key)
    rob = load_metric(roberta_key)
    if not bert or not rob:
        print("  [skip] per_role_compare：请先生成 clean dev 的 BERT hard 与 RoBERTa hard 指标")
        return
    if not same_eval_split(bert, rob):
        print("  [skip] per_role_compare：BERT 与 RoBERTa 的验证集不一致，不能逐角色直接比较")
        return
    if not HAS_CJK_FONT:
        print("  [skip] per_role_compare：当前环境没有中文字体；在 Windows 安装/启用微软雅黑后可绘制角色中文标签。")
        return
    if not bert.get("per_role") or not rob.get("per_role"):
        print("  [skip] per_role_compare：至少一个指标 JSON 缺少 per_role")
        return
    roles = set(bert["per_role"]) | set(rob["per_role"])
    items = []
    for role in roles:
        b = bert["per_role"].get(role, {})
        r = rob["per_role"].get(role, {})
        support = max(int(b.get("tp", 0) + b.get("fn", 0)),
                      int(r.get("tp", 0) + r.get("fn", 0)))
        if support < min_support:
            continue
        mean = (float(b.get("f1", 0)) + float(r.get("f1", 0))) / 2
        items.append((role, float(b.get("f1", 0)), float(r.get("f1", 0)), support, mean))
    items.sort(key=lambda z: z[4])
    items = items[:18]
    if not items:
        print("  [skip] per_role_compare：没有满足支持数阈值的角色")
        return
    labels = [f"{role}（{support}）" for role, _, _, support, _ in items]
    y = np.arange(len(items))
    h = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    b = ax.barh(y-h/2, [z[1] for z in items], h, label="BERT+CRF hard", color="#4C78A8")
    r = ax.barh(y+h/2, [z[2] for z in items], h, label="RoBERTa+CRF hard", color="#F58518")
    for bars, values in ((b, [z[1] for z in items]), (r, [z[2] for z in items])):
        for bar, value in zip(bars, values):
            ax.text(value+0.008, bar.get_y()+bar.get_height()/2, f"{value:.3f}",
                    va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("论元 F1")
    ax.set_title(f"BERT 与 RoBERTa 的角色级论元 F1 对比（{dev_file_of(bert)}，n={sample_count(bert)}）", pad=14)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_per_role_compare.png")


def _load_train_log(run: str) -> list[dict] | None:
    for path in (LOG_DIR / f"{run}_train.json", LOG_DIR / f"train_{run}.json"):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        epochs = raw.get("epochs", []) if isinstance(raw, dict) else raw
        if epochs:
            return epochs
    return None


def _epoch_val_f1(row: dict):
    if "inner_val_span_f1" in row:
        return row["inner_val_span_f1"]
    return (row.get("val") or {}).get("inner_span_f1")


# 6. BERT / RoBERTa 训练曲线。
def fig_training_curves(runs: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    plotted = 0
    for i, (run, label) in enumerate(runs):
        log = _load_train_log(run)
        if not log:
            print(f"  [skip] training_curves：没有 {run} 的训练日志")
            continue
        epochs = [row.get("epoch", idx+1) for idx, row in enumerate(log)]
        axes[0].plot(epochs, [row.get("train_loss", float("nan")) for row in log],
                     marker="o", color=colors[i % len(colors)], label=label)
        vals = [_epoch_val_f1(row) for row in log]
        if any(value is not None for value in vals):
            axes[1].plot(epochs, [value if value is not None else float("nan") for value in vals],
                         marker="o", color=colors[i % len(colors)], label=label)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("  [skip] training_curves：没有可用训练日志")
        return
    axes[0].set_title("Training loss" if not HAS_CJK_FONT else "训练损失", pad=12)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Inner validation span F1 (epoch selection)" if not HAS_CJK_FONT else
                      "内层验证集 span F1（用于选 epoch）", pad=12)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("span F1")
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("BERT / RoBERTa fine-tuning curves" if not HAS_CJK_FONT else
                 "BERT / RoBERTa 微调训练曲线", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, "fig_training_curves.png")


# 7. 事件类型混淆矩阵：根据指标文件中的 data_file/dev_file 选择同一份 gold。
def _load_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _type_confusion_matrix(gold: list[dict], pred_rows: list[dict], types: list[str]):
    """行=金标事件类型，列=预测事件类型 + 未抽取/其他；不把漏抽静默丢掉。"""
    tid = {name: i for i, name in enumerate(types)}
    other_col = len(types)
    mat = np.zeros((len(types), len(types) + 1), dtype=int)
    pred = {(row.get("id"), row.get("sent_id")): row for row in pred_rows}
    for row in gold:
        gold_event = row.get("event") or {}
        gold_type = gold_event.get("type", "")
        if gold_type not in tid:
            continue
        predicted_row = pred.get((row.get("id"), row.get("sent_id")), {})
        events = predicted_row.get("events") or []
        pred_type = events[0].get("type", "") if events else ""
        col = tid[pred_type] if pred_type in tid else other_col
        mat[tid[gold_type], col] += 1
    return mat


def _draw_type_confusion(ax, mat: np.ndarray, types: list[str], title: str) -> None:
    labels = types + (["未抽取/其他"] if HAS_CJK_FONT else ["No event / other"])
    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else "#222222")
    ax.set_xlabel("Predicted type" if not HAS_CJK_FONT else "预测类型")
    ax.set_ylabel("Gold type" if not HAS_CJK_FONT else "金标类型")
    ax.set_title(title, pad=12)
    ax.grid(False)
    return im


def fig_type_confusion(method_key: str, method_label: str) -> None:
    metric = load_metric(method_key)
    if not metric:
        print(f"  [skip] type_confusion：缺少 {method_key}.json")
        return
    gold_file = DATA_DIR / dev_file_of(metric)
    pred_file = PRED_DIR / f"pred_{method_key}.jsonl"
    if not gold_file.exists() or not pred_file.exists():
        print(f"  [skip] type_confusion：缺少 gold={gold_file.name} 或 pred={pred_file.name}")
        return
    if not HAS_CJK_FONT:
        print("  [skip] type_confusion：当前环境没有中文字体；Windows 安装微软雅黑后可生成事件类型中文混淆矩阵。")
        return

    gold = _load_jsonl_rows(gold_file)
    pred_rows = _load_jsonl_rows(pred_file)
    types = sorted({(row.get("event") or {}).get("type", "") for row in gold} - {""})
    if not types:
        print(f"  [skip] type_confusion：{gold_file.name} 中没有事件类型")
        return
    mat = _type_confusion_matrix(gold, pred_rows, types)
    fig, ax = plt.subplots(figsize=(max(9.2, 0.58*len(types)), 8.0))
    im = _draw_type_confusion(ax, mat, types,
        f"事件类型混淆矩阵 — {display_text(method_label)}\n{gold_file.name} / n={len(gold)}")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Row-normalized recall")
    fig.tight_layout()
    safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", method_key)
    save(fig, f"fig_type_confusion_{safe_key}.png")


def fig_type_confusion_compare(
    bert_key: str = "bert_crf_256_clean_hard",
    roberta_key: str = "roberta_crf_256_clean_hard",
) -> None:
    """同一 clean dev 下并排比较 BERT/RoBERTa 事件类型混淆矩阵。"""
    bert = load_metric(bert_key)
    rob = load_metric(roberta_key)
    if not bert or not rob:
        print("  [skip] type_confusion_compare：缺少 BERT 或 RoBERTa 完整指标 JSON")
        return
    if not same_eval_split(bert, rob):
        print("  [skip] type_confusion_compare：BERT 与 RoBERTa 不是同一份验证集")
        return
    if not HAS_CJK_FONT:
        print("  [skip] type_confusion_compare：当前环境没有中文字体；Windows 安装微软雅黑后可绘制")
        return
    gold_file = DATA_DIR / dev_file_of(bert)
    bpred_file = PRED_DIR / f"pred_{bert_key}.jsonl"
    rpred_file = PRED_DIR / f"pred_{roberta_key}.jsonl"
    missing = [p.name for p in (gold_file, bpred_file, rpred_file) if not p.exists()]
    if missing:
        print(f"  [skip] type_confusion_compare：缺少文件 {', '.join(missing)}")
        return
    gold = _load_jsonl_rows(gold_file)
    types = sorted({(row.get("event") or {}).get("type", "") for row in gold} - {""})
    if not types:
        print("  [skip] type_confusion_compare：金标没有事件类型")
        return
    bmat = _type_confusion_matrix(gold, _load_jsonl_rows(bpred_file), types)
    rmat = _type_confusion_matrix(gold, _load_jsonl_rows(rpred_file), types)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.4), squeeze=False)
    im = _draw_type_confusion(axes[0, 0], bmat, types, "BERT+CRF hard")
    _draw_type_confusion(axes[0, 1], rmat, types, "RoBERTa+CRF hard")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label="Row-normalized recall")
    fig.suptitle(f"BERT vs RoBERTa event type confusion ({gold_file.name}, n={len(gold)})",
                 y=1.01, fontsize=14)
    fig.tight_layout()
    save(fig, "fig_type_confusion_bert_roberta_compare.png")


# 8. 延迟、参数量、训练时间及抽取 F1。
def fig_latency_cost(methods=METHODS) -> None:
    rows = []
    for key, label in methods:
        if key == "roberta_crf_256_clean_hard_console_summary" and load_metric("roberta_crf_256_clean_hard"):
            continue
        metric = load_metric(key)
        if not metric:
            continue
        meta = meta_of(metric)
        rows.append([display_text(label), f"{sample_count(metric)} / {dev_file_of(metric)}",
                     str(meta.get("ms_per_sentence", "-")),
                     str(meta.get("params_m", "-")),
                     str(meta.get("train_s", "-")),
                     f"{metric['trigger']['f1']:.3f}",
                     f"{metric['argument']['f1']:.3f}",
                     f"{metric['event_exact']['f1']:.3f}"])
    if not rows:
        print("  [skip] latency_cost：没有可用指标")
        return
    fig, ax = plt.subplots(figsize=(13.0, 0.48*len(rows)+2.0))
    ax.axis("off")
    cols = (["Method", "Evaluation split", "Latency (ms/sent)", "Params (M)", "Train (s)",
             "Trigger F1", "Argument F1", "Event F1"] if not HAS_CJK_FONT else
            ["方法", "验证集", "延迟(ms/句)", "参数量(M)", "训练(s)", "触发词F1", "论元F1", "事件级F1"])
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.42)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#4C78A8")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FB")
    ax.set_title("Precision / latency / model-size trade-off (evaluation split shown per row)" if not HAS_CJK_FONT else
                 "工程视角：精度 / 延迟 / 参数量权衡（验证集口径单独标注）", pad=18, fontsize=13.5)
    fig.tight_layout()
    save(fig, "fig_latency_cost.png")


def main():
    ap = argparse.ArgumentParser(description="生成 BERT / RoBERTa 对照图表")
    ap.add_argument("--only", nargs="*", default=None,
                    help="method_comparison reference_comparison paired_comparison illegal_bio "
                         "per_role_f1 per_role_compare training_curves type_confusion "
                         "type_confusion_compare latency_cost")
    ap.add_argument("--role_method", default="roberta_crf_256_clean_hard")
    ap.add_argument("--curve_runs", nargs="*", default=None)
    ap.add_argument("--confusion_methods", nargs="*", default=None,
                    help="需要绘制混淆矩阵的方法 key，可同时传 BERT 与 RoBERTa")
    args = ap.parse_args()
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {
        "method_comparison", "reference_comparison", "paired_comparison", "illegal_bio", "per_role_f1",
        "per_role_compare", "training_curves", "type_confusion", "type_confusion_compare", "latency_cost"}

    if "method_comparison" in want:
        fig_method_comparison()
    if "reference_comparison" in want:
        fig_reference_comparison()
    if "paired_comparison" in want:
        fig_paired_comparison()
    if "illegal_bio" in want:
        fig_illegal_bio()
    if "per_role_f1" in want:
        metric = load_metric(args.role_method)
        label = (meta_of(metric).get("display_name") if metric else None) or args.role_method
        fig_per_role(args.role_method, label)
    if "per_role_compare" in want:
        fig_per_role_compare()
    if "training_curves" in want:
        run_specs = args.curve_runs or [
            ("bert_crf_256_clean", "BERT+CRF"),
            ("roberta_crf_256_clean", "RoBERTa+CRF"),
        ]
        if args.curve_runs:
            run_specs = [(run, run) for run in args.curve_runs]
        fig_training_curves(run_specs)
    if "type_confusion" in want:
        selected = args.confusion_methods or [
            "bert_crf_256_clean_hard", "roberta_crf_256_clean_hard",
            "bert_crf_256", "roberta_crf_256_clean",
        ]
        for key in selected:
            metric = load_metric(key)
            label = (meta_of(metric).get("display_name") if metric else None) or key
            fig_type_confusion(key, label)
    if "type_confusion_compare" in want:
        fig_type_confusion_compare()
    if "latency_cost" in want:
        fig_latency_cost()
    print("完成。")

if __name__ == "__main__":
    main()
