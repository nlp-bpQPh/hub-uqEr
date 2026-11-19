"""汇总所有实验 JSON → 表格（CSV/Markdown）+ 报告用图。

主指标用「去重测试集」（剔除与训练集完全同句的样本），同时保留全量 test 指标，
用来展示数据泄漏对不同方法的影响。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfg import DATASET_SPECS, OUT_DIR, FIG_DIR

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR.mkdir(parents=True, exist_ok=True)

DS_ORDER = ["tnews", "thucnews", "chnsenticorp"]
DS_SHORT = {"tnews": "TNEWS 15类", "thucnews": "THUCNews 10类", "chnsenticorp": "ChnSentiCorp 情感"}
METHOD_ORDER = {"rule_keyword": 0, "tfidf_svm": 1, "frozen_probe": 2, "encoder_finetune": 3}


def load_all():
    rows = []
    for p in sorted(Path(OUT_DIR).glob("*.json")):
        if p.name in ("smoke.json", "dataset_stats.json"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "report" not in d or "dataset" not in d:
            continue
        r, rc = d["report"], d.get("report_clean", d["report"])
        rows.append({
            "dataset": d["dataset"],
            "dataset_display": DATASET_SPECS[d["dataset"]]["display"],
            "method": d.get("method"),
            "model_key": d.get("model_key", ""),
            "model_display": d.get("model_display", d.get("method")),
            "n_test": r.get("n"),
            "acc": r.get("acc"), "macro_f1": r.get("macro_f1"),
            "n_clean": rc.get("n"),
            "acc_clean": rc.get("acc"), "macro_f1_clean": rc.get("macro_f1"),
            "removed_rate": rc.get("removed_rate", 0.0),
            "ece": rc.get("ece", r.get("ece")),
            "train_seconds": d.get("train_seconds", d.get("fit_seconds")),
            "peak_gpu_mb": d.get("peak_gpu_mb"),
            "file": p.name,
        })
    return pd.DataFrame(rows)


def method_label(row):
    if row["method"] == "encoder_finetune":
        return f"{row['model_display']} 微调"
    if row["method"] == "frozen_probe":
        return f"{row['model_display']}"
    return row["model_display"]


def build_tables(df):
    df = df.copy()
    df["label"] = df.apply(method_label, axis=1)
    df["order"] = df["method"].map(METHOD_ORDER).fillna(9)
    df = df.sort_values(["dataset", "order", "label"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "results.csv", index=False, encoding="utf-8-sig")

    lines = ["# 实验结果汇总（主指标：去重测试集 Macro-F1）\n"]
    for ds in DS_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        lines.append(f"## {DATASET_SPECS[ds]['display']}\n")
        lines.append("| 方法 | Acc(全量) | F1(全量) | Acc(去重) | F1(去重) | 去重样本 | 训练耗时 | 显存峰值 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            t = "-" if pd.isna(r["train_seconds"]) else f"{r['train_seconds']:.0f}s"
            m = "-" if pd.isna(r["peak_gpu_mb"]) else f"{r['peak_gpu_mb']:.0f}MB"
            lines.append(f"| {r['label']} | {r['acc']:.4f} | {r['macro_f1']:.4f} | "
                         f"{r['acc_clean']:.4f} | {r['macro_f1_clean']:.4f} | {r['n_clean']} | {t} | {m} |")
        lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def fig_methods_tnews(df):
    sub = df[df["dataset"] == "tnews"].sort_values("macro_f1_clean")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(sub) + 1.5))
    colors = ["#8c8c8c" if m != "encoder_finetune" else "#2f6fdd" for m in sub["method"]]
    bars = ax.barh(sub["label"], sub["macro_f1_clean"], color=colors)
    for b, v in zip(bars, sub["macro_f1_clean"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(sub["macro_f1_clean"]) * 1.18)
    ax.set_xlabel("Macro-F1（8,901 条去重验证集）")
    ax.set_title("TNEWS 15 类：换预训练模型后的方法对比", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_tnews_methods.png", dpi=160)
    plt.close(fig)


def fig_cross_dataset(df):
    methods = [("tfidf_svm", "TF-IDF+SVM"), ("bert-base", "BERT-base"),
               ("roberta-wwm-ext", "RoBERTa-wwm-ext"), ("macbert-base", "MacBERT-base")]
    dsets = [d for d in DS_ORDER if not df[df["dataset"] == d].empty]
    vals = {m[1]: [] for m in methods}
    for ds in dsets:
        for key, name in methods:
            if key == "tfidf_svm":
                v = df[(df.dataset == ds) & (df.method == "tfidf_svm")]["macro_f1_clean"]
            else:
                v = df[(df.dataset == ds) & (df.model_key == key) &
                       (df.method == "encoder_finetune")]["macro_f1_clean"]
            vals[name].append(float(v.iloc[0]) if len(v) else np.nan)

    x = np.arange(len(dsets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (_, name) in enumerate(methods):
        bars = ax.bar(x + (i - 1.5) * width, vals[name], width, label=name)
        for b, v in zip(bars, vals[name]):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_SHORT[d] for d in dsets])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro-F1（去重测试集）")
    ax.set_title("三个数据集上，方法排序是否稳定", fontsize=12, pad=14)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cross_dataset.png", dpi=160)
    plt.close(fig)


def fig_curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in [("bert-base", "#888888"), ("roberta-wwm-ext", "#2f6fdd"),
                       ("macbert-base", "#d9822b")]:
        p = Path(OUT_DIR) / f"enc_{key}_tnews.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        h = d["history"]
        ax.plot([x["epoch"] for x in h], [x["dev_macro_f1"] for x in h],
                marker="o", label=d["model_display"], color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev Macro-F1")
    ax.set_xticks([1, 2, 3])
    ax.set_title("TNEWS 训练曲线（dev 选 epoch，test 不参与）", fontsize=12)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_training_curves.png", dpi=160)
    plt.close(fig)


def fig_per_class():
    files = [("bert-base", "BERT-base", "#888888"),
             ("roberta-wwm-ext", "RoBERTa-wwm-ext", "#2f6fdd"),
             ("macbert-base", "MacBERT-base", "#d9822b")]
    data = {}
    for key, name, color in files:
        p = Path(OUT_DIR) / f"enc_{key}_tnews.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        data[name] = d["report"]["per_class_recall"]
    if not data:
        return
    labels = DATASET_SPECS["tnews"]["labels"]
    x = np.arange(len(labels))
    n = len(data)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (name, rec) in enumerate(data.items()):
        vals = [rec[str(j)] or 0 for j in range(len(labels))]
        ax.bar(x + (i - (n - 1) / 2) * 0.27, vals, 0.27, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_title("TNEWS 每类召回：换模型主要改善哪些类", fontsize=12)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_per_class_recall.png", dpi=160)
    plt.close(fig)


def fig_cost(df):
    """训练成本：同一数据集内 TF-IDF 与三个编码器的耗时（对数轴）。"""
    dsets = [d for d in DS_ORDER if not df[df["dataset"] == d].empty]
    methods = [("tfidf_svm", "TF-IDF+SVM"), ("bert-base", "BERT-base"),
               ("roberta-wwm-ext", "RoBERTa-wwm-ext"), ("macbert-base", "MacBERT-base")]
    vals = {name: [] for _, name in methods}
    for ds in dsets:
        for key, name in methods:
            if key == "tfidf_svm":
                v = df[(df.dataset == ds) & (df.method == "tfidf_svm")]["train_seconds"]
            else:
                v = df[(df.dataset == ds) & (df.model_key == key) &
                       (df.method == "encoder_finetune")]["train_seconds"]
            vals[name].append(float(v.iloc[0]) if len(v) else np.nan)

    x = np.arange(len(dsets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (_, name) in enumerate(methods):
        bars = ax.bar(x + (i - 1.5) * width, vals[name], width, label=name)
        for b, v in zip(bars, vals[name]):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.0f}s" if v >= 10 else f"{v:.1f}s",
                        ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 2000)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_SHORT[d] for d in dsets])
    ax.set_ylabel("训练耗时（秒，对数轴）")
    ax.set_title("训练成本：换预训练权重几乎不增加成本，换方法才增加两个数量级", fontsize=11.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_cost.png", dpi=160)
    plt.close(fig)


def fig_confusion():
    best = None
    for key in ("roberta-wwm-ext", "macbert-base", "bert-base"):
        p = Path(OUT_DIR) / f"enc_{key}_tnews.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if best is None or d["report"]["macro_f1"] > best[1]["report"]["macro_f1"]:
                best = (key, d)
    if not best:
        return
    d = best[1]
    cm = np.array(d["report"]["confusion_matrix"], dtype=float)
    cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    labels = DATASET_SPECS["tnews"]["labels"]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"TNEWS 混淆矩阵（行归一化）· {d['model_display']}", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_confusion_matrix.png", dpi=160)
    plt.close(fig)


def fig_leakage(df):
    """数据泄漏：同一方法在「全量 test」与「去重 test」上的差（只画重复率高的数据集）。"""
    sub = df[(df["removed_rate"] >= 0.02) & (df["method"].isin(["tfidf_svm", "encoder_finetune"]))]
    if sub.empty:
        return
    sub = sub.sort_values("macro_f1", ascending=False)
    short = {"tfidf_svm": "TF-IDF", "bert-base": "BERT-base",
             "roberta-wwm-ext": "RoBERTa", "macbert-base": "MacBERT"}
    labels = [short.get(r["model_key"], r["label"]) if r["method"] == "encoder_finetune"
              else "TF-IDF" for _, r in sub.iterrows()]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - 0.19, sub["macro_f1"], 0.38, label="全量 test（含重复）")
    ax.bar(x + 0.19, sub["macro_f1_clean"], 0.38, label="去重 test")
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i - 0.19, r["macro_f1"] + 0.004, f"{r['macro_f1']:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.19, r["macro_f1_clean"] + 0.004, f"{r['macro_f1_clean']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.80, 0.99)
    ax.set_ylabel("Macro-F1")
    ax.set_title("ChnSentiCorp：测试集 27.6% 与训练集重复，去重后所有方法都掉点", fontsize=11.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_leakage.png", dpi=160)
    plt.close(fig)


def main():
    df = load_all()
    df = build_tables(df)
    fig_methods_tnews(df)
    fig_cross_dataset(df)
    fig_curves()
    fig_per_class()
    fig_cost(df)
    fig_confusion()
    fig_leakage(df)
    print(df[["dataset", "label", "acc_clean", "macro_f1_clean", "train_seconds"]].to_string(index=False))
    print(f"\n→ {OUT_DIR / 'summary.md'}\n→ {FIG_DIR}")


if __name__ == "__main__":
    main()
