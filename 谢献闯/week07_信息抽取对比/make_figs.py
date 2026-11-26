# -*- coding: utf-8 -*-
"""
按 outputs/w7_compare/compare_all.json 生成报告用图（中文标签，Windows 微软雅黑）。

图：
  1. fig_main_compare.png     —— DuEE-Fin 主对比（trigger / argument / event 三组柱）
  2. fig_f1_vs_latency.png    —— 论元 F1 vs 单句延迟（对数横轴，标注方法名）
  3. fig_dataset_compare.png  —— 同方法跨数据集（DuEE-Fin 13 类 vs DuEE1.0 65 类）
  4. fig_llm_scale.png        —— 生成式链路：模型规模 × 是否微调
用法：python w7_ext/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "w7_compare" / "compare_all.json"
OUT = ROOT / "outputs" / "w7_compare" / "figs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def load():
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("status") == "已跑"]


def short(r):
    """图上用的短名（避免长中文标签重叠）。"""
    m = {
        "规则（词典+正则）": "规则",
        "BiLSTM-CRF（无预训练）": "BiLSTM-CRF",
        "BERT+Linear（联合 149 标签）": "BERT+Linear",
        "BERT+CRF（联合 149 标签）": "BERT+CRF",
        "RoBERTa-wwm-ext+CRF": "RoBERTa+CRF",
        "两段式流水线（检测→类型→论元）": "两段式",
        "Qwen3-0.6B zero-shot": "0.6B zero",
        "Qwen3-1.7B zero-shot": "1.7B zero",
        "Qwen3-0.6B few-shot(6 例)": "0.6B few",
        "Qwen3-0.6B + LoRA SFT": "0.6B SFT",
        "Qwen3-1.7B + LoRA SFT": "1.7B SFT",
        "Qwen3-0.6B via 自建 OpenAI API": "0.6B API",
        "Qwen3-0.6B few-shot via API": "0.6B API few",
    }
    return m.get(r["display"], r["display"])


def fig_main(rows):
    sub = [r for r in rows if r["dataset"] == "DuEE-Fin"]
    if not sub:
        return
    labels = [short(r) for r in sub]
    tr = [r["trigger_f1"] for r in sub]
    ar = [r["argument_f1"] for r in sub]
    ev = [r["event_f1"] for r in sub]
    x = np.arange(len(sub))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(11, 0.95 * len(sub)), 5.2))
    b1 = ax.bar(x - w, tr, w, label="触发词 F1", color="#4C78A8")
    b2 = ax.bar(x, ar, w, label="论元 F1", color="#F58518")
    b3 = ax.bar(x + w, ev, w, label="事件级 F1", color="#54A24B")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("DuEE-Fin 事件抽取：多方法横向对比（dev 930 句全量；生成式方法为固定 200 条抽样）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_compare.png")
    plt.close(fig)


def fig_latency(rows):
    pts = [(r, short(r)) for r in rows if r["dataset"] == "DuEE-Fin" and r.get("latency_ms")]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for r, name in pts:
        is_llm = r["layer"] in ("L3", "L4", "API")
        ax.scatter(r["latency_ms"], r["argument_f1"],
                   s=110 if is_llm else 80,
                   color="#E45756" if is_llm else "#4C78A8",
                   marker="o" if is_llm else "s", zorder=3)
        ax.annotate(f"{name}\n({r['latency_ms']:.0f} ms)", (r["latency_ms"], r["argument_f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("单句延迟 / ms（对数轴）")
    ax.set_ylabel("论元 F1")
    ax.set_title("精度–延迟权衡（方块=判别式，圆点=生成式/API）")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_f1_vs_latency.png")
    plt.close(fig)


def fig_dataset(rows):
    pairs = [("规则", "规则（词典+正则）", "w7_rule_duee1"),
             ("BERT+CRF", "BERT+CRF（联合 149 标签）", "w7_bert_crf_duee1"),
             ("0.6B zero", "Qwen3-0.6B zero-shot", "w7_qwen06_zero_duee1"),
             ("0.6B SFT", "Qwen3-0.6B + LoRA SFT", "w7_qwen06_sft_duee1")]
    labels, fin_v, d1_v = [], [], []
    for nice, fin_disp, d1_name in pairs:
        f = next((r for r in rows if r["dataset"] == "DuEE-Fin" and r["display"] == fin_disp), None)
        d = next((r for r in rows if r["name"] == d1_name), None)
        if not f or not d:
            continue
        labels.append(nice)
        fin_v.append(f["argument_f1"])
        d1_v.append(d["argument_f1"])
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b1 = ax.bar(x - w / 2, fin_v, w, label="DuEE-Fin（13 类 / 92 槽位，930 句）", color="#4C78A8")
    b2 = ax.bar(x + w / 2, d1_v, w, label="DuEE 1.0（65 类 / 216 槽位，1258 句）", color="#F58518")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("论元 F1")
    ax.set_title("换数据集：同一套方法在「金融 13 类」与「通用 65 类」上的论元 F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_compare.png")
    plt.close(fig)


def fig_llm_scale(rows):
    want = [("Qwen3-0.6B zero-shot", "0.6B\nzero-shot"),
            ("Qwen3-1.7B zero-shot", "1.7B\nzero-shot"),
            ("Qwen3-0.6B few-shot(6 例)", "0.6B\nfew-shot"),
            ("Qwen3-0.6B via 自建 OpenAI API", "0.6B\nAPI"),
            ("Qwen3-0.6B + LoRA SFT", "0.6B\nLoRA SFT"),
            ("Qwen3-1.7B + LoRA SFT", "1.7B\nLoRA SFT")]
    labels, vals = [], []
    for disp, lab in want:
        r = next((x for x in rows if x["dataset"] == "DuEE-Fin" and x["display"] == disp), None)
        if r:
            labels.append(lab)
            vals.append(r["argument_f1"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = ["#72B7B2", "#72B7B2", "#72B7B2", "#B279A2", "#E45756", "#E45756"][:len(vals)]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("论元 F1（固定 200 条抽样）")
    ax.set_title("生成式链路：模型规模 / 提示策略 / 调用方式 / 微调 的影响")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_llm_scale.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_main(rows)
    fig_latency(rows)
    fig_dataset(rows)
    fig_llm_scale(rows)
    print("figures →", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
