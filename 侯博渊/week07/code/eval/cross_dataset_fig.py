# -*- coding: utf-8 -*-
"""跨数据集对比图：DuEE-Fin vs DuEE 1.0（只画口径可比的三条线）。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "report"
FIG = ROOT / "outputs" / "figures"
for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    fin = json.loads((REPORT / "fin_reference.json").read_text(encoding="utf-8"))["methods"]
    new = {}
    for key, path in (("rule", "disc/rule.json"), ("bilstm_crf", "disc/bilstm_crf.json"),
                      ("llm_local_zero", "llm/local_zero.json")):
        p = ROOT / "outputs" / "metrics" / path
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            new[key] = {"trigger": r["trigger"]["f1"], "argument": r["argument"]["f1"]}

    names = ["规则（零训练）", "BiLSTM-CRF", "本地 Qwen3-0.6B zero"]
    keys = ["rule", "bilstm_crf", "llm_local_zero"]
    fin_keys = ["rule", "bilstm_crf", "llm_local_zero_n60"]
    x = range(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in ((axes[0], "trigger", "Trigger F1"), (axes[1], "argument", "Argument F1")):
        fin_v = [fin[fk][f"{metric}_f1"] for fk in fin_keys]
        new_v = [new.get(k, {}).get(metric, 0) for k in keys]
        ax.bar([i - 0.18 for i in x], fin_v, width=0.36, label="DuEE-Fin（课程）", color="#E45756")
        ax.bar([i + 0.18 for i in x], new_v, width=0.36, label="DuEE 1.0（本次）", color="#4C78A8")
        for i, (a, b) in enumerate(zip(fin_v, new_v)):
            ax.text(i - 0.18, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
            ax.text(i + 0.18, b + 0.02, f"{b:.3f}", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("同一方法、不同数据集：触发词与论元的相对变化", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cross_dataset.png", dpi=150)
    print("written", FIG / "fig_cross_dataset.png")


if __name__ == "__main__":
    main()
