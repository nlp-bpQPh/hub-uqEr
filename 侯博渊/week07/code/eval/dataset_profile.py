# -*- coding: utf-8 -*-
"""数据集画像图：DuEE 1.0（新）vs DuEE-Fin（课程原数据集）。"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIN_DIR = Path(r"E:\python\第七周 抽取任务\week7-信息抽取\code\data\processed")
NEW_DIR = ROOT / "data" / "processed_full65"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    fin = load(FIN_DIR / "train_extract.jsonl")
    new = load(NEW_DIR / "train_extract.jsonl")
    fin_lens = [len(r["text"]) for r in fin]
    new_lens = [len(r["text"]) for r in new]
    fin_tc = Counter(r["event"]["type"] for r in fin)
    new_tc = Counter(r["event"]["type"] for r in new)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    items = new_tc.most_common(15)
    ax.barh(range(len(items))[::-1], [c for _, c in items], color="#4C78A8")
    ax.set_yticks(range(len(items))[::-1])
    ax.set_yticklabels([t.split("-")[-1] for t, _ in items], fontsize=9)
    ax.set_xlabel("训练句数（单事件句）")
    ax.set_title(f"DuEE 1.0：65 类长尾（前 15 类占 {sum(c for _, c in items)/len(new):.0%}）", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    for lens, label, color in ((fin_lens, "DuEE-Fin（课程原数据）", "#E45756"),
                               (new_lens, "DuEE 1.0（本次换的数据）", "#4C78A8")):
        xs = sorted(lens)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=f"{label}（中位 {xs[len(xs)//2]} 字）", color=color, linewidth=1.8)
    ax.set_xlim(0, 400)
    ax.set_xlabel("句子长度（字符）")
    ax.set_ylabel("累计比例 (CDF)")
    ax.set_title("句长分布：金融公告长句 vs 开放域短句", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "fig_dataset_profile.png", dpi=150)
    plt.close(fig)

    stats = {
        "fin": {"train": len(fin), "types": len(fin_tc), "mean_len": sum(fin_lens)/len(fin_lens),
                "p50_len": sorted(fin_lens)[len(fin_lens)//2], "roles": 61},
        "duee10": {"train_single": len(new), "types_seen": len(new_tc), "mean_len": sum(new_lens)/len(new_lens),
                   "p50_len": sorted(new_lens)[len(new_lens)//2], "roles": 121},
    }
    with open(ROOT / "outputs" / "report" / "dataset_profile.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
