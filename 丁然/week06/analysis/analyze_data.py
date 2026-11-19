"""分析 TNEWS 原始数据并生成统计图和报告。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))
MPL_DIR = WEEK_DIR / "outputs" / "analysis" / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from labels import LABELS  # noqa: E402


MODEL_PATH = Path("~/Downloads/模型/bert/bert-base-chinese").expanduser()


def read_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[int], ratio: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti SC",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def quality_stats(
    train_rows: list[dict],
    val_rows: list[dict],
) -> dict[str, int]:
    train_count = Counter(row["sentence"] for row in train_rows)
    val_count = Counter(row["sentence"] for row in val_rows)
    train_map = defaultdict(set)
    val_map = defaultdict(set)
    for row in train_rows:
        train_map[row["sentence"]].add(row["label"])
    for row in val_rows:
        val_map[row["sentence"]].add(row["label"])
    overlap = set(train_map) & set(val_map)
    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_unique": len(train_count),
        "val_unique": len(val_count),
        "train_duplicates": sum(count - 1 for count in train_count.values()),
        "val_duplicates": sum(count - 1 for count in val_count.values()),
        "train_conflicts": sum(len(labels) > 1 for labels in train_map.values()),
        "val_conflicts": sum(len(labels) > 1 for labels in val_map.values()),
        "cross_overlap": len(overlap),
        "cross_conflicts": sum(
            train_map[sentence] != val_map[sentence] for sentence in overlap
        ),
    }


def plot_class_distribution(
    train_rows: list[dict],
    val_rows: list[dict],
    output_path: Path,
) -> None:
    train = Counter(row["label"] for row in train_rows)
    val = Counter(row["label"] for row in val_rows)
    positions = np.arange(len(LABELS))
    height = 0.38
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(positions - height / 2, [train[i] for i in range(15)], height, label="训练集")
    axis.barh(
        positions + height / 2,
        [val[i] for i in range(15)],
        height,
        label="验证集",
    )
    axis.set_yticks(positions, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("样本数")
    axis.set_title("15 个类别的样本分布")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_lengths(
    train_lengths: list[int],
    val_lengths: list[int],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    maximum = max(train_lengths)
    bins = np.arange(0, maximum + 5, 5)
    axes[0].hist(train_lengths, bins=bins, alpha=0.65, label="训练集")
    axes[0].hist(val_lengths, bins=bins, alpha=0.55, label="验证集")
    axes[0].axvline(percentile(train_lengths, 0.95), color="red", linestyle="--")
    axes[0].axvline(percentile(train_lengths, 0.99), color="green", linestyle="--")
    axes[0].set_xlabel("字符数")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("新闻字符长度分布")
    axes[0].legend()

    for values, name in ((train_lengths, "训练集"), (val_lengths, "验证集")):
        ordered = np.sort(values)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[1].plot(ordered, cumulative, label=name)
    axes[1].axhline(0.95, color="red", linestyle="--")
    axes[1].axhline(0.99, color="green", linestyle="--")
    axes[1].set_xlabel("字符数")
    axes[1].set_ylabel("累计占比")
    axes[1].set_title("长度累计分布")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_length_by_class(train_rows: list[dict], output_path: Path) -> None:
    lengths = [
        [len(row["sentence"]) for row in train_rows if row["label"] == label]
        for label in range(15)
    ]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.boxplot(lengths, tick_labels=LABELS, showfliers=False)
    axis.set_ylabel("字符数")
    axis.set_title("各类别新闻长度分布")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_token_lengths(train_rows: list[dict], output_path: Path) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    lengths = [
        len(tokenizer.encode(row["sentence"], add_special_tokens=True))
        for row in train_rows
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(lengths, bins=35)
    for ratio, color in ((0.95, "red"), (0.99, "green")):
        axis.axvline(
            percentile(lengths, ratio),
            color=color,
            linestyle="--",
            label=f"P{int(ratio * 100)}={percentile(lengths, ratio):.1f}",
        )
    axis.set_xlabel("BERT token 数")
    axis.set_ylabel("样本数")
    axis.set_title("BERT token 长度分布")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
    }


def plot_quality(quality: dict[str, int], output_path: Path) -> None:
    keys = [
        "train_duplicates",
        "val_duplicates",
        "train_conflicts",
        "val_conflicts",
        "cross_overlap",
        "cross_conflicts",
    ]
    names = ["训练重复", "验证重复", "训练冲突", "验证冲突", "交叉同句", "交叉冲突"]
    values = [quality[key] for key in keys]
    figure, axis = plt.subplots(figsize=(9, 4))
    bars = axis.bar(names, values)
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("数量")
    axis.set_title("原始数据重复与标签冲突")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 week06 数据分析图表")
    parser.add_argument(
        "--output-dir",
        default=str(WEEK_DIR / "outputs" / "analysis"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_chinese_font()

    train_rows = read_rows(WEEK_DIR / "data" / "train.json")
    val_rows = read_rows(WEEK_DIR / "data" / "val.json")
    train_lengths = [len(row["sentence"]) for row in train_rows]
    val_lengths = [len(row["sentence"]) for row in val_rows]
    quality = quality_stats(train_rows, val_rows)

    plot_class_distribution(
        train_rows,
        val_rows,
        figure_dir / "class_distribution.png",
    )
    plot_lengths(train_lengths, val_lengths, figure_dir / "length_distribution.png")
    plot_length_by_class(train_rows, figure_dir / "length_by_class.png")
    token_summary = plot_token_lengths(train_rows, figure_dir / "token_lengths.png")
    plot_quality(quality, figure_dir / "data_quality.png")

    train_counts = Counter(row["label"] for row in train_rows)
    val_counts = Counter(row["label"] for row in val_rows)
    class_rows = [
        {
            "label_id": index,
            "label": label,
            "train_count": train_counts[index],
            "train_ratio": train_counts[index] / len(train_rows),
            "val_count": val_counts[index],
        }
        for index, label in enumerate(LABELS)
    ]
    save_csv(output_dir / "class_stats.csv", class_rows)
    save_csv(
        output_dir / "data_quality.csv",
        [{"metric": key, "value": value} for key, value in quality.items()],
    )

    report = [
        "# week06 数据分析",
        "",
        f"- 训练集：{len(train_rows):,} 条",
        f"- 验证集：{len(val_rows):,} 条",
        f"- 字符长度均值：{statistics.mean(train_lengths):.2f}",
        f"- 字符长度 P95：{percentile(train_lengths, 0.95):.1f}",
        f"- 字符长度 P99：{percentile(train_lengths, 0.99):.1f}",
        f"- BERT token P95：{token_summary['p95']:.1f}",
        f"- BERT token P99：{token_summary['p99']:.1f}",
        f"- 训练与验证交叉同句：{quality['cross_overlap']:,} 条",
        "",
        "图表位于 `figures/`。",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"数据分析完成: {output_dir}")


if __name__ == "__main__":
    main()
