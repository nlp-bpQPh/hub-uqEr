"""一次运行全部 BERT 消融实验，每个实验固定训练一轮。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


BERT_DIR = Path(__file__).resolve().parent
WEEK_DIR = BERT_DIR.parent
EPOCHS = 1

# 脚本会把右侧参数显式传给 bert_train.py，不依赖训练脚本的默认值。
EXPERIMENTS: list[tuple[str, list[str], str]] = [
    ("mean", ["--pooling", "mean"], "Mean 池化"),
    ("max", ["--pooling", "max"], "Max 池化"),
    ("warmup", ["--warmup-ratio", "0.1"], "10% Warmup"),
    ("focal", ["--loss", "focal"], "Focal Loss"),
    ("fgm", ["--fgm"], "FGM 对抗训练"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_best_model(candidate: Path, best_dir: Path) -> None:
    """先备份旧模型，成功后删除备份，确保只保留一份权重。"""

    backup = best_dir.with_name(best_dir.name + "_backup")
    if backup.exists():
        shutil.rmtree(backup)
    if best_dir.exists():
        best_dir.rename(backup)
    try:
        shutil.copytree(candidate, best_dir)
    except Exception:
        if best_dir.exists():
            shutil.rmtree(best_dir)
        if backup.exists():
            backup.rename(best_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def collect_summary(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = ["cls", *[name for name, _, _ in EXPERIMENTS]]
    for name in names:
        metrics_path = output_root / name / "metrics.json"
        meta_path = output_root / name / "meta.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        meta = read_json(meta_path) if meta_path.exists() else {}
        rows.append(
            {
                "experiment": name,
                "pooling": meta.get("pooling", "cls"),
                "warmup_ratio": meta.get("warmup_ratio", 0.0),
                "loss": meta.get("loss", "ce"),
                "fgm": meta.get("fgm", False),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    return sorted(rows, key=lambda row: row["macro_f1"], reverse=True)


def write_summary(output_root: Path) -> list[dict[str, Any]]:
    rows = collect_summary(output_root)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# BERT 全部实验（每个 1 epoch）",
        "",
        "| 实验 | 池化 | Warmup | Loss | FGM | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['pooling']} | "
            f"{row['warmup_ratio']:.2f} | {row['loss']} | {row['fgm']} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} |"
        )
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一次运行全部 BERT 消融实验，每个实验固定 1 epoch"
    )
    parser.add_argument(
        "--output-root",
        default=str(WEEK_DIR / "outputs" / "bert"),
    )
    parser.add_argument(
        "--best-model-dir",
        default=str(WEEK_DIR / "outputs" / "bert" / "best_model"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="自检时只用少量样本")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    best_dir = Path(args.best_model_dir)
    cls_metrics_path = output_root / "cls" / "metrics.json"
    if not cls_metrics_path.exists():
        raise FileNotFoundError(f"缺少已有 cls 结果: {cls_metrics_path}")
    if not args.smoke and not (best_dir / "model.pt").exists():
        raise FileNotFoundError(f"缺少已有最优模型: {best_dir / 'model.pt'}")

    best_metrics = read_json(cls_metrics_path)
    best_name = "cls"
    print(
        f"已有 cls: accuracy={best_metrics['accuracy']:.4f} "
        f"macro_f1={best_metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(f"计划运行 {len(EXPERIMENTS)} 个实验，每个固定 {EPOCHS} epoch：", flush=True)
    for index, (name, extra_args, description) in enumerate(EXPERIMENTS, start=1):
        print(f"  {index}. {name}: {description} {extra_args}", flush=True)

    for index, (name, extra_args, description) in enumerate(EXPERIMENTS, start=1):
        print(
            f"\n[{index}/{len(EXPERIMENTS)}] 开始 {name}（{description}，1 epoch）",
            flush=True,
        )
        experiment_dir = output_root / name
        with tempfile.TemporaryDirectory(prefix=f"bert_{name}_") as temp_model_dir:
            command = [
                sys.executable,
                str(BERT_DIR / "bert_train.py"),
                "--output-dir",
                str(experiment_dir),
                "--model-dir",
                temp_model_dir,
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                *extra_args,
            ]
            if args.smoke:
                command.append("--smoke")
            subprocess.run(command, check=True)

            metrics = read_json(experiment_dir / "metrics.json")
            if not args.smoke and metrics["macro_f1"] > best_metrics["macro_f1"]:
                replace_best_model(Path(temp_model_dir), best_dir)
                best_metrics = metrics
                best_name = name
                print(f"模型已更新为当前最优: {name}", flush=True)
            else:
                print(
                    f"{name} 未超过当前最优 {best_name}: "
                    f"{metrics['macro_f1']:.4f} <= "
                    f"{best_metrics['macro_f1']:.4f}",
                    flush=True,
                )

    rows = write_summary(output_root)
    if not args.smoke:
        best_text = json.dumps(
            {"experiment": best_name, "metrics": best_metrics},
            ensure_ascii=False,
            indent=2,
        )
        (output_root / "best_experiment.json").write_text(
            best_text,
            encoding="utf-8",
        )
        (best_dir / "metrics.json").write_text(
            json.dumps(best_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (best_dir / "best_experiment.json").write_text(
            best_text,
            encoding="utf-8",
        )

    print("\n全部实验完成")
    print(f"最优实验: {best_name}")
    if args.smoke:
        print("当前为冒烟自检，没有替换正式最优模型")
    else:
        print(f"唯一模型权重: {best_dir / 'model.pt'}")
    print(f"结果汇总: {output_root / 'summary.csv'}")
    print(f"共 {len(rows)} 个实验结果")


if __name__ == "__main__":
    main()
