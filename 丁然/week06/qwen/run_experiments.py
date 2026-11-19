"""运行 Qwen LoRA、zero-shot ICL 和 few-shot ICL 三个实验。"""

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


QWEN_DIR = Path(__file__).resolve().parent
WEEK_DIR = QWEN_DIR.parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_best(candidate: Path, best_dir: Path) -> None:
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


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen LoRA 与 in-context learning 对比"
    )
    parser.add_argument("--output-root", default=str(WEEK_DIR / "outputs" / "qwen"))
    parser.add_argument(
        "--best-model-dir",
        default=str(WEEK_DIR / "outputs" / "qwen" / "best_model"),
    )
    parser.add_argument("--shots-per-class", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    best_dir = Path(args.best_model_dir)
    common = [
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.smoke:
        common.append("--smoke")

    print("正式对比包含：LoRA 微调、zero-shot ICL、few-shot ICL", flush=True)

    for name, mode in (("zero_shot_icl", "zero_shot"), ("few_shot_icl", "few_shot")):
        print(f"\n开始 {name}", flush=True)
        command = [
            sys.executable,
            str(QWEN_DIR / "icl_evaluate.py"),
            "--mode",
            mode,
            "--output-dir",
            str(output_root / name),
            "--eval-limit",
            str(args.eval_limit),
            *common,
        ]
        if mode == "few_shot":
            command.extend(
                ["--shots-per-class", str(args.shots_per_class)]
            )
        run_command(command)

    print("\n开始 lora（全量训练集，1 epoch）", flush=True)
    lora_dir = output_root / "lora"
    with tempfile.TemporaryDirectory(prefix="qwen_lora_") as temp_model_dir:
        command = [
            sys.executable,
            str(QWEN_DIR / "qwen_train.py"),
            "--output-dir",
            str(lora_dir),
            "--model-dir",
            temp_model_dir,
            "--eval-limit",
            str(args.eval_limit),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            *common,
        ]
        run_command(command)
        if not args.smoke:
            replace_best(Path(temp_model_dir), best_dir)

    rows: list[dict[str, Any]] = []
    for name in ("lora", "zero_shot_icl", "few_shot_icl"):
        metrics = read_json(output_root / name / "metrics.json")
        rows.append(
            {
                "experiment": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    rows.sort(key=lambda row: row["macro_f1"], reverse=True)

    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "best_experiment.json").write_text(
        json.dumps(
            {
                "best_method": rows[0]["experiment"],
                "metrics": rows[0],
                "note": "ICL 不产生训练权重；best_model 保存的是唯一训练的 LoRA 模型。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n全部实验完成", flush=True)
    for row in rows:
        print(
            f"{row['experiment']}: accuracy={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f}",
            flush=True,
        )
    if args.smoke:
        print("当前为冒烟自检，没有保存正式 LoRA adapter", flush=True)
    else:
        print(f"LoRA adapter: {best_dir}", flush=True)
    print(f"结果汇总: {output_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
