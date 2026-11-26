"""训练两阶段流水线的全部组件并生成对比结果。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PIPELINE_DIR = Path(__file__).resolve().parent
WEEK_DIR = PIPELINE_DIR.parent
DEFAULT_OUTPUT_ROOT = WEEK_DIR / "outputs" / "pipeline"


def run_command(command: list[str], log_path: Path) -> float:
    """执行子进程，同时把输出写到终端和日志。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n运行：" + " ".join(command), flush=True)
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PIPELINE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"命令失败，日志={log_path}")
    return time.perf_counter() - start_time


def append_args(
    command: list[str],
    args: argparse.Namespace,
    include_extract: bool,
) -> None:
    """把公共数据、BERT 和设备参数追加到子命令。"""

    values = {
        "--detect-path": args.detect_path,
        "--schema-path": args.schema_path,
        "--model-path": args.model_path,
        "--max-length": args.max_length,
        "--batch-size": args.batch_size,
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
        "--freeze-layers": args.freeze_layers,
        "--seed": args.seed,
        "--device": args.device,
    }
    if include_extract:
        values["--extract-path"] = args.extract_path
    for name, value in values.items():
        if value is not None:
            command.extend([name, str(value)])
    if args.smoke:
        command.append("--smoke")
    if args.data_debug_count is not None:
        command.extend(
            ["--data-debug-count", str(args.data_debug_count)]
        )


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并评估两阶段流水线")
    parser.add_argument(
        "--stage2-methods",
        nargs="+",
        choices=["linear", "crf"],
        default=["linear", "crf"],
    )
    parser.add_argument("--detect-path", default=None)
    parser.add_argument("--extract-path", default=None)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--freeze-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-debug-count", type=int, default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="two_stage")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.output_root).expanduser() / args.run_name
    split_path = run_dir / "split.json"
    stage1_dir = run_dir / "stage1"
    results = []

    stage1_command = [
        sys.executable,
        str(PIPELINE_DIR / "train_stage1.py"),
        "--output-dir",
        str(stage1_dir),
        "--split-path",
        str(split_path),
        "--overwrite-split",
    ]
    append_args(stage1_command, args, include_extract=False)
    stage1_duration = run_command(
        stage1_command,
        run_dir / "logs" / "stage1.log",
    )
    stage1_metrics = load_metrics(stage1_dir / "metrics.json")
    results.append(
        {
            "method": "stage1_detector",
            "duration_seconds": stage1_duration,
            "metrics": stage1_metrics,
        }
    )

    for decoder in args.stage2_methods:
        stage2_dir = run_dir / "stage2" / decoder
        stage2_command = [
            sys.executable,
            str(PIPELINE_DIR / "train_stage2.py"),
            "--decoder",
            decoder,
            "--output-dir",
            str(stage2_dir),
            "--split-path",
            str(split_path),
        ]
        append_args(stage2_command, args, include_extract=True)
        duration = run_command(
            stage2_command,
            run_dir / "logs" / f"stage2_{decoder}.log",
        )

        evaluate_path = run_dir / "evaluation" / f"{decoder}.json"
        evaluate_command = [
            sys.executable,
            str(PIPELINE_DIR / "evaluate_pipeline.py"),
            "--stage1-dir",
            str(stage1_dir),
            "--stage2-dir",
            str(stage2_dir),
            "--split-path",
            str(split_path),
            "--output",
            str(evaluate_path),
        ]
        if args.detect_path:
            evaluate_command.extend(["--detect-path", args.detect_path])
        if args.extract_path:
            evaluate_command.extend(["--extract-path", args.extract_path])
        if args.device:
            evaluate_command.extend(["--device", args.device])
        if args.seed is not None:
            evaluate_command.extend(["--seed", str(args.seed)])
        run_command(
            evaluate_command,
            run_dir / "logs" / f"evaluate_{decoder}.log",
        )
        stage2_metrics = load_metrics(stage2_dir / "metrics.json")
        pipeline_metrics = load_metrics(evaluate_path)
        results.append(
            {
                "method": f"pipeline_{decoder}",
                "duration_seconds": duration,
                "stage2_metrics": stage2_metrics,
                "end_to_end_metrics": pipeline_metrics,
            }
        )

    summary = {
        "run_name": args.run_name,
        "stage1": stage1_metrics,
        "pipelines": results[1:],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 两阶段事件流水线",
        "",
        "## 第一阶段",
        "",
        f"- 类型 micro-F1：{stage1_metrics['type_micro_f1']:.4f}",
        f"- 类型 macro-F1：{stage1_metrics['type_macro_f1']:.4f}",
        f"- 多标签完全匹配：{stage1_metrics['subset_accuracy']:.4f}",
        f"- 阈值：{stage1_metrics['threshold']:.2f}",
        "",
        "## 第二阶段与端到端",
        "",
        "| 第二阶段 | 给定类型 触发词 F1 | 给定类型 论元 F1 | 端到端 类型召回 | 端到端 论元 F1 | 端到端 完整事件 EM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results[1:]:
        stage2 = result["stage2_metrics"]
        end_to_end = result["end_to_end_metrics"]
        lines.append(
            f"| {result['method']} | "
            f"{stage2['trigger_f1']:.4f} | "
            f"{stage2['arg_f1']:.4f} | "
            f"{end_to_end['stage1_gold_type_recall']:.4f} | "
            f"{end_to_end['pipeline_end_to_end']['arg_f1']:.4f} | "
            f"{end_to_end['pipeline_end_to_end']['event_exact_match']:.4f} |"
        )
    (run_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"\n两阶段结果: {(run_dir / 'comparison.md').resolve()}")


if __name__ == "__main__":
    main()

