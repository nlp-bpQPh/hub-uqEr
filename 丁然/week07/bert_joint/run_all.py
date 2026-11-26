"""依次训练多个事件抽取方案并生成统一对比结果。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
WEEK_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_ROOT = WEEK_DIR / "outputs" / "experiments"

# 所有方案使用同一组指标，后续两段式也必须输出这些字段。
METRIC_COLUMNS = [
    "event_type_accuracy",
    "type_macro_f1",
    "trigger_exact_match",
    "trigger_precision",
    "trigger_recall",
    "trigger_f1",
    "arg_exact_match",
    "arg_precision",
    "arg_recall",
    "arg_f1",
    "event_exact_match",
    "selection_score",
    "loss",
]


def build_parser() -> argparse.ArgumentParser:
    # 公共训练参数会原样传给 train.py。
    parser = argparse.ArgumentParser(
        description="批量训练 linear 和 crf，并输出统一对比",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["linear", "crf"],
        default=["linear", "crf"],
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="linear_crf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--freeze-layers", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-debug-count", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="不训练，只读取当前 run-name 下已有结果并重新生成对比",
    )
    parser.add_argument(
        "--extra-result",
        action="append",
        default=[],
        metavar="名称=目录",
        help="把已有结果加入对比，目录中需包含 meta.json 和 metrics.json",
    )
    return parser


def _optional_args(args: argparse.Namespace) -> list[str]:
    """只向 train.py 传递用户明确设置的参数。"""

    # 值为 None 表示沿用 train.py 自己的默认配置。
    values = {
        "--model-path": args.model_path,
        "--train-path": args.train_path,
        "--schema-path": args.schema_path,
        "--max-length": args.max_length,
        "--batch-size": args.batch_size,
        "--eval-batch-size": args.eval_batch_size,
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
        "--weight-decay": args.weight_decay,
        "--freeze-layers": args.freeze_layers,
        "--val-ratio": args.val_ratio,
        "--max-train": args.max_train,
        "--max-val": args.max_val,
        "--seed": args.seed,
        "--device": args.device,
        "--data-debug-count": args.data_debug_count,
    }
    result: list[str] = []
    for name, value in values.items():
        if value is not None:
            result.extend([name, str(value)])
    if args.smoke:
        result.append("--smoke")
    return result


def run_training(
    method: str,
    output_dir: Path,
    log_path: Path,
    common_args: list[str],
) -> float:
    """运行单个训练任务，同时把完整日志写入文件。"""

    # 使用 subprocess 隔离两个任务，避免 CRF 状态影响下一次训练。
    command = [
        sys.executable,
        str(PROJECT_DIR / "train.py"),
        "--decoder",
        method,
        "--output-dir",
        str(output_dir),
        *common_args,
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    print(f"\n开始训练 {method}", flush=True)
    print("命令：" + " ".join(command), flush=True)

    # 一边输出到终端，一边写入日志，便于观察和事后排查。
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
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

    duration = time.perf_counter() - start_time
    if return_code != 0:
        raise RuntimeError(
            f"{method} 训练失败，返回码={return_code}，日志={log_path}"
        )
    return duration


def load_result(
    method: str,
    result_dir: Path,
    status: str = "success",
    duration_seconds: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """统一读取各方案的元数据和指标。"""

    # 先建立公共字段，即使训练失败也能写入对比表。
    result: dict[str, Any] = {
        "method": method,
        "status": status,
        "duration_seconds": duration_seconds,
        "result_dir": str(result_dir.resolve()),
        "error": error,
    }
    meta_path = result_dir / "meta.json"
    metrics_path = result_dir / "metrics.json"
    if status != "success":
        return result
    if not meta_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"{result_dir} 缺少 meta.json 或 metrics.json"
        )
    result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    result["metrics"] = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )
    return result


def parse_extra_result(value: str) -> tuple[str, Path]:
    # 外部结果采用“名称=目录”的参数形式。
    if "=" not in value:
        raise ValueError(
            f"--extra-result 格式应为 名称=目录，收到: {value}"
        )
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise ValueError(f"--extra-result 参数不完整: {value}")
    return name, Path(directory).expanduser()


def write_comparison(
    output_dir: Path,
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """同时输出 JSON、CSV 和 Markdown，方便后续追加两段式结果。"""

    # 失败方案保留在表中，但只有成功方案参与最佳方案选择。
    successful = [
        result
        for result in results
        if result["status"] == "success"
        and "metrics" in result
    ]
    best_method = None
    if successful:
        # selection_score 综合类型、触发词和论元三项指标。
        best = max(
            successful,
            key=lambda result: result["metrics"].get(
                "selection_score",
                float("-inf"),
            ),
        )
        best_method = best["method"]

    # 完整 JSON 保留元数据，便于以后复现实验。
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_method": best_method,
        "best_metric": "selection_score",
        "training_args": {
            "methods": args.methods,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "smoke": args.smoke,
        },
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV 适合用表格软件或 pandas 做后续统计。
    with (output_dir / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["method", "status", "duration_seconds", *METRIC_COLUMNS]
        )
        for result in results:
            metrics = result.get("metrics", {})
            writer.writerow(
                [
                    result["method"],
                    result["status"],
                    (
                        f"{result['duration_seconds']:.2f}"
                        if result["duration_seconds"] is not None
                        else ""
                    ),
                    *[metrics.get(column, "") for column in METRIC_COLUMNS],
                ]
            )

    # Markdown 提供人可直接阅读的横向对比。
    lines = [
        "# 事件抽取方案对比",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 最佳方案：{best_method or '无'}",
        f"- 选择指标：`{summary['best_metric']}`",
        "",
        "| 方法 | 状态 | 类型准确率 | 类型 F1 | 触发词 EM | 触发词 F1 | 论元 EM | 论元 F1 | 完整事件 EM | 选择分数 | 耗时(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result.get("metrics", {})
        duration = result["duration_seconds"]
        lines.append(
            "| {method} | {status} | {type_acc:.4f} | {type_f1:.4f} | "
            "{trigger_em:.4f} | {trigger_f1:.4f} | "
            "{arg_em:.4f} | {arg_f1:.4f} | "
            "{event_em:.4f} | {score:.4f} | {duration} |".format(
                method=result["method"],
                status=result["status"],
                type_acc=metrics.get(
                    "event_type_accuracy",
                    metrics.get("type_accuracy", 0.0),
                ),
                type_f1=metrics.get("type_macro_f1", 0.0),
                trigger_em=metrics.get("trigger_exact_match", 0.0),
                trigger_f1=metrics.get("trigger_f1", 0.0),
                arg_em=metrics.get("arg_exact_match", 0.0),
                arg_f1=metrics.get("arg_f1", 0.0),
                event_em=metrics.get("event_exact_match", 0.0),
                score=metrics.get("selection_score", 0.0),
                duration=(
                    f"{duration:.2f}" if duration is not None else "-"
                ),
            )
        )
    lines.extend(
        [
            "",
            "后续两段式结果可放入独立目录，再通过以下命令追加：",
            "",
            "```bash",
            "python run_all.py --compare-only --run-name linear_crf "
            "--extra-result two_stage=../outputs/two_stage",
            "```",
            "",
        ]
    )
    (output_dir / "comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    # 1. 读取批量训练参数。
    args = build_parser().parse_args()

    # 2. 所有方案放在同一个实验目录下。
    # 每个实验名对应独立目录，避免不同运行互相覆盖。
    run_dir = Path(args.output_root).expanduser() / args.run_name
    logs_dir = run_dir / "logs"

    # 3. 生成发送给 train.py 的公共参数。
    common_args = _optional_args(args)
    results: list[dict[str, Any]] = []

    # 4. 按顺序训练，避免两个 BERT 同时占用内存。
    for method in args.methods:
        result_dir = run_dir / method
        try:
            if args.compare_only:
                # 4.1 对比模式直接读取已有结果。
                results.append(load_result(method, result_dir))
            else:
                # 4.2 每个方案使用独立日志文件。
                log_path = logs_dir / f"{method}.log"
                # 4.3 启动子进程训练，并记录总耗时。
                duration = run_training(
                    method,
                    result_dir,
                    log_path,
                    common_args,
                )
                # 4.4 读取训练脚本生成的元数据和指标。
                results.append(
                    load_result(
                        method,
                        result_dir,
                        duration_seconds=duration,
                    )
                )
        except Exception as error:
            # 4.5 单方案失败时继续处理其他方案。
            print(f"{method} 失败: {error}", file=sys.stderr, flush=True)
            results.append(
                load_result(
                    method,
                    result_dir,
                    status="failed",
                    error=str(error),
                )
            )

    # 5. 追加两段式或其他已有实验结果。
    for extra in args.extra_result:
        # 5.1 解析“名称=结果目录”。
        name, result_dir = parse_extra_result(extra)
        try:
            # 5.2 只要求目录中存在 meta.json 和 metrics.json。
            results.append(load_result(name, result_dir))
        except Exception as error:
            # 5.3 外部结果错误时记录失败状态。
            results.append(
                load_result(
                    name,
                    result_dir,
                    status="failed",
                    error=str(error),
                )
            )

    # 6. 生成 JSON、CSV 和 Markdown 对比文件。
    summary = write_comparison(run_dir, results, args)

    # 7. 输出报告位置和最佳方案。
    print(f"\n对比结果: {(run_dir / 'comparison.md').resolve()}", flush=True)
    print(f"最佳方案: {summary['best_method']}", flush=True)

    # 8. 至少一个方案成功时返回 0，否则返回失败码。
    return 0 if any(result["status"] == "success" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

