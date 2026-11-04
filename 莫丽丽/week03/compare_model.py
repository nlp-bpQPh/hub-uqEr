# -*- coding: utf-8 -*-
"""compare_models.py —— 可复现地比较 5 组 12 语种识别模型。

实验组：
  1. rnn_baseline   ：RNN + LayerNorm + Dropout（基线）
  2. lstm           ：将 RNN 替换为单向 LSTM
  3. bilstm         ：将 RNN 替换为双向 LSTM
  4. rnn_no_layernorm：只移除 LayerNorm
  5. rnn_no_dropout ：只移除 Dropout

所有实验使用相同数据划分、字符表、批大小、学习率、训练轮数、随机种子及测试集。
默认运行完整 8 epochs；可用 --epochs 1 进行快速冒烟测试。

运行：
    python compare_models.py
    python compare_models.py --epochs 8

输出：
    comparisons/v12/summary.csv          总对比表
    comparisons/v12/summary.json         包括逐语种准确率的详细结果
    comparisons/v12/models/*.pt         五个训练后的模型
    comparisons/v12/logs/*.log           每组训练日志
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPERIMENTS = [
    {
        "name": "rnn_baseline",
        "display_name": "RNN 基线",
        "model_type": "rnn",
        "flags": [],
        "layernorm": True,
        "dropout": True,
    },
    {
        "name": "lstm",
        "display_name": "单向 LSTM",
        "model_type": "lstm",
        "flags": [],
        "layernorm": True,
        "dropout": True,
    },
    {
        "name": "bilstm",
        "display_name": "双向 LSTM",
        "model_type": "bilstm",
        "flags": [],
        "layernorm": True,
        "dropout": True,
    },
    {
        "name": "rnn_no_layernorm",
        "display_name": "RNN（去掉 LayerNorm）",
        "model_type": "rnn",
        "flags": ["--no-layernorm"],
        "layernorm": False,
        "dropout": True,
    },
    {
        "name": "rnn_no_dropout",
        "display_name": "RNN（去掉 Dropout）",
        "model_type": "rnn",
        "flags": ["--no-dropout"],
        "layernorm": True,
        "dropout": False,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="训练并比较 5 种语种分类模型")
    parser.add_argument("--data-dir", default=str(HERE / "data" / "langid" / "v12"),
                        help="数据目录，默认 12 语种数据 data/langid/v12")
    parser.add_argument("--epochs", type=int, default=8, help="每组训练轮数，默认 8")
    parser.add_argument("--seed", type=int, default=42, help="所有实验组共同使用的随机种子")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU 线程数，默认 1")
    parser.add_argument("--output-dir", default="",
                        help="结果目录；默认 comparisons/v12 或 comparisons/v8")
    parser.add_argument("--skip-existing", action="store_true",
                        help="指标 JSON 已存在时跳过该组（用于续跑），不建议用在更换参数后")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs 必须大于等于 1")

    data_dir = Path(args.data_dir).resolve()
    if not (data_dir / "train.csv").is_file():
        raise SystemExit(f"找不到训练数据：{data_dir / 'train.csv'}")
    profile = data_dir.name if data_dir.name in {"v8", "v12"} else "custom"
    default_output = HERE / "comparisons" / profile
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    metric_dir = output_dir / "metrics"
    for directory in (output_dir, model_dir, log_dir, metric_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results = []
    train_script = HERE / "train_langid_cls.py"
    print("=" * 88)
    print("语种识别模型对比实验")
    print(f"数据目录：{data_dir}")
    print(f"训练轮数：{args.epochs}；线程数：{args.threads}；统一随机种子：{args.seed}")
    print("各组共享训练/验证/测试划分及字符表；每个实验组都从同一随机种子重新开始。")
    print("=" * 88)

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        model_path = model_dir / f"{experiment['name']}.pt"
        metrics_path = metric_dir / f"{experiment['name']}.json"
        log_path = log_dir / f"{experiment['name']}.log"

        if args.skip_existing and metrics_path.is_file():
            try:
                with metrics_path.open(encoding="utf-8") as f:
                    metrics = json.load(f)
                # 保存可随项目移动的相对模型路径，而不是当前机器上的绝对路径。
                metrics["model_path"] = os.path.relpath(model_path, output_dir)
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                print(f"[{index}/{len(EXPERIMENTS)}] 跳过已存在结果：{experiment['display_name']}")
                results.append({**experiment, **metrics})
                continue
            except (OSError, json.JSONDecodeError):
                pass

        cmd = [
            sys.executable, str(train_script),
            "--data-dir", str(data_dir),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--threads", str(args.threads),
            "--model-type", experiment["model_type"],
            "--save", str(model_path),
            "--metrics-out", str(metrics_path),
            *experiment["flags"],
        ]
        print(f"\n[{index}/{len(EXPERIMENTS)}] 开始：{experiment['display_name']}")
        print(f"    LayerNorm={experiment['layernorm']} | Dropout={experiment['dropout']}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                cmd, cwd=str(HERE), stdout=log_file, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        if process.returncode != 0:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-35:])
            except OSError:
                pass
            raise SystemExit(
                f"实验 {experiment['name']} 运行失败，退出码 {process.returncode}。\n"
                f"日志位置：{log_path}\n日志末尾：\n{tail}"
            )
        if not metrics_path.is_file():
            raise SystemExit(f"训练脚本未生成指标文件：{metrics_path}（请查看 {log_path}）")
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        # 指标文件保存相对于 output-dir 的路径，压缩包解压到其他目录也能定位模型。
        metrics["model_path"] = os.path.relpath(model_path, output_dir)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        results.append({**experiment, **metrics})
        print(f"    完成：test accuracy={metrics['test_accuracy']:.4f}，"
              f"macro-F1={metrics['macro_f1']:.4f}，耗时={metrics['train_seconds']:.1f}s")
        print(f"    日志：{log_path}")

    # 保持设定的实验顺序，便于直接比较每项改动的影响。
    csv_path = output_dir / "summary.csv"
    columns = [
        "name", "display_name", "model_type", "layernorm", "dropout",
        "num_parameters", "epochs", "train_size", "dev_size", "test_size",
        "train_seconds", "dev_accuracy", "test_accuracy", "macro_f1", "model_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 104)
    print("汇总结果（最终 epoch；建议重点比较 accuracy 和 macro-F1）")
    print(f"{'实验组':<25}{'LayerNorm':>11}{'Dropout':>10}{'参数量':>13}"
          f"{'Dev Acc':>11}{'Test Acc':>12}{'Macro-F1':>12}{'训练秒数':>12}")
    print("-" * 104)
    for row in results:
        print(f"{row['display_name']:<25}{str(row['layernorm']):>11}"
              f"{str(row['dropout']):>10}{row.get('num_parameters', 0):>13,}"
              f"{row.get('dev_accuracy', 0):>11.4f}"
              f"{row.get('test_accuracy', 0):>12.4f}"
              f"{row.get('macro_f1', 0):>12.4f}"
              f"{row.get('train_seconds', 0):>12.1f}")
    print("\n文件已写入：")
    print(f"  CSV 对比表：{csv_path}")
    print(f"  详细 JSON： {json_path}")
    print(f"  模型文件：  {model_dir}")
    print(f"  训练日志：  {log_dir}")



if __name__ == "__main__":
    main()
