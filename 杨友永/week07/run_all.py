"""一键跑三组实验：BERT-linear / BERT-CRF / Qwen-LoRA。

示例：
    python run_all.py                       # 全量
    python run_all.py --limit 2000 --eval-limit 500   # 冒烟
    python run_all.py --only bert-linear,bert-crf     # 只跑 BERT
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="逗号分隔：bert-linear,bert-crf,llm-lora")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--epochs-bert", type=int, default=3)
    p.add_argument("--epochs-llm", type=int, default=2)
    p.add_argument("--device", default=None)
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd), flush=True)
    t = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"[exit {proc.returncode}] {time.time() - t:.0f}s", flush=True)
    return proc.returncode


def main():
    args = parse_args()
    selected = set(args.only.split(",")) if args.only else {"bert-linear", "bert-crf", "llm-lora"}
    common = []
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.eval_limit:
        common += ["--eval-limit", str(args.eval_limit)]
    if args.device:
        common += ["--device", args.device]

    failures = []
    if "bert-linear" in selected:
        rc = run([sys.executable, "-m", "bert.train", "--head", "linear",
                  "--epochs", str(args.epochs_bert), "--batch-size", "32",
                  "--lr", "3e-5", *common])
        failures += ["bert-linear"] if rc else []

    if "bert-crf" in selected:
        rc = run([sys.executable, "-m", "bert.train", "--head", "crf",
                  "--epochs", str(args.epochs_bert), "--batch-size", "16",
                  "--lr", "3e-5", *common])
        failures += ["bert-crf"] if rc else []

    if "llm-lora" in selected:
        rc = run([sys.executable, "-m", "llm.train",
                  "--epochs", str(args.epochs_llm), "--batch-size", "8",
                  "--grad-accum", "4", "--lr", "2e-4", "--lora-r", "16",
                  *common])
        failures += ["llm-lora"] if rc else []

    print("\n[report] 生成对比报告 …")
    run([sys.executable, "report.py"])
    if failures:
        print(f"[warn] 以下实验失败: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
