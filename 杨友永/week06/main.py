"""统一入口。

    python main.py overview            # 数据总览
    python main.py bert --exp all      # BERT 七个实验
    python main.py llm --mode all      # LLM：LoRA + 全参 SFT
    python main.py report              # 生成对比报告
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("overview", help="数据总览")
    p = sub.add_parser("bert", help="BERT 实验")
    p.add_argument("--exp", default="all")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)

    p = sub.add_parser("llm", help="LLM 实验")
    p.add_argument("--mode", default="all", choices=["all", "lora", "full"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)

    sub.add_parser("report", help="生成对比报告")

    a = ap.parse_args()
    if a.cmd == "overview":
        import data_overview
        sys.argv = [sys.argv[0], "--report"]
        data_overview.main()
    elif a.cmd == "bert":
        import bert.run_bert as rb
        sys.argv = [sys.argv[0], "--exp", a.exp] + (
            ["--epochs", str(a.epochs)] if a.epochs else [])
        rb.main()
    elif a.cmd == "llm":
        import llm.run_llm as rl
        sys.argv = [sys.argv[0], "--mode", a.mode] + (
            ["--epochs", str(a.epochs)] if a.epochs else []) + (
            ["--limit", str(a.limit)] if a.limit else [])
        rl.main()
    else:
        from report import build_report
        build_report()


if __name__ == "__main__":
    main()
