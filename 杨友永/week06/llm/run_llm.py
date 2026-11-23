"""LLM 实验入口。

    python -m llm.run_llm --mode all
    python -m llm.run_llm --mode lora --epochs 2
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.train import LLMConfig, train  # noqa: E402

from paths import qwen_dir  # noqa: E402

MODEL_DIR = qwen_dir()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "lora", "full"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfgs = {
        "lora": LLMConfig(mode="lora", lr=2e-4),
        "full": LLMConfig(mode="full", lr=2e-5, batch_size=4),
    }
    modes = ["lora", "full"] if a.mode == "all" else [a.mode]
    for mode in modes:
        cfg = cfgs[mode]
        if a.epochs is not None:
            cfg = replace(cfg, epochs=a.epochs)
        if a.batch_size is not None:
            cfg = replace(cfg, batch_size=a.batch_size)
        if a.limit is not None:
            cfg = replace(cfg, limit=a.limit)
        print(f"\n===== {mode} =====")
        train(cfg, MODEL_DIR, device_str=a.device)

    if len(modes) > 1:
        from report import build_report
        build_report()


if __name__ == "__main__":
    main()
