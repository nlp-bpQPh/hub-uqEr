"""BERT 实验入口。

    python -m bert.run_bert --exp all
    python -m bert.run_bert --exp cls
    python -m bert.run_bert --exp mean --epochs 2
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bert.train import BertConfig, train  # noqa: E402

from paths import bert_dir  # noqa: E402

MODEL_PATH = bert_dir()

# 七个实验：四种池化 + 三种训练技巧（技巧都加在 cls 基线上）
PRESETS: dict[str, BertConfig] = {
    "cls":       BertConfig(pooling="cls"),
    "mean":      BertConfig(pooling="mean"),
    "max":       BertConfig(pooling="max"),
    "cls_mean":  BertConfig(pooling="cls_mean"),
    "cls_fgm":   BertConfig(pooling="cls", fgm_eps=1.0),
    "cls_ls":    BertConfig(pooling="cls", label_smoothing=0.1),
    "cls_focal": BertConfig(pooling="cls", focal_gamma=2.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", *PRESETS])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    names = list(PRESETS) if a.exp == "all" else [a.exp]
    for name in names:
        cfg = PRESETS[name]
        if a.epochs is not None:
            cfg = replace(cfg, epochs=a.epochs)
        if a.lr is not None:
            cfg = replace(cfg, lr=a.lr)
        if a.batch_size is not None:
            cfg = replace(cfg, batch_size=a.batch_size)
        print(f"\n===== {name} =====")
        train(cfg, run_id=f"bert_{name}", model_path=MODEL_PATH, device_str=a.device)

    if len(names) > 1:
        from report import build_report
        build_report()


if __name__ == "__main__":
    main()
