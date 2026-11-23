"""模型与数据路径解析。

项目目录调整过多次（pretrain/ 被移走、数据集挪到 data/），
所以这里不写死路径，按优先级依次探测；环境变量可覆盖。

    CLASSIFY_BERT_DIR   指定 BERT 目录
    CLASSIFY_QWEN_DIR   指定 Qwen3 目录
    PRODUCT_DATA_DIR    指定数据集目录（见 dataset.py）
"""
from __future__ import annotations

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
HW = BASE.parent                      # homework/
DL = HW.parent                        # DeepLearning/

BERT_NAME = "bert-base-chinese"
QWEN_NAME = "Qwen3-0.6B"


def _resolve(name: str, env_key: str, extra: list[Path]) -> Path:
    env = os.environ.get(env_key)
    candidates = ([Path(env)] if env else []) + [
        BASE / "pretrain" / name,
        BASE / "pretrain_models" / name,
        HW / name,
        DL / "model" / name,
        Path("/root/text_class/pretrain_models") / name,
        Path("/root/pretrain_models") / name,
        *extra,
    ]
    for path in candidates:
        if (path / "config.json").exists():
            return path
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"找不到模型 {name}，已尝试:\n{tried}\n"
                            f"可用环境变量 {env_key} 显式指定")


def bert_dir() -> Path:
    return _resolve(BERT_NAME, "CLASSIFY_BERT_DIR", [])


def qwen_dir() -> Path:
    return _resolve(QWEN_NAME, "CLASSIFY_QWEN_DIR", [])


if __name__ == "__main__":
    for fn in (bert_dir, qwen_dir):
        try:
            print(f"{fn.__name__}: {fn()}")
        except FileNotFoundError as e:
            print(e)
