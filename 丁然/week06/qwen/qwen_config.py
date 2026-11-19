"""Qwen LoRA 分类任务配置。"""

from pathlib import Path

import torch


QWEN_DIR = Path(__file__).resolve().parent
WEEK_DIR = QWEN_DIR.parent
DEFAULT_MODEL_PATH = (
    "~/.cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
)
DEFAULT_TRAIN_PATH = WEEK_DIR / "data" / "train.json"
DEFAULT_VAL_PATH = WEEK_DIR / "data" / "val.json"
DEFAULT_OUTPUT_ROOT = WEEK_DIR / "outputs" / "qwen"
MAX_LENGTH = 48
NUM_LABELS = 15
SHOTS_PER_CLASS = 1
EVAL_LIMIT = 500


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
