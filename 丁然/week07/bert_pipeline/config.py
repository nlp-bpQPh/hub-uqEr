"""两阶段事件流水线配置。"""

from __future__ import annotations

from pathlib import Path

import torch


PIPELINE_DIR = Path(__file__).resolve().parent
WEEK_DIR = PIPELINE_DIR.parent

DEFAULT_MODEL_PATH = "~/Downloads/模型/bert/bert-base-chinese"
DEFAULT_DETECT_PATH = WEEK_DIR / "data" / "train_detect.jsonl"
DEFAULT_EXTRACT_PATH = WEEK_DIR / "data" / "train_extract.jsonl"
DEFAULT_SCHEMA_PATH = WEEK_DIR / "data" / "event_schema.json"
DEFAULT_OUTPUT_ROOT = WEEK_DIR / "outputs" / "pipeline"
DEFAULT_SPLIT_PATH = DEFAULT_OUTPUT_ROOT / "split.json"

DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 1


def get_device(name: str = "auto") -> torch.device:
    """自动选择可用的训练设备。"""

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

