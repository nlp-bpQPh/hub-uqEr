"""事件抽取模型配置。"""

from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent
WEEK_DIR = PROJECT_DIR.parent

DEFAULT_MODEL_PATH = "~/Downloads/模型/bert/bert-base-chinese"
DEFAULT_TRAIN_PATH = WEEK_DIR / "data" / "train_extract.jsonl"
DEFAULT_SCHEMA_PATH = WEEK_DIR / "data" / "event_schema.json"
DEFAULT_OUTPUT_ROOT = WEEK_DIR / "outputs"

# 256 已覆盖绝大多数标注；384 会明显拉长同批次中的最长序列。
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 1


def get_device(name: str = "auto") -> torch.device:
    """自动选择当前可用的训练设备。"""

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

