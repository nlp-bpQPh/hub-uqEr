"""BERT 分类任务配置。"""

from pathlib import Path

import torch


BERT_DIR = Path(__file__).resolve().parent
WEEK_DIR = BERT_DIR.parent
DEFAULT_MODEL_PATH = "~/Downloads/模型/bert/bert-base-chinese"
DEFAULT_TRAIN_PATH = WEEK_DIR / "data" / "train.json"
DEFAULT_VAL_PATH = WEEK_DIR / "data" / "val.json"
DEFAULT_OUTPUT_DIR = WEEK_DIR / "outputs" / "bert" / "cls"
DEFAULT_MODEL_DIR = WEEK_DIR / "outputs" / "bert" / "best_model"
MAX_LENGTH = 48
NUM_LABELS = 15


def get_device(name: str = "auto") -> torch.device:
    """自动选择当前可用的训练设备。"""

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
