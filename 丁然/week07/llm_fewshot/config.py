"""LLM few-shot 配置。"""

from __future__ import annotations

from pathlib import Path

import torch


ICL_DIR = Path(__file__).resolve().parent
WEEK_DIR = ICL_DIR.parent

DEFAULT_MODEL_PATH = (
    "~/.cache/modelscope/models/"
    "Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
)
DEFAULT_SCHEMA_PATH = WEEK_DIR / "data" / "event_schema.json"
DEFAULT_TRAIN_EXTRACT_PATH = WEEK_DIR / "data" / "train_extract.jsonl"
DEFAULT_OUTPUT_ROOT = WEEK_DIR / "outputs" / "llm_fewshot"


def get_device(name: str = "auto") -> torch.device:
    """自动选择推理设备。"""

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

