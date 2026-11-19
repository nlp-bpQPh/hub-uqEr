"""Qwen 主干、LoRA 和分类头。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging


transformers_logging.set_verbosity_error()


class QwenClassifier(nn.Module):
    """使用最后一个有效 token 表示的 Qwen 分类器。"""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_labels: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> SimpleNamespace:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = outputs.last_hidden_state[rows, lengths]
        return SimpleNamespace(logits=self.classifier(self.dropout(pooled)))


def load_tokenizer(model_path: str | Path) -> Any:
    path = str(Path(model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_lora_model(
    model_path: str | Path,
    num_labels: int = 15,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> QwenClassifier:
    path = str(Path(model_path).expanduser().resolve())
    encoder = AutoModel.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.float32,
    )
    encoder = get_peft_model(
        encoder,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )
    return QwenClassifier(
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        num_labels=num_labels,
    )


def load_lora_model(model_dir: str | Path) -> tuple[QwenClassifier, Any, dict[str, Any], torch.device]:
    from transformers import AutoConfig

    from qwen_config import get_device

    directory = Path(model_dir)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(
        directory / "base_config",
        local_files_only=True,
    )
    base = AutoModel.from_config(config, dtype=torch.float32)
    encoder = PeftModel.from_pretrained(base, directory)
    model = QwenClassifier(
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        num_labels=meta["num_labels"],
        dropout=0.0,
    )
    model.classifier.load_state_dict(
        torch.load(
            directory / "classifier_head.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    device = get_device()
    model.to(device).eval()
    return model, load_tokenizer(directory), meta, device
