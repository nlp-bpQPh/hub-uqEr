"""工程配置：路径与训练超参，全部带类型注解。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """所有磁盘位置。"""
    model_dir: Path = ROOT / "Qwen2.5-0.5B"
    train_file: Path = ROOT / "sft_train" / "sft_train.jsonl"
    val_file: Path = ROOT / "sft_train" / "sft_val.jsonl"
    out_dir: Path = ROOT / "outputs"

    @property
    def ckpt_dir(self) -> Path:
        """微调后的模型目录。"""
        return self.out_dir / "full"

    @property
    def result_file(self) -> Path:
        return self.out_dir / "train_result.json"


@dataclass(frozen=True)
class TrainConfig:
    """全参 SFT 的超参（0.5B 在 8 GB 显存上可跑）。"""
    max_len: int = 256
    batch_size: int = 4
    grad_accum: int = 4
    grad_ckpt: bool = True
    epochs: float = 3.0
    lr: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class ChatConfig:
    """推理默认参数。"""
    max_new_tokens: int = 64
    temperature: float = 0.0
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4
    max_input_chars: int = 2000


PATH: Paths = Paths()
TRAIN: TrainConfig = TrainConfig()
CHAT: ChatConfig = ChatConfig()
