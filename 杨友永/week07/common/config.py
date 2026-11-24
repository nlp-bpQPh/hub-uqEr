"""全局配置：路径解析、标签集、三组实验的超参数。

设计原则：
  1. 换机器只改环境变量或保证 pretrain_models/ 跟着走；
  2. 云主机上是复用上一个工程（text_class）里的模型，所以候选路径里包含它；
  3. 所有实验配置集中在这里，train 脚本只读不改。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("NER_DATA_DIR", ROOT / "data" / "peoples_daily"))
OUTPUT_DIR = Path(os.getenv("NER_OUTPUT_DIR", ROOT / "outputs"))


def _first_dir(candidates: list[str | Path | None]) -> Path:
    """返回第一个存在的目录；都不存在时返回最后一个候选（方便报错定位）。"""
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c).resolve()
    return Path(candidates[-1]).expanduser()


def resolve_bert_path() -> Path:
    return _first_dir([
        os.getenv("BERT_PATH"),
        ROOT / "pretrain_models" / "bert-base-chinese",
        Path("/root/text_class/pretrain_models/bert-base-chinese"),
        Path("/cloud/cloud-ssd1/text_class/pretrain_models/bert-base-chinese"),
        Path.home() / "Desktop/AILearn/Python/DeepLearning/homework/week7_text_tag/pretrain_models/bert-base-chinese",
    ])


def resolve_llm_path() -> Path:
    return _first_dir([
        os.getenv("LLM_PATH"),
        ROOT / "pretrain_models" / "Qwen2.5-0.5B-Instruct",
        Path("/root/text_class/pretrain_models/Qwen2.5-0.5B-Instruct"),
        Path("/cloud/cloud-ssd1/text_class/pretrain_models/Qwen2.5-0.5B-Instruct"),
        Path.home() / "Desktop/AILearn/Python/DeepLearning/homework/week7_text_tag/pretrain_models/Qwen2.5-0.5B-Instruct",
    ])


BERT_PATH = resolve_bert_path()
LLM_PATH = resolve_llm_path()


@dataclass
class DataConfig:
    data_dir: Path = DATA_DIR
    train_file: str = "train.json"
    val_file: str = "validation.json"
    test_file: str = "test.json"
    label_file: str = "label_names.json"
    max_len: int = 128
    num_workers: int = 0 if os.name == "nt" else 2

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def val_path(self) -> Path:
        return self.data_dir / self.val_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file

    @property
    def label_path(self) -> Path:
        return self.data_dir / self.label_file


@dataclass
class BertConfig:
    model_path: Path = field(default_factory=lambda: BERT_PATH)
    max_len: int = 128
    batch_size: int = 32
    epochs: int = 3
    lr: float = 3e-5          # BERT 主干
    head_lr_mult: float = 5.0  # 分类头 / CRF 用更大的学习率
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class LLMConfig:
    model_path: Path = field(default_factory=lambda: LLM_PATH)
    max_len: int = 128
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42


def load_label_names() -> list[str]:
    with open(DataConfig().label_path, encoding="utf-8") as f:
        return json.load(f)


def env_summary() -> str:
    import platform
    import sys
    return (f"platform={platform.system()} python={sys.version.split()[0]} "
            f"cwd={Path.cwd()} root={ROOT}")


def check_assets() -> list[str]:
    """训练前检查数据与模型，返回问题列表。"""
    problems: list[str] = []
    d = DataConfig()
    for name, p in [("训练集", d.train_path), ("验证集", d.val_path),
                    ("测试集", d.test_path), ("标签表", d.label_path)]:
        if not p.exists():
            problems.append(f"缺少{name}: {p}")
    for name, p in [("BERT", BERT_PATH), ("LLM", LLM_PATH)]:
        if not p.is_dir():
            problems.append(f"缺少{name}模型目录: {p}")
        else:
            has = any((p / f).exists() for f in ("pytorch_model.bin", "model.safetensors"))
            if not has:
                problems.append(f"{name}目录里没有权重文件: {p}")
    return problems
