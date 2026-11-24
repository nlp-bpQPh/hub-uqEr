"""LLM 序列标注模型：Qwen2.5-0.5B-Instruct + LoRA + 线性标注头。

思路：把 decoder-only 模型当"上下文编码器"用，取每个 token 的隐状态接
Linear(num_labels) 做逐 token 分类——与 BERT 方案在同一个任务形式上比较，
只是底座换成了 LLM，并用 LoRA 低成本微调。
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_tokenizer_and_model(model_path: str, num_labels: int, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path),
        num_labels=num_labels,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.TOKEN_CLS,
        modules_to_save=["score"],   # 标注头要训练、保存
        bias="none",
    )
    return get_peft_model(model, cfg)
