"""加载 Qwen3 并可选注入 LoRA。"""
from __future__ import annotations

from pathlib import Path

import torch

from llm.lora import apply_lora, lora_parameters


def load_llm(model_dir: str | Path, device, lora: bool = False,
             r: int = 16, alpha: int = 32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # 生成/取最后一个位置必须左填充

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, dtype=torch.float32)
    except TypeError:                  # 旧版 transformers
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype=torch.float32)

    if lora:
        # 先把整个底座冻结，再注入 LoRA —— 否则 embedding / LayerNorm / MLP
        # 仍然是可训练的，实际等于「LoRA + 大半全参微调」
        for p in model.parameters():
            p.requires_grad = False
        n = apply_lora(model, r=r, alpha=alpha)
        params = lora_parameters(model)
    else:
        for p in model.parameters():
            p.requires_grad = True
        n, params = 0, [p for p in model.parameters() if p.requires_grad]

    model = model.to(device)
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] {'LoRA' if lora else '全参'}  替换 {n} 层  "
          f"可训练 {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return tok, model, params
