"""最小 LoRA 实现（不依赖 peft）：把低秩增量挂在指定线性层旁边。

    y = W x + (alpha / r) * B(A(x))

A 随机初始化、B 置零 —— 训练开始时增量恒为 0，不破坏预训练权重。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_dim, out_dim = base.in_features, base.out_features
        self.r, self.scale = r, alpha / r
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.scale


def apply_lora(model: nn.Module, r: int = 16, alpha: int = 32,
               targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
               ) -> int:
    """替换所有名字以 targets 结尾的 Linear，返回替换数量。"""
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(module, r, alpha))
        replaced += 1
    return replaced


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if p.requires_grad]
