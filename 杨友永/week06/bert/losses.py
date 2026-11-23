"""损失与对抗训练：CE / Label Smoothing / Focal Loss / FGM。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL = -(1-p_t)^gamma * log(p_t)，把梯度集中到难样本。"""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        return (-((1 - logp_t.exp()) ** self.gamma) * logp_t).mean()


def build_criterion(kind: str = "ce", label_smoothing: float = 0.0,
                    focal_gamma: float = 2.0) -> nn.Module:
    """kind: ce / ls / focal。"""
    if kind == "focal":
        return FocalLoss(focal_gamma)
    if kind == "ls":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss()


class FGM:
    """Fast Gradient Method：在 embedding 上加沿梯度方向的扰动。

    attack() 之后前向一次拿到对抗梯度，restore() 复原再统一 step。
    """

    def __init__(self, model: nn.Module, eps: float = 1.0,
                 name: str = "word_embeddings") -> None:
        self.model, self.eps, self.name = model, eps, name
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.name in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = p.grad.norm()
                if norm > 0:
                    p.data.add_(self.eps * p.grad / norm)

    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}
