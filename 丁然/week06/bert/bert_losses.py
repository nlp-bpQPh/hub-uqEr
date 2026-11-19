"""交叉熵和 Focal Loss。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """降低易分样本权重，重点学习难分样本。"""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        target_log_prob = log_prob.gather(1, labels.unsqueeze(1)).squeeze(1)
        probability = target_log_prob.exp()
        return (-((1 - probability) ** self.gamma) * target_log_prob).mean()


def build_loss(name: str, gamma: float = 2.0) -> nn.Module:
    if name == "focal":
        return FocalLoss(gamma)
    return nn.CrossEntropyLoss()
