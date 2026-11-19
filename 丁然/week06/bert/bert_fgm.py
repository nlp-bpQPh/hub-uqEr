"""FGM 对抗训练。"""

import torch


class FGM:
    """沿词嵌入梯度加入一次小扰动。"""

    def __init__(self, model: torch.nn.Module, epsilon: float = 0.5) -> None:
        self.epsilon = epsilon
        self.embedding_weight = model.get_input_embeddings().weight
        self.backup: torch.Tensor | None = None

    def attack(self) -> None:
        gradient = self.embedding_weight.grad
        if gradient is None:
            return
        norm = torch.norm(gradient)
        if not torch.isfinite(norm) or norm.item() == 0:
            return
        self.backup = self.embedding_weight.detach().clone()
        self.embedding_weight.data.add_(
            self.epsilon * gradient.detach() / (norm + 1e-8)
        )

    def restore(self) -> None:
        if self.backup is not None:
            self.embedding_weight.data.copy_(self.backup)
            self.backup = None
