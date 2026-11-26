"""第二阶段使用的批量线性链 CRF。"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    """支持变长序列的 CRF 损失和 Viterbi 解码。"""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回平均负对数似然。"""

        mask = mask.bool() & tags.ne(-100)
        log_partition = self._log_partition(emissions, mask)
        gold_score = self._gold_score(emissions, tags, mask)
        return (log_partition - gold_score).mean()

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """动态规划计算所有标签路径的对数配分。"""

        _, sequence_length, _ = emissions.shape
        score = (
            self.start_transitions.view(1, -1)
            + emissions[:, 0, :]
        )
        for time_index in range(1, sequence_length):
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, time_index, :].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(
                mask[:, time_index].unsqueeze(1),
                next_score,
                score,
            )
        return torch.logsumexp(
            score + self.end_transitions.view(1, -1),
            dim=1,
        )

    def _gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算真实标签路径得分。"""

        batch_size = emissions.shape[0]
        batch_indices = torch.arange(
            batch_size,
            device=emissions.device,
        )
        lengths = mask.long().sum(dim=1)
        first_tags = tags[:, 0]
        score = (
            self.start_transitions[first_tags]
            + emissions[batch_indices, 0, first_tags]
        )

        if emissions.shape[1] > 1:
            previous_tags = tags[:, :-1].clamp(min=0)
            next_tags = tags[:, 1:].clamp(min=0)
            emission_score = emissions[:, 1:, :].gather(
                2,
                next_tags.unsqueeze(-1),
            ).squeeze(-1)
            transition_score = self.transitions[previous_tags, next_tags]
            score = score + (
                (transition_score + emission_score)
                * mask[:, 1:].to(score.dtype)
            ).sum(dim=1)

        last_indices = lengths - 1
        last_tags = tags[batch_indices, last_indices]
        return score + self.end_transitions[last_tags]

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """使用 Viterbi 算法恢复最优标签路径。"""

        mask = mask.bool()
        _, sequence_length, _ = emissions.shape
        score = (
            self.start_transitions.view(1, -1)
            + emissions[:, 0, :]
        )
        history: list[torch.Tensor] = []

        for time_index in range(1, sequence_length):
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
            )
            best_score, best_tag = next_score.max(dim=1)
            next_score = best_score + emissions[:, time_index, :]
            score = torch.where(
                mask[:, time_index].unsqueeze(1),
                next_score,
                score,
            )
            history.append(best_tag)

        final_score = score + self.end_transitions.view(1, -1)
        best_last_tags = final_score.argmax(dim=-1)
        lengths = mask.long().sum(dim=1).tolist()

        paths: list[list[int]] = []
        for batch_index, length in enumerate(lengths):
            tag = int(best_last_tags[batch_index].item())
            path = [tag]
            for time_index in range(length - 1, 0, -1):
                tag = int(
                    history[time_index - 1][batch_index, tag].item()
                )
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths

