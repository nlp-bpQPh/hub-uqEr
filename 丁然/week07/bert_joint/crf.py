"""批量条件随机场，支持变长序列和 Viterbi 解码。"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    """线性链 CRF。"""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
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

        # 1. 有效路径位置必须同时满足 attention=1 且已有真实标签。
        mask = mask.bool() & tags.ne(-100)

        # 2. 计算所有可能路径的对数配分。
        log_partition = self._log_partition(emissions, mask)

        # 3. 计算真实标签路径的分数。
        gold_score = self._gold_score(emissions, tags, mask)

        # 4. 负对数似然 = logZ - 真实路径得分。
        return (log_partition - gold_score).mean()

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _, sequence_length, _ = emissions.shape
        # alpha 保存到当前时刻每个结束标签的对数配分。
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
            active = mask[:, time_index].unsqueeze(1)
            score = torch.where(active, next_score, score)
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
        batch_size = emissions.shape[0]
        batch_indices = torch.arange(
            batch_size,
            device=emissions.device,
        )
        lengths = mask.long().sum(dim=1)
        last_indices = lengths - 1
        first_tags = tags[:, 0]
        score = (
            self.start_transitions[first_tags]
            + emissions[batch_indices, 0, first_tags]
        )

        if emissions.shape[1] > 1:
            # padding 标签为 -100，先替换成 0，再通过 mask 排除。
            previous_tags = tags[:, :-1].clamp(min=0)
            next_tags = tags[:, 1:].clamp(min=0)
            # 真实路径得分必须包含每个位置对应标签的发射分数。
            emission_score = emissions[:, 1:, :].gather(
                2,
                next_tags.unsqueeze(-1),
            ).squeeze(-1)
            transition_score = self.transitions[previous_tags, next_tags]
            score = score + (
                (transition_score + emission_score)
                * mask[:, 1:].to(score.dtype)
            ).sum(dim=1)

        last_tags = tags[batch_indices, last_indices]
        return score + self.end_transitions[last_tags]

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """用 Viterbi 求每个样本得分最高的标签序列。"""

        mask = mask.bool()
        batch_size, sequence_length, _ = emissions.shape
        # 路径得分 = 起始转移 + 发射分数 + 标签间转移 + 结束转移。
        score = (
            self.start_transitions.view(1, -1)
            + emissions[:, 0, :]
        )
        history: list[torch.Tensor] = []

        for time_index in range(1, sequence_length):
            # 当前状态只依赖上一时刻的结束状态。
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
            )
            best_score, best_tag = next_score.max(dim=1)
            next_score = (
                best_score + emissions[:, time_index, :]
            )
            active = mask[:, time_index].unsqueeze(1)
            # padding 位置不推进状态，因此保持上一时刻分数。
            score = torch.where(active, next_score, score)
            history.append(best_tag)

        final_score = score + self.end_transitions.view(1, -1)
        best_last_tags = final_score.argmax(dim=-1)
        lengths = mask.long().sum(dim=1).tolist()

        paths: list[list[int]] = []
        for batch_index, length in enumerate(lengths):
            tag = int(best_last_tags[batch_index].item())
            path = [tag]
            # 从最后一个有效标签反向恢复整条最优路径。
            for time_index in range(length - 1, 0, -1):
                tag = int(
                    history[time_index - 1][batch_index, tag].item()
                )
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths

