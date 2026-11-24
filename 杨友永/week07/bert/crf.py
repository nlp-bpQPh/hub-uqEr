"""线性链 CRF（批量版，带 mask），用于 BERT + Linear + CRF。

为什么手写而不是用 torchcrf：
  - 依赖少，行为可控；
  - 明确展示前向算法（配分函数）和 Viterbi 解码；
  - 支持 batch 内变长（mask）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # ------------------------------------------------------------ 打分
    def _score(self, emissions, tags, mask) -> torch.Tensor:
        """给定标签序列的对数似然分子（未减配分函数）。"""
        batch, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)

        for i in range(1, seq_len):
            m = mask[:, i]
            emit = emissions[:, i].gather(1, tags[:, i : i + 1]).squeeze(1)
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            score = score + (emit + trans) * m
            # 该位置无效时保持分数不变（乘 0 实现跳过）

        lengths = mask.sum(dim=1).long() - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    # ------------------------------------------------------------ 配分函数
    def _partition(self, emissions, mask) -> torch.Tensor:
        """前向算法：log Z。"""
        batch, seq_len, num_tags = emissions.shape
        # alpha: (B, C)
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            m = mask[:, i].unsqueeze(1)  # (B, 1)
            emit = emissions[:, i].unsqueeze(1)  # (B, 1, C)
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit
            new_alpha = torch.logsumexp(scores, dim=1)  # (B, C)
            alpha = torch.where(m.bool(), new_alpha, alpha)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    # ------------------------------------------------------------ 损失
    def nll(self, emissions, tags, mask, reduction: str = "mean") -> torch.Tensor:
        """负对数似然：log Z - score。tags 里的 -100 需由调用方先处理。"""
        log_z = self._partition(emissions, mask)
        score = self._score(emissions, tags, mask)
        loss = log_z - score
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    # ------------------------------------------------------------ 解码
    @torch.no_grad()
    def decode(self, emissions, mask) -> list[list[int]]:
        """Viterbi 解码，返回每条的标签 id 序列（只含有效位置）。"""
        batch, seq_len, num_tags = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []

        for i in range(1, seq_len):
            m = mask[:, i].unsqueeze(1)
            scores = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, i].unsqueeze(1)
            )
            best_score, best_path = scores.max(dim=1)  # (B, C)
            new_score = torch.where(m.bool(), best_score, score)
            history.append(best_path)
            score = new_score

        score = score + self.end_transitions.unsqueeze(0)
        best_last = score.argmax(dim=1)  # (B,)
        lengths = mask.sum(dim=1).long()

        paths: list[list[int]] = []
        for b in range(batch):
            L = int(lengths[b])
            path = [int(best_last[b])]
            for i in range(L - 1, 0, -1):
                path.append(int(history[i - 1][b, path[-1]]))
            paths.append(list(reversed(path)))
        return paths


def pack_sequences(emissions, tags, mask):
    """把带 mask 的变长序列压紧成"有效位置在前"的定长 batch。

    CRF 的 start/end 转移必须作用在**真正的序列边界**上：
    [CLS]/[SEP]/subword 续接/padding 都不能参与打分。
    压紧后每个样本的有效 token 从位置 0 开始连续排列，padding 统一在尾部。
    """
    batch, _, num_tags = emissions.shape
    lens = mask.sum(dim=1).long()
    max_len = int(lens.max().item()) if batch else 0

    packed_em = emissions.new_zeros((batch, max_len, num_tags))
    packed_tags = torch.zeros(
        (batch, max_len), dtype=torch.long, device=emissions.device
    )
    packed_mask = torch.zeros((batch, max_len), dtype=mask.dtype, device=mask.device)

    for b in range(batch):
        idx = mask[b].bool().nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        packed_em[b, :n] = emissions[b, idx]
        if tags is not None:
            packed_tags[b, :n] = tags[b, idx]
        packed_mask[b, :n] = 1

    return packed_em, packed_tags, packed_mask
