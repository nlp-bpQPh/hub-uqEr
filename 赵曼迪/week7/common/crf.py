# -*- coding: utf-8 -*-
"""
手写 CRF 修复版，Linear分支完全不受任何改动，仅修复CRF训练逻辑；
valid_tag_mask：bool[B,L] True表示该token拥有真实gold标签；False代表CLS/SEP/重复子词，不参与gold_score。
配分函数logZ仍然跑完整全部token序列，解码逻辑不变。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

NEG_INF = -1e4


class CRF(nn.Module):
    def __init__(self, num_tags: int, constraint: torch.Tensor | None = None,
                 start_constraint: torch.Tensor | None = None,
                 batch_first: bool = True):
        super().__init__()
        assert batch_first, "本项目统一用 batch_first=(B, L, C)"
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        if constraint is None:
            constraint = torch.ones(num_tags, num_tags, dtype=torch.bool)
        self.register_buffer("constraint", constraint.bool(), persistent=True)
        if start_constraint is None:
            start_constraint = torch.ones(num_tags, dtype=torch.bool)
        self.register_buffer("start_constraint", start_constraint.bool(), persistent=True)

    def _effective_transitions(self) -> torch.Tensor:
        if self.constraint.all():
            return self.transitions
        return self.transitions.masked_fill(~self.constraint, NEG_INF)

    def _effective_start(self) -> torch.Tensor:
        if self.start_constraint.all():
            return self.start_transitions
        return self.start_transitions.masked_fill(~self.start_constraint, NEG_INF)

    def _gather_emit(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)

    def _gold_score(self, emissions, tags, mask, valid_tag_mask):
        """
        emissions: [B, L, num_tags]
        tags: [B, L] 包含-100
        mask: attention_mask [B, L]
        valid_tag_mask: [B, L] bool True代表该token是汉字首子词，拥有真实gold标签
        """
        B, L, N = emissions.shape
        gold_score = torch.zeros(B, device=emissions.device, dtype=emissions.dtype)

        for b in range(B):
            for t in range(L):
                if not mask[b, t]:
                    break
                # =========发射分数：只有valid位置才累加发射得分 =========
                if valid_tag_mask[b, t]:
                    tag_id = tags[b, t]
                    gold_score[b] += emissions[b, t, tag_id]

                # =========转移分数：当前t与t‑1都必须是valid，才累加转移 =========
                if t >= 1 and valid_tag_mask[b, t] and valid_tag_mask[b, t-1]:
                    prev_tag = tags[b, t-1]
                    curr_tag = tags[b, t]
                    gold_score[b] += self.transitions[prev_tag, curr_tag]
        return gold_score


    def _forward_algorithm(self, emissions: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        B, L, C = emissions.shape
        trans = self._effective_transitions()
        alpha = self._effective_start().unsqueeze(0) + emissions[:, 0]
        for i in range(1, L):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            m = mask[:, i].unsqueeze(1).bool()
            alpha = torch.where(m, new_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor,
                           mask: torch.Tensor, valid_tag_mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        valid_tag_mask = valid_tag_mask.bool()
        logZ = self._forward_algorithm(emissions, mask)
        gold = self._gold_score(emissions, tags, mask, valid_tag_mask)
        return (logZ - gold).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor,
               return_scores: bool = False):
        mask = mask.bool()
        B, L, C = emissions.shape
        trans = self._effective_transitions()
        alpha = self._effective_start().unsqueeze(0) + emissions[:, 0]
        backpointers = torch.zeros(B, L, C, dtype=torch.long, device=emissions.device)

        for i in range(1, L):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0)
            best_score, best_prev = scores.max(dim=1)
            new_alpha = best_score + emissions[:, i]
            m = mask[:, i].unsqueeze(1).bool()
            alpha = torch.where(m, new_alpha, alpha)
            backpointers[:, i] = best_prev

        final_scores = alpha + self.end_transitions.unsqueeze(0)
        best_last = final_scores.argmax(dim=1)
        lengths = mask.long().sum(dim=1)

        paths = []
        for b in range(B):
            Lb = int(lengths[b].item())
            tag = int(best_last[b].item())
            path = [tag]
            for i in range(Lb - 1, 0, -1):
                tag = int(backpointers[b, i, tag].item())
                path.append(tag)
            paths.append(path[::-1])

        if not return_scores:
            return paths
        scores = []
        for b, path in enumerate(paths):
            seq_len = max(len(path), 1)
            tag_t = torch.tensor(path, dtype=torch.long, device=emissions.device)
            emit = self._gather_emit(emissions[b:b + 1, :seq_len], tag_t.unsqueeze(0)).squeeze(0)
            trans_sum = (trans[tag_t[:-1], tag_t[1:]].sum() if seq_len > 1
                         else torch.zeros((), device=emissions.device))
            path_score = (self._effective_start()[tag_t[0]] + emit.sum()
                          + trans_sum + self.end_transitions[tag_t[-1]])
            sub_mask = mask[b:b + 1, :seq_len]
            logZ = self._forward_algorithm(emissions[b:b + 1, :seq_len], sub_mask)
            conf = float(torch.exp((path_score - logZ[0]) / max(seq_len, 1)).clamp(0.0, 1.0).item())
            scores.append(conf)
        return paths, scores


class CRFClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_tags: int,
                 dropout: float = 0.1, use_crf: bool = True,
                 constraint: torch.Tensor | None = None,
                 start_constraint: torch.Tensor | None = None):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.num_tags = num_tags
        self.use_crf = use_crf
        self.crf = (CRF(num_tags, constraint=constraint, start_constraint=start_constraint)
                    if use_crf else None)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        return self.classifier(self.dropout(hidden))

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, valid_tag_mask=None):
        # self.emissions：encoder + dropout + proj，输出发射分数 (B,L,num_tags)
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            if self.use_crf:
                # 【关键】不再把‑100替换成0！直接传入原始labels，由crf内部依靠valid_tag_mask过滤监督点位
                loss = self.crf.neg_log_likelihood(emit, labels, attention_mask, valid_tag_mask=valid_tag_mask)
            else:
                # --------------------------
                # Linear分支：完全原样保留，没有任何修改，保证linear效果不变
                # --------------------------
                loss = nn.functional.cross_entropy(
                    emit.view(-1, self.num_tags), labels.view(-1), ignore_index=-100)
        return emit, loss

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, token_type_ids=None,
                return_scores: bool = False):
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        if self.use_crf:
            if return_scores:
                return self.crf.decode(emit, attention_mask, return_scores=True)
            return self.crf.decode(emit, attention_mask)
        pred = emit.argmax(dim=-1)
        paths, scores = [], []
        for b in range(pred.shape[0]):
            n = int(attention_mask[b].sum().item())
            paths.append(pred[b, :n].tolist())
            if return_scores:
                probs = torch.softmax(emit[b, :n], dim=-1).max(dim=-1).values
                scores.append(float(probs.mean().item()))
        return (paths, scores) if return_scores else paths


if __name__ == "__main__":
    torch.manual_seed(0)
    C, L = 3, 4
    crf = CRF(num_tags=C)
    emit = torch.randn(2, L, C)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    tags = torch.tensor([[0, 1, 1, 0], [0, 2, 0, 0]])
    valid_tag_mask = torch.ones_like(tags, dtype=torch.bool)
    loss = crf.neg_log_likelihood(emit, tags, mask, valid_tag_mask)
    paths = crf.decode(emit, mask)
    print(f"loss={loss.item():.4f}")
    print(f"解码路径={paths}")
