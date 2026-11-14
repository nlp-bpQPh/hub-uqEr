# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLM(nn.Module):
    """Pre-LN Decoder-only Transformer，使用因果注意力预测下一个字符。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        max_len: int = 256,
        dropout: float = 0.05,
        pad_id: int = 0,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or n_head <= 0 or n_layer <= 0 or max_len <= 0:
            raise ValueError("vocab_size、d_model、n_head、n_layer、max_len 必须为正数。")
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_head={n_head} 整除。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须处于 [0, 1) 范围内。")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)

        self.tok = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos = nn.Embedding(self.max_len, self.d_model)
        self.drop = nn.Dropout(dropout)

        # TransformerEncoderLayer 提供 MHA + FFN + 残差 + Pre-LN；
        # 配合下三角因果掩码后，每个位置只能访问当前及之前的 Token。
        base_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            base_layer,
            num_layers=self.n_layer,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )

        # PyTorch 会复制 base_layer；逐层重新初始化，避免不同层从完全相同权重起步。
        for block in self.blocks.layers:
            self._init_block(block)

        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok.weight[self.pad_id].zero_()

    @staticmethod
    def _init_block(block: nn.TransformerEncoderLayer) -> None:
        """对每层注意力和前馈网络分别初始化；LayerNorm 缩放设为 1。"""
        for name, parameter in block.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)
            elif "norm" in name and name.endswith("weight"):
                nn.init.ones_(parameter)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """输入 (batch, seq)，返回每个位置对词表的 logits (batch, seq, vocab)。"""
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是二维 Tensor，形状为 (batch, seq_len)。")
        _, seq_len = input_ids.shape
        if seq_len < 1 or seq_len > self.max_len:
            raise ValueError(f"序列长度必须处于 [1, {self.max_len}]，实际为 {seq_len}。")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.drop(self.tok(input_ids) + self.pos(positions))

        # bool mask 中 True 表示禁止注意；上三角屏蔽未来 Token。
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding_mask = input_ids.eq(self.pad_id)
        hidden = self.blocks(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 96,
        eos_id: int = 2,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        forbidden_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """使用贪心解码（默认）或温度采样生成回答 Token。"""
        if not prompt_ids:
            raise ValueError("prompt_ids 不能为空。")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须至少为 1。")
        if temperature < 0:
            raise ValueError("temperature 不能为负数。")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty 必须大于 0。")

        self.eval()
        device = next(self.parameters()).device
        output = [int(token_id) for token_id in prompt_ids]
        # 特殊 Token 不应作为普通答案内容生成；EOS 保留为合法停止符。
        forbidden = set(forbidden_ids or (0, 1, 3, 4, 5))

        for _ in range(max_new_tokens):
            # 对超长上下文只保留最近 max_len 个 Token；短问答数据通常不会触发截断。
            current = output[-self.max_len :]
            input_tensor = torch.tensor([current], dtype=torch.long, device=device)
            logits = self(input_tensor)[0, -1].float().clone()

            for token_id in forbidden:
                if 0 <= token_id < logits.numel():
                    logits[token_id] = float("-inf")

            # 默认 repetition_penalty=1，不惩罚汉字自然重复；仅在用户明确设置时启用。
            if repetition_penalty != 1.0:
                for token_id in set(output[len(prompt_ids) :]):
                    if 0 <= token_id < logits.numel():
                        if logits[token_id] >= 0:
                            logits[token_id] /= repetition_penalty
                        else:
                            logits[token_id] *= repetition_penalty

            if temperature <= 1e-8:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(float(temperature), 1e-5)
                if 0 < top_k < logits.numel():
                    kth_value = torch.topk(logits, top_k).values[-1]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1).item())

            output.append(next_id)
            if next_id == eos_id:
                break

        return output[len(prompt_ids) :]
