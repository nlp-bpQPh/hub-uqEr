# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID, USER_ID, ASSISTANT_ID = range(len(SPECIAL_TOKENS))
IGNORE_INDEX = -100

# 纯 SFT 使用唯一模板；训练和推理必须调用同一个 encode_prompt()。
PROMPT_PREFIX = "用户问题：\n"
PROMPT_SUFFIX = "\n助手回答：\n"


def load_jsonl(path: str) -> List[dict]:
    """读取 JSONL；校验每条记录包含非空 instruction 和 output。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据文件不存在：{path}")

    rows: List[dict] = []
    with open(path, "r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 格式错误：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_no} 行必须是 JSON object。")
            instruction = row.get("instruction")
            output = row.get("output")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path} 第 {line_no} 行 instruction 必须是非空字符串。")
            if not isinstance(output, str) or not output.strip():
                raise ValueError(f"{path} 第 {line_no} 行 output 必须是非空字符串。")
            rows.append({"instruction": instruction.strip(), "output": output.strip()})

    if not rows:
        raise ValueError(f"数据文件没有有效样本：{path}")
    return rows


class CharTokenizer:
    """轻量字符级 Tokenizer；将训练集出现的 Unicode 字符映射为 ID。"""

    def __init__(self, characters: Sequence[str]):
        unique: List[str] = []
        seen = set(SPECIAL_TOKENS)
        for char in characters:
            if char not in seen:
                unique.append(char)
                seen.add(char)
        self.itos = list(SPECIAL_TOKENS) + unique
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def from_records(cls, rows: Sequence[dict]) -> "CharTokenizer":
        """仅从训练集的问题和答案构建词表，避免验证集信息泄漏。"""
        counter = Counter(PROMPT_PREFIX + PROMPT_SUFFIX)
        for row in rows:
            counter.update(row["instruction"])
            counter.update(row["output"])
        # 排序后构建词表，保证相同数据和特殊 Token 顺序对应相同 ID。
        return cls(sorted(counter.keys()))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        """按字符编码；训练词表之外的字符映射为 UNK。"""
        return [self.stoi.get(char, UNK_ID) for char in text]

    def decode(self, token_ids: Iterable[int], skip_special: bool = True) -> str:
        """将 Token ID 还原为文本；默认跳过所有特殊 Token。"""
        output: List[str] = []
        for value in token_ids:
            index = int(value)
            if index < 0 or index >= len(self.itos):
                continue
            token = self.itos[index]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            output.append(token)
        return "".join(output)

    def count_unknown(self, text: str) -> int:
        """统计输入中的 OOV 字符数量，推理时用于提醒词表覆盖不足。"""
        return sum(char not in self.stoi for char in text)


def build_prompt(instruction: str) -> str:
    """纯 SFT Prompt；不包含材料检索、拒答或答案本身。"""
    return PROMPT_PREFIX + instruction.strip() + PROMPT_SUFFIX


def encode_prompt(tokenizer: CharTokenizer, instruction: str) -> List[int]:
    """构造 BOS + USER + Prompt + ASSISTANT，供训练与推理共用。"""
    return [BOS_ID, USER_ID] + tokenizer.encode(build_prompt(instruction)) + [ASSISTANT_ID]


@dataclass
class EncodedExample:
    """一条经过 next-token shift 的模型输入和监督标签。"""
    input_ids: List[int]
    labels: List[int]


class SFTDataset(Dataset):
    """构造纯 SFT 样本；仅 Assistant 的目标回答和 EOS 参与 Loss。"""

    def __init__(
        self,
        rows: Sequence[dict],
        tokenizer: CharTokenizer,
        max_seq_len: int = 256,
        max_target_chars: int = 128,
    ) -> None:
        if max_seq_len < 32:
            raise ValueError("max_seq_len 至少为 32。")
        if max_target_chars < 1:
            raise ValueError("max_target_chars 必须大于 0。")

        self.examples: List[EncodedExample] = []
        self.truncated_prompts = 0
        self.truncated_targets = 0
        self.unknown_instruction_chars = 0
        self.unknown_target_chars = 0

        for row_index, row in enumerate(rows):
            instruction = row["instruction"].strip()
            answer = row["output"].strip()
            self.unknown_instruction_chars += tokenizer.count_unknown(instruction)
            self.unknown_target_chars += tokenizer.count_unknown(answer)

            if len(answer) > max_target_chars:
                answer = answer[:max_target_chars]
                self.truncated_targets += 1

            target_ids = tokenizer.encode(answer) + [EOS_ID]
            # 总序列 full_ids 长度最多为 max_seq_len + 1，因为 input_ids 会右移一位。
            if len(target_ids) > max_seq_len:
                target_ids = target_ids[: max_seq_len - 1] + [EOS_ID]
                self.truncated_targets += 1

            prompt_ids = encode_prompt(tokenizer, instruction)
            prompt_budget = max_seq_len + 1 - len(target_ids)
            if prompt_budget < 4:
                raise ValueError(f"第 {row_index + 1} 条样本过长，无法保留完整回答。")

            if len(prompt_ids) > prompt_budget:
                self.truncated_prompts += 1
                # 保留 BOS、USER、问题末尾和 ASSISTANT，避免截掉问题末尾的重要参数。
                keep_body = prompt_budget - 3
                prompt_ids = prompt_ids[:2] + prompt_ids[-(keep_body + 1) :]

            full_ids = prompt_ids + target_ids
            input_ids = full_ids[:-1]
            # input 的最后一个 Prompt Token（ASSISTANT）预测 answer 的第一个 Token。
            # 因此前 prompt_len-1 个标签全部为 -100，answer 与 EOS 从位置 prompt_len-1 开始。
            labels = [IGNORE_INDEX] * (len(prompt_ids) - 1) + target_ids

            if len(input_ids) != len(labels) or len(input_ids) > max_seq_len:
                raise RuntimeError("输入与标签长度不一致，请检查 next-token shift。")
            self.examples.append(EncodedExample(input_ids, labels))

        if not self.examples:
            raise ValueError("没有可用的 SFT 样本。")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.examples[index]
        return {
            "input_ids": torch.tensor(item.input_ids, dtype=torch.long),
            "labels": torch.tensor(item.labels, dtype=torch.long),
        }


def collate_sft_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """将变长样本右侧 Padding；Padding 位置的 label 一律为 -100。"""
    max_length = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_length), IGNORE_INDEX, dtype=torch.long)
    for row_index, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row_index, :length] = item["input_ids"]
        labels[row_index, :length] = item["labels"]
    return {"input_ids": input_ids, "labels": labels}


def dataset_summary(rows: Sequence[dict], tokenizer: CharTokenizer) -> dict:
    """生成基础数据质量统计；不自动修改或纠正标签。"""
    instructions = [row["instruction"].strip() for row in rows]
    outputs = [row["output"].strip() for row in rows]
    total_chars = sum(len(text) for text in instructions + outputs)
    unknown = sum(tokenizer.count_unknown(text) for text in instructions + outputs)
    return {
        "samples": len(rows),
        "duplicate_instructions": len(instructions) - len(set(instructions)),
        "avg_instruction_chars": sum(map(len, instructions)) / len(rows),
        "avg_output_chars": sum(map(len, outputs)) / len(rows),
        "max_output_chars": max(map(len, outputs)),
        "unknown_chars": unknown,
        "unknown_rate": unknown / max(total_chars, 1),
    }
