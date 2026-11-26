"""加载本地 Qwen Instruct 模型并执行 few-shot 生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from config import get_device


transformers_logging.set_verbosity_error()


class LLMFewShot:
    """不训练参数的大语言模型 in-context 推理器。"""

    def __init__(
        self,
        model_path: str | Path,
        device_name: str = "auto",
        attn_implementation: str = "eager",
    ) -> None:
        path = str(Path(model_path).expanduser().resolve())
        self.device = get_device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = (
            torch.float16
            if self.device.type in {"cuda", "mps"}
            else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            dtype=dtype,
            # Qwen2 使用分组查询注意力，MPS 的 SDPA 路径可能不兼容。
            # eager 会显式扩展 KV head，避免 mps_matmul 维度错误。
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.eval()

        if not self.tokenizer.chat_template:
            raise ValueError("当前 tokenizer 缺少 chat_template")

    def generate(
        self,
        messages: list[dict[str, str]],
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> str:
        """根据对话消息生成一段文本。"""

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {
            name: value.to(self.device)
            for name, value in encoded.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

