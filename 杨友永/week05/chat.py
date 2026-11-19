"""问答：加载微调后的模型，接收 ChatRequest，返回 ChatResponse。"""
from __future__ import annotations

import time
from typing import Any

# torch < 2.4 的 MPS 后端缺 aten::isin（generate 会用到），
# 打开回退让个别算子走 CPU，其余仍在 GPU 上跑。
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*aten::isin.*")

import torch

from config import CHAT, PATH, ChatConfig
from device import describe, pick_device, pick_dtype
from schemas import ChatRequest, ChatResponse


def stop_token_ids(tok: Any) -> list[int]:
    """chat template 的 <|im_end|> 不等于 eos(<|endoftext|>)，两个都要停。"""
    ids = [tok.eos_token_id]
    for token in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    return ids


class ChatEngine:
    """封装 tokenizer + model，对外只暴露一个 ask()。"""

    def __init__(self, tok: Any, model: Any, device: torch.device, name: str,
                 cfg: ChatConfig = CHAT) -> None:
        self.tokenizer = tok
        self.model = model
        self.device = device
        self.name = name
        self.cfg = cfg

    @classmethod
    def load(cls, device: str | None = None, cfg: ChatConfig = CHAT) -> "ChatEngine":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 模型用微调后的，tokenizer 始终用底座的（SFT 不改 tokenizer，
        # 混用不同版本的 tokenizer.json 会报 untagged enum ModelWrapper 错误）
        path = PATH.ckpt_dir if PATH.ckpt_dir.exists() else PATH.model_dir
        dev = pick_device(device)
        tok = AutoTokenizer.from_pretrained(str(PATH.model_dir), local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = pick_dtype(dev)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), local_files_only=True, dtype=dtype)
        except TypeError:                   # transformers < 5
            model = AutoModelForCausalLM.from_pretrained(
                str(path), local_files_only=True, torch_dtype=dtype)
        model = model.to(dev).eval()
        return cls(tok, model, dev, name=path.name, cfg=cfg)

    @torch.no_grad()
    def ask(self, request: ChatRequest) -> ChatResponse:
        """一次问答：拼模板 → 生成 → 切掉特殊 token。"""
        question = request.question.strip()[: self.cfg.max_input_chars]
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=False).to(self.device)

        kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": stop_token_ids(self.tokenizer),
            "repetition_penalty": self.cfg.repetition_penalty,
            "no_repeat_ngram_size": self.cfg.no_repeat_ngram_size,
        }
        if request.temperature > 0:            # 0 = 贪心解码
            kwargs.update(do_sample=True, temperature=request.temperature)
        else:
            kwargs.update(do_sample=False)

        started = time.perf_counter()
        output = self.model.generate(**inputs, **kwargs)
        latency = (time.perf_counter() - started) * 1000

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return ChatResponse(
            answer=text.split("<|im_end|>")[0].strip(),
            latency_ms=round(latency, 1),
            model=self.name,
        )


def main() -> None:
    """命令行交互。"""
    engine = ChatEngine.load()
    print(f"模型 {engine.name} · 设备 {describe(engine.device)}")
    print("输入问题，回车退出。")
    while True:
        question = input("\n问题> ").strip()
        if not question:
            break
        response = engine.ask(ChatRequest(question=question))
        print(f"回答> {response.answer}")


if __name__ == "__main__":
    main()
