# -*- coding: utf-8 -*-
"""
第 4 步：和 SFT 后的模型对话。

用法：
    python src/chat.py                        # 交互式（输入 q 退出，/clear 清空上下文）
    python src/chat.py "给定两个数字，选择较大数。 数字1: 238 数字2: 564"
    python src/chat.py --base                 # 只加载基座模型，看 SFT 之前是什么样
    python src/chat.py --stream               # 流式输出
"""
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load(model_dir, adapter, use_adapter=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device).eval()
    if use_adapter:
        model = PeftModel.from_pretrained(model, adapter).eval()
    return model, tokenizer, device


def build_chat_prompt(messages):
    """messages = [{"role": "user"/"assistant", "content": ...}, ...]"""
    parts = []
    for m in messages:
        parts.append(f"{config.IM_START}{m['role']}\n{m['content'].strip()}{config.IM_END}\n")
    parts.append(f"{config.IM_START}assistant\n")
    return "".join(parts)


@torch.no_grad()
def answer(model, tokenizer, messages, device, max_new_tokens=config.MAX_NEW_TOKENS,
           temperature=0.0, stream=False):
    prompt = build_chat_prompt(messages)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.9 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(config.IM_END),
        repetition_penalty=1.05,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    if stream:
        from threading import Thread
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(gen_kwargs, **enc, streamer=streamer)
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        out = ""
        for piece in streamer:
            print(piece, end="", flush=True)
            out += piece
        thread.join()
        print()
        return out

    gen = model.generate(**gen_kwargs, **enc)
    new_tokens = gen[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="SFT 模型对话")
    parser.add_argument("prompt", nargs="*", help="单轮问题；不传则进入交互模式")
    parser.add_argument("--model", default=str(config.MODEL_DIR))
    parser.add_argument("--adapter", default=str(config.ADAPTER_DIR))
    parser.add_argument("--base", action="store_true", help="不加载 LoRA，只看基座模型")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    args = parser.parse_args()

    use_adapter = not args.base
    if use_adapter and not (Path(args.adapter) / "adapter_config.json").exists():
        raise SystemExit(f"找不到 LoRA 适配器 {args.adapter}，先跑 python src/train_sft.py")

    model, tokenizer, device = load(args.model, args.adapter, use_adapter)
    tag = "基座模型（SFT 前）" if args.base else "SFT 模型（基座 + LoRA）"
    print(f"已加载：{tag} | 设备 {device}")

    if args.prompt:
        q = " ".join(args.prompt)
        print(f"\n用户：{q}\n助手：", end="" if args.stream else "\n")
        out = answer(model, tokenizer, [{"role": "user", "content": q}], device,
                     args.max_new_tokens, args.temperature, args.stream)
        if args.stream:
            pass
        else:
            print(out)
        return

    print("输入问题回车即可，q 退出，/clear 清空多轮上下文。\n")
    history = []
    while True:
        try:
            q = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q == "/clear":
            history.clear()
            print("（上下文已清空）")
            continue

        history.append({"role": "user", "content": q})
        print("助手：", end="" if args.stream else "\n")
        out = answer(model, tokenizer, history, device, args.max_new_tokens,
                     args.temperature, args.stream)
        if not args.stream:
            print(out)
        history.append({"role": "assistant", "content": out})
        print()


if __name__ == "__main__":
    main()
