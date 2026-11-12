# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from typing import List, Tuple

import torch

from dataset import CharTokenizer, encode_prompt, load_jsonl
from transformer_lm import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_device(value: str) -> torch.device:
    """解析设备，并在指定 CUDA 不可用时给出明确错误。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    return device


def load_checkpoint(path: str, device: torch.device) -> Tuple[TransformerLM, CharTokenizer]:
    """读取本项目训练生成的 checkpoint，恢复模型结构和词表。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"模型文件不存在：{path}。请先运行 train_sft.py。")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 兼容旧版 PyTorch
        checkpoint = torch.load(path, map_location=device)

    itos = checkpoint["itos"]
    tokenizer = CharTokenizer(itos[6:])
    model = TransformerLM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def prepare_prompt(instruction: str, tokenizer: CharTokenizer, model: TransformerLM) -> List[int]:
    """生成与训练阶段完全相同的纯 SFT Prompt，并保护问题末尾。"""
    prompt_ids = encode_prompt(tokenizer, instruction)
    if len(prompt_ids) > model.max_len:
        # 前两个 ID 是 BOS/USER，最后一个 ID 是 ASSISTANT；保留问题尾部。
        body_budget = model.max_len - 3
        prompt_ids = prompt_ids[:2] + prompt_ids[-(body_budget + 1) :]
    return prompt_ids


def generate_answer(
    question: str,
    model: TransformerLM,
    tokenizer: CharTokenizer,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> str:
    """纯 SFT 问答；默认贪心解码，保证相同 checkpoint 和输入结果可复现。"""
    unknown = tokenizer.count_unknown(question)
    prompt_ids = prepare_prompt(question, tokenizer, model)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_id=2,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    result = tokenizer.decode(generated_ids).strip()
    if unknown:
        print(f"警告：问题中有 {unknown} 个字符不在训练词表中，已映射为 UNK。")
    return result or "模型没有生成有效答案。请检查训练过程、词表和生成长度。"


def normalize_text(text: str) -> str:
    """用于严格文本匹配的宽松规范化：小写、去空白和标点，不改变数字字符。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def char_f1(prediction: str, reference: str) -> float:
    """计算字符多重集 F1，作为表面重合度指标，不等价于语义正确率。"""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return float(pred == ref)
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_generation(
    model: TransformerLM,
    tokenizer: CharTokenizer,
    data_path: str,
    output_path: str,
    max_new_tokens: int,
    limit: int = 0,
) -> None:
    """对数据集逐条真实生成，报告 Exact Match 与字符 F1，并保存预测记录。"""
    rows = load_jsonl(data_path)
    if limit > 0:
        rows = rows[:limit]

    exact = 0
    normalized_exact = 0
    f1_total = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            prediction = generate_answer(
                row["instruction"],
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            reference = row["output"].strip()
            exact += int(prediction.strip() == reference)
            normalized_exact += int(normalize_text(prediction) == normalize_text(reference))
            f1_total += char_f1(prediction, reference)
            file.write(
                json.dumps(
                    {
                        "instruction": row["instruction"],
                        "reference": reference,
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % 50 == 0 or index == len(rows):
                print(f"生成评估进度：{index}/{len(rows)}")

    total = max(len(rows), 1)
    print("\n生成评估结果（Greedy Decoding）")
    print(f"样本数：{len(rows)}")
    print(f"Exact Match：{exact / total:.4f} ({exact}/{len(rows)})")
    print(f"Normalized Exact Match：{normalized_exact / total:.4f} ({normalized_exact}/{len(rows)})")
    print(f"平均字符 F1：{f1_total / total:.4f}")
    print("注意：字符 F1 只衡量文本重合，不代表答案事实正确；建议抽样人工检查。")
    print(f"逐条预测已保存：{output_path}")


def build_parser() -> argparse.ArgumentParser:
    """构建纯 SFT 命令行参数；不提供 RAG/Hybrid 选项。"""
    parser = argparse.ArgumentParser(description="纯 Transformer SFT 推理与生成评估。")
    parser.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoints", "best.pt"))
    parser.add_argument("--question", default=None, help="单条问题；不提供则进入交互模式。")
    parser.add_argument("--eval-data", default=None, help="可选：对 JSONL 执行逐条生成评估。")
    parser.add_argument("--eval-limit", type=int, default=0, help="最多评估多少条；0 表示全部。")
    parser.add_argument("--prediction-file", default=os.path.join(HERE, "logs", "generation_predictions.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 使用贪心解码；大于 0 时随机采样。准确率测试建议保持 0。")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须至少为 1。")
    if args.eval_limit < 0:
        raise ValueError("eval-limit 不能为负数。")
    if args.question and args.eval_data:
        raise ValueError("--question 和 --eval-data 不能同时使用。")

    device = resolve_device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    print(f"已加载模型：{args.checkpoint}")
    print(f"设备：{device}；词表大小：{tokenizer.vocab_size}；最大上下文：{model.max_len}")

    if args.eval_data:
        evaluate_generation(
            model,
            tokenizer,
            args.eval_data,
            args.prediction_file,
            args.max_new_tokens,
            args.eval_limit,
        )
        return

    if args.question:
        print("\n助手：" + generate_answer(
            args.question, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ))
        return

    print("进入纯 SFT 交互模式。输入 exit / quit / q 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not question:
            continue
        answer = generate_answer(
            question, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print("\n助手：" + answer)


if __name__ == "__main__":
    main()
