"""使用本地大语言模型对单条文本做 few-shot 事件抽取。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ICL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ICL_DIR))

from config import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TRAIN_EXTRACT_PATH,
)
from model import LLMFewShot  # noqa: E402
from parser import normalize_output  # noqa: E402
from prompts import build_messages, select_demonstrations  # noqa: E402
from schema import EventSchema, load_jsonl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM few-shot 事件抽取")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--train-extract-path",
        default=str(DEFAULT_TRAIN_EXTRACT_PATH),
    )
    parser.add_argument("--shots-per-type", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
    )
    parser.add_argument("--text", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # 1. 读取 schema 和训练集 few-shot 示例。
    schema = EventSchema(args.schema_path)
    demonstrations = select_demonstrations(
        load_jsonl(args.train_extract_path),
        max(args.shots_per_type, 0),
        args.seed,
    )

    # 2. 加载本地原始 Qwen Instruct 权重，不加载任何微调权重。
    runtime = LLMFewShot(
        args.model_path,
        args.device,
        args.attn_implementation,
    )
    text = args.text or input("请输入新闻文本：")

    # 3. 构造对话 prompt 并生成 JSON。
    messages = build_messages(text, demonstrations, schema)
    raw_output = runtime.generate(
        messages,
        args.max_input_tokens,
        args.max_new_tokens,
    )

    # 4. 校验 JSON，并把模型输出的文字映射回原文位置。
    result = normalize_output(raw_output, text, schema)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

