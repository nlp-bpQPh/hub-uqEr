# -*- coding: utf-8 -*-
"""量一下两种 schema 下 prompt 的 token 长度（决定 max_length，CPU only）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm_zero_shot"))

from event_prompt import build_messages  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained(r"E:\大模型学习\week7_models\Qwen3-0.6B")


def schema_of(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"event_types": {k: v for k, v in raw.items() if not k.startswith("__")},
            "enums": raw.get("__enums__", {})}


def count(msgs):
    text = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return len(TOK(text)["input_ids"])


def main():
    for name, path in [("DuEE-Fin", ROOT / "data/processed/event_schema.json"),
                       ("DuEE1.0", ROOT / "data/duee1/event_schema.json")]:
        sch = schema_of(path)
        base = count(build_messages("测试句子：某公司发布公告。", sch, few_shot=False))
        few = count(build_messages("测试句子：某公司发布公告。", sch, few_shot=True))
        print(f"{name:10s} types={len(sch['event_types']):3d} zero={base:5d} few={few:5d} tokens")


if __name__ == "__main__":
    main()
