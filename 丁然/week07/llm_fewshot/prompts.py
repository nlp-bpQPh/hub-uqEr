"""构造 few-shot 对话示例。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

from schema import EventSchema


def compact_event(row: dict[str, Any]) -> dict[str, Any]:
    """保留模型需要生成的类型、文字和角色，不传位置。"""

    event = row["event"]
    return {
        "type": event["type"],
        "trigger": event["trigger"]["text"],
        "args": [
            {
                "role": argument["role"],
                "text": argument["text"],
            }
            for argument in event["args"]
        ],
    }


def select_demonstrations(
    rows: list[dict[str, Any]],
    shots_per_type: int,
    seed: int,
) -> list[dict[str, Any]]:
    """每种事件选取相同数量的 few-shot 示例。"""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["event"]["type"]].append(row)

    rng = random.Random(seed)
    demonstrations: list[dict[str, Any]] = []
    for event_type in sorted(buckets):
        bucket = list(buckets[event_type])
        rng.shuffle(bucket)
        # 优先使用较短示例，降低 0.5B 模型的 prompt 长度和生成负担。
        bucket.sort(
            key=lambda row: (
                len(row["text"]),
                len(row["event"]["args"]),
            )
        )
        demonstrations.extend(bucket[:shots_per_type])
    rng.shuffle(demonstrations)
    return demonstrations


def build_messages(
    text: str,
    demonstrations: list[dict[str, Any]],
    schema: EventSchema,
) -> list[dict[str, str]]:
    """构造 system、few-shot 示例和当前待抽取文本。"""

    system = (
        "你是中文财经新闻事件抽取器。只输出一个合法 JSON 对象，"
        "不要输出 Markdown、解释或额外文字。\n"
        "输出格式：\n"
        '{"events":[{"type":"事件类型","trigger":"触发词",'
        '"args":[{"role":"论元角色","text":"论元文字"}]}]}\n'
        "没有事件时输出 {\"events\":[]}。\n"
        "每个输入最多输出一个事件。每个角色最多输出一次。"
        "不要输出原文中不存在的文字，不要猜测论元。\n"
        "只能使用下面的事件类型和对应角色：\n"
        f"{schema.prompt_text()}"
    )
    messages = [{"role": "system", "content": system}]
    for row in demonstrations:
        messages.append(
            {
                "role": "user",
                "content": f"新闻：{row['text']}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"events": [compact_event(row)]},
                    ensure_ascii=False,
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"新闻：{text}",
        }
    )
    return messages

