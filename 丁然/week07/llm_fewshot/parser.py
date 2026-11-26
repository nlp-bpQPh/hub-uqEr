"""解析大语言模型生成结果并映射回字符位置。"""

from __future__ import annotations

import json
from typing import Any

from schema import EventSchema


ROLE_ALIASES = {
    "股份回购": {
        "回购数量": "回购股份数量",
        "回购金额": "交易金额",
        "回购价格": "每股交易价格",
    },
}


def merge_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON 存在重复键时保留首个值，并合并重复 args。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key not in result:
            result[key] = value
        elif key == "args" and isinstance(result[key], list) and isinstance(value, list):
            result[key].extend(value)
    return result


def repair_bracket_structure(text: str) -> str:
    """补齐模型遗漏的 ] 或 }，不修改已有内容。"""

    pairs = {"{": "}", "[": "]"}
    reverse_pairs = {value: key for key, value in pairs.items()}
    stack: list[str] = []
    result: list[str] = []
    in_string = False
    escaped = False

    for character in text:
        if in_string:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            result.append(character)
            continue

        if character in pairs:
            stack.append(character)
            result.append(character)
            continue

        if character in reverse_pairs:
            # 当前闭合符与栈顶不匹配时，先补齐前面遗漏的闭合符。
            while stack and stack[-1] != reverse_pairs[character]:
                result.append(pairs[stack.pop()])
            if stack:
                stack.pop()
            result.append(character)
            continue

        result.append(character)

    while stack:
        result.append(pairs[stack.pop()])
    return "".join(result)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """从模型输出中截取第一个完整 JSON 对象。"""

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型输出中没有 JSON 对象")
    json_text = raw_text[start:end + 1]
    try:
        return json.loads(
            json_text,
            object_pairs_hook=merge_duplicate_pairs,
        )
    except json.JSONDecodeError:
        # 只修复括号层级，不猜测缺失字符串或逗号。
        return json.loads(
            repair_bracket_structure(json_text),
            object_pairs_hook=merge_duplicate_pairs,
        )


def locate_mention(
    text: str,
    mention: str,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """按文字查找原位置，并优先使用尚未占用的区间。"""

    if not mention:
        return None
    cursor = 0
    candidates: list[tuple[int, int]] = []
    while True:
        start = text.find(mention, cursor)
        if start < 0:
            break
        candidates.append((start, start + len(mention)))
        cursor = start + 1
    if not candidates:
        return None
    for span in candidates:
        if span not in used:
            used.add(span)
            return span
    return candidates[0]


def normalize_output(
    raw_text: str,
    source_text: str,
    schema: EventSchema,
) -> dict[str, Any]:
    """校验模型 JSON，并把文字转换成 start/end。"""

    try:
        data = extract_json_object(raw_text)
    except Exception as error:
        return {
            "text": source_text,
            "events": [],
            "raw_output": raw_text,
            "error": f"JSON解析失败: {error}",
        }

    events = []
    for item in data.get("events", []):
        event_type = item.get("type")
        if event_type not in schema.event_types:
            continue

        used: set[tuple[int, int]] = set()
        trigger_span = locate_mention(
            source_text,
            str(item.get("trigger", "")),
            used,
        )
        trigger = None
        if trigger_span is not None:
            start, end = trigger_span
            trigger = {
                "text": source_text[start:end],
                "start": start,
                "end": end,
            }

        args = []
        valid_roles = schema.valid_roles(event_type)
        for argument in item.get("args", []):
            role = argument.get("role")
            mention = str(argument.get("text", ""))
            role = ROLE_ALIASES.get(event_type, {}).get(role, role)
            if role not in valid_roles:
                continue
            span = locate_mention(source_text, mention, used)
            if span is None:
                continue
            start, end = span
            args.append(
                {
                    "role": role,
                    "text": source_text[start:end],
                    "start": start,
                    "end": end,
                }
            )
        events.append(
            {
                "event_type": event_type,
                "trigger": trigger,
                "args": args,
            }
        )
    return {
        "text": source_text,
        "events": events,
        "raw_output": raw_text,
    }

