"""事件 schema 和 JSONL 读取。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EventSchema:
    """事件类型、允许角色和训练示例读取。"""

    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.event_types = [
            name for name in data if name != "__enums__"
        ]
        self.type_roles = {
            event_type: list(data[event_type])
            for event_type in self.event_types
        }

    def valid_roles(self, event_type: str) -> set[str]:
        return set(self.type_roles.get(event_type, []))

    def prompt_text(self) -> str:
        """把允许的事件类型和角色写入 system prompt。"""

        lines = []
        for event_type in self.event_types:
            roles = "、".join(self.type_roles[event_type])
            lines.append(f"- {event_type}：{roles}")
        return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取训练事件 JSONL。"""

    with Path(path).open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

