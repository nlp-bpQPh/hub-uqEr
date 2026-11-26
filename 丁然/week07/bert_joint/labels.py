"""事件类型和 BIO 标签定义。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LabelSpace:
    """把事件类型写进触发词标签，统一为 149 类序列标签。"""

    def __init__(self, schema_path: str | Path) -> None:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            raise ValueError("事件 schema 必须是 JSON 对象")

        self.event_types = [
            name for name in schema if name != "__enums__"
        ]
        self.type_roles = {
            event_type: list(schema[event_type])
            for event_type in self.event_types
        }

        # 同名角色在多个事件中只保留一个标签，减少类别数量。
        self.roles: list[str] = []
        seen_roles: set[str] = set()
        for event_type in self.event_types:
            for role in self.type_roles[event_type]:
                if role not in seen_roles:
                    seen_roles.add(role)
                    self.roles.append(role)

        # 标签 0 是 O，之后是 13 种事件类型对应的触发词 B/I。
        self.outside_id = 0
        self.event_trigger_tag_ids: dict[str, tuple[int, int]] = {}
        self.trigger_tag_to_event: dict[int, str] = {}
        self.trigger_tag_is_begin: dict[int, bool] = {}
        next_id = 1
        for event_type in self.event_types:
            begin_id = next_id
            inside_id = next_id + 1
            next_id += 2
            self.event_trigger_tag_ids[event_type] = (
                begin_id,
                inside_id,
            )
            self.trigger_tag_to_event[begin_id] = event_type
            self.trigger_tag_to_event[inside_id] = event_type
            self.trigger_tag_is_begin[begin_id] = True
            self.trigger_tag_is_begin[inside_id] = False

        self.role_to_id = {
            role: index for index, role in enumerate(self.roles)
        }
        self.role_tag_ids: dict[str, tuple[int, int]] = {}
        self.tag_to_role: dict[int, str] = {}
        self.tag_is_begin: dict[int, bool] = {}

        # 后续标签是 61 个论元角色各自的 B/I。
        for role in self.roles:
            begin_id = next_id
            inside_id = next_id + 1
            next_id += 2
            self.role_tag_ids[role] = (begin_id, inside_id)
            self.tag_to_role[begin_id] = role
            self.tag_to_role[inside_id] = role
            self.tag_is_begin[begin_id] = True
            self.tag_is_begin[inside_id] = False

        self.num_labels = next_id
        self.label_names = self._build_label_names()
        self.text_label_ids = list(range(self.num_labels))
        self._validate_counts()

    def _build_label_names(self) -> list[str]:
        names = ["O"]
        for event_type in self.event_types:
            names.extend(
                [
                    f"B-{event_type}-触发词",
                    f"I-{event_type}-触发词",
                ]
            )
        for role in self.roles:
            names.extend([f"B-{role}", f"I-{role}"])
        return names

    def _validate_counts(self) -> None:
        if len(self.event_types) != 13:
            raise ValueError(f"预期 13 类事件，实际为 {len(self.event_types)}")
        if len(self.roles) != 61:
            raise ValueError(f"预期 61 个论元角色，实际为 {len(self.roles)}")
        if self.num_labels != 149:
            raise ValueError(f"预期 149 个标签，实际为 {self.num_labels}")

    def valid_roles(self, event_type: str) -> set[str]:
        return set(self.type_roles.get(event_type, []))

    def to_dict(self) -> dict[str, Any]:
        """保存重建标签空间所需的字段。"""

        return {
            "format_version": 2,
            "event_types": self.event_types,
            "roles": self.roles,
            "type_roles": self.type_roles,
            "num_labels": self.num_labels,
            "label_names": self.label_names,
            "outside_id": self.outside_id,
            "event_trigger_tag_ids": self.event_trigger_tag_ids,
            "role_tag_ids": self.role_tag_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelSpace":
        """从保存的 JSON 恢复标签空间，不依赖原始 schema 文件。"""

        if data.get("format_version") != 2:
            raise ValueError("旧版 138 标签模型不兼容，需要重新训练")
        instance = cls.__new__(cls)
        instance.event_types = list(data["event_types"])
        instance.roles = list(data["roles"])
        instance.type_roles = {
            event_type: list(roles)
            for event_type, roles in data["type_roles"].items()
        }
        instance.outside_id = int(data["outside_id"])
        instance.event_trigger_tag_ids = {
            event_type: tuple(tag_ids)
            for event_type, tag_ids in data[
                "event_trigger_tag_ids"
            ].items()
        }
        instance.trigger_tag_to_event = {}
        instance.trigger_tag_is_begin = {}
        for event_type, (begin_id, inside_id) in (
            instance.event_trigger_tag_ids.items()
        ):
            instance.trigger_tag_to_event[begin_id] = event_type
            instance.trigger_tag_to_event[inside_id] = event_type
            instance.trigger_tag_is_begin[begin_id] = True
            instance.trigger_tag_is_begin[inside_id] = False
        instance.role_to_id = {
            role: index for index, role in enumerate(instance.roles)
        }
        instance.role_tag_ids = {
            role: tuple(tag_ids)
            for role, tag_ids in data["role_tag_ids"].items()
        }
        instance.tag_to_role = {}
        instance.tag_is_begin = {}
        for role, (begin_id, inside_id) in instance.role_tag_ids.items():
            instance.tag_to_role[begin_id] = role
            instance.tag_to_role[inside_id] = role
            instance.tag_is_begin[begin_id] = True
            instance.tag_is_begin[inside_id] = False
        instance.num_labels = int(data["num_labels"])
        instance.label_names = list(data["label_names"])
        instance.text_label_ids = list(range(instance.num_labels))
        return instance

