"""两阶段流水线共享的数据切分和指标计算。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


class EventSchema:
    """保存事件类型及其允许的论元角色。"""

    def __init__(self, schema_path: str | Path) -> None:
        schema = json.loads(
            Path(schema_path).read_text(encoding="utf-8")
        )
        self.event_types = [
            name for name in schema if name != "__enums__"
        ]
        self.type_roles = {
            event_type: list(schema[event_type])
            for event_type in self.event_types
        }
        self.event_to_id = {
            event_type: index
            for index, event_type in enumerate(self.event_types)
        }
        self.id_to_event = {
            index: event_type
            for event_type, index in self.event_to_id.items()
        }
        self.roles = sorted(
            {
                role
                for roles in self.type_roles.values()
                for role in roles
            }
        )

    def valid_roles(self, event_type: str) -> set[str]:
        return set(self.type_roles.get(event_type, []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_types": self.event_types,
            "id_to_event": {
                str(index): event_type
                for index, event_type in self.id_to_event.items()
            },
            "type_roles": self.type_roles,
            "roles": self.roles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventSchema":
        """从模型目录中的 schema.json 恢复事件结构。"""

        instance = cls.__new__(cls)
        instance.event_types = list(data["event_types"])
        instance.type_roles = {
            event_type: list(roles)
            for event_type, roles in data["type_roles"].items()
        }
        instance.event_to_id = {
            event_type: index
            for index, event_type in enumerate(instance.event_types)
        }
        instance.id_to_event = {
            index: event_type
            for event_type, index in instance.event_to_id.items()
        }
        instance.roles = list(data["roles"])
        return instance


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """逐行读取 JSONL 数据。"""

    with Path(path).open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def filter_by_ids(
    rows: list[dict[str, Any]],
    article_ids: set[str],
) -> list[dict[str, Any]]:
    """按文章 id 过滤数据，保证同一文章不会跨集合。"""

    return [row for row in rows if row["id"] in article_ids]


def ensure_article_split(
    detect_rows: list[dict[str, Any]],
    split_path: str | Path,
    val_ratio: float,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """按文章 id 生成并复用训练/验证切分。"""

    split_path = Path(split_path)
    if split_path.is_file() and not overwrite:
        return json.loads(split_path.read_text(encoding="utf-8"))

    article_types: dict[str, set[str]] = defaultdict(set)
    for row in detect_rows:
        article_types[row["id"]].update(row["types"])

    article_ids = sorted(article_types)
    rng = random.Random(seed)
    rng.shuffle(article_ids)

    # 每种事件类型至少保留一个验证文章，降低小类验证集为空的概率。
    validation_ids: set[str] = set()
    for event_type in sorted(
        {event_type for values in article_types.values() for event_type in values}
    ):
        candidates = [
            article_id
            for article_id in article_ids
            if event_type in article_types[article_id]
            and article_id not in validation_ids
        ]
        if candidates:
            validation_ids.add(candidates[0])

    # 再随机补足到目标验证比例。
    target_val = max(1, round(len(article_ids) * val_ratio))
    for article_id in article_ids:
        if len(validation_ids) >= target_val:
            break
        validation_ids.add(article_id)

    train_ids = [
        article_id
        for article_id in article_ids
        if article_id not in validation_ids
    ]
    split = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(validation_ids),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return split


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-12)


def binary_f1(
    gold: list[list[int]],
    prediction: list[list[int]],
    class_index: int,
) -> float:
    """计算单个二分类标签的 F1。"""

    tp = fp = fn = 0
    for gold_row, pred_row in zip(gold, prediction):
        gold_value = gold_row[class_index]
        pred_value = pred_row[class_index]
        tp += gold_value == 1 and pred_value == 1
        fp += gold_value == 0 and pred_value == 1
        fn += gold_value == 1 and pred_value == 0
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def detector_metrics(
    gold: list[list[int]],
    prediction: list[list[int]],
    event_types: list[str],
) -> dict[str, float]:
    """计算多标签事件检测指标。"""

    flat_gold = [value for row in gold for value in row]
    flat_pred = [value for row in prediction for value in row]
    tp = sum(g == 1 and p == 1 for g, p in zip(flat_gold, flat_pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(flat_gold, flat_pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(flat_gold, flat_pred))
    micro_f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    type_f1 = [
        binary_f1(gold, prediction, index)
        for index in range(len(event_types))
    ]

    has_gold = [int(any(row)) for row in gold]
    has_pred = [int(any(row)) for row in prediction]
    has_tp = sum(g == 1 and p == 1 for g, p in zip(has_gold, has_pred))
    has_fp = sum(g == 0 and p == 1 for g, p in zip(has_gold, has_pred))
    has_fn = sum(g == 1 and p == 0 for g, p in zip(has_gold, has_pred))
    return {
        "has_event_accuracy": sum(
            g == p for g, p in zip(has_gold, has_pred)
        ) / max(len(gold), 1),
        "has_event_precision": _safe_div(has_tp, has_tp + has_fp),
        "has_event_recall": _safe_div(has_tp, has_tp + has_fn),
        "has_event_f1": _safe_div(2 * has_tp, 2 * has_tp + has_fp + has_fn),
        "type_micro_f1": micro_f1,
        "type_macro_f1": sum(type_f1) / max(len(type_f1), 1),
        "subset_accuracy": sum(
            gold_row == pred_row
            for gold_row, pred_row in zip(gold, prediction)
        ) / max(len(gold), 1),
    }


def tune_threshold(
    probabilities: list[list[float]],
    gold: list[list[int]],
    event_types: list[str],
) -> tuple[float, dict[str, float]]:
    """在验证集上选择多标签统一阈值。"""

    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_score = (-1.0, -1.0)
    for step in range(1, 20):
        threshold = step / 20
        prediction = [
            [int(probability >= threshold) for probability in row]
            for row in probabilities
        ]
        metrics = detector_metrics(gold, prediction, event_types)
        # 先优化类型 micro-F1，再在同分时比较多标签完全匹配。
        # 不把 subset_accuracy 直接相加，避免大量负样本导致“全不预测”占优。
        score = metrics["type_micro_f1"]
        tie_breaker = metrics["subset_accuracy"]
        pair = (score, tie_breaker)
        if pair > best_score:
            best_score = pair
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics

