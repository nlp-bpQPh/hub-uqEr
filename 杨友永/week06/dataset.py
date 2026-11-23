"""② Dataset：商品名 → 品类（10 分类）。

数据格式（jsonl，每行一条）：
    {"product_name": "优百润珍珠奶茶180g", "category": "即饮奶茶"}

只提供 train.jsonl / test.jsonl，没有验证集；
本模块从 train 里按类别分层切出一份验证集（默认 10%）。

    ds = ProductDataset("train")
    ds = ProductDataset("val")
    ds = ProductDataset("test")
    text, label = ds[0]
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from torch.utils.data import Dataset

BASE_DIR: Final = Path(__file__).resolve().parent
DATASET_NAME: Final = "product-classification-hiring-demo"
FILES: Final = {"train": "train.jsonl", "test": "test.jsonl"}


def resolve_data_dir() -> Path:
    """定位数据集目录。

    数据位置调整过几次，这里按优先级依次尝试：
    1. 环境变量 PRODUCT_DATA_DIR（最高优先级，方便临时切换）
    2. data/<数据集名>
    3. <数据集名>（旧位置）
    都找不到就报错，并列出尝试过的路径。
    """
    env = os.environ.get("PRODUCT_DATA_DIR")
    candidates = ([Path(env)] if env else []) + [
        BASE_DIR / "data" / DATASET_NAME,
        BASE_DIR / DATASET_NAME,
    ]
    for path in candidates:
        if (path / FILES["train"]).exists():
            return path
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"找不到数据集目录，已尝试:\n{tried}")


DATA_DIR: Final = resolve_data_dir()
VAL_RATIO: Final = 0.1
SEED: Final = 42


@dataclass(frozen=True, slots=True)
class Sample:
    """一条样本。slots=True 让 2 万条数据省掉约一半对象开销。"""
    text: str          # 商品名
    label: int         # 类别 id
    category: str      # 类别名（便于报错和错误分析）


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_label_map() -> dict[str, int]:
    """从训练集构建 类别名 → id。

    按名称排序而不是按频率：频率会随数据变化，排序不会。
    这样换一份数据、换个随机种子，类别 id 仍然稳定。
    """
    rows = read_jsonl(DATA_DIR / FILES["train"])
    names = sorted({r["category"] for r in rows})
    return {name: i for i, name in enumerate(names)}


def _stratified_split(rows: list[dict], ratio: float, seed: int
                      ) -> tuple[list[dict], list[dict]]:
    """按类别分层切分，保证验证集的类别比例和训练集一致。"""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    train, val = [], []
    for items in by_cat.values():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class ProductDataset(Dataset[Sample]):
    """商品名分类数据集。

    split:
        "train"  训练集（去掉验证集后的部分）
        "val"    从训练集分层切出的验证集
        "test"   官方测试集
    """

    def __init__(self, split: str = "train", path: str | Path | None = None,
                 limit: int | None = None, label_map: dict[str, int] | None = None,
                 val_ratio: float = VAL_RATIO, seed: int = SEED) -> None:
        self.split = split
        self.label_map = label_map or build_label_map()
        self.id2name = {v: k for k, v in self.label_map.items()}

        rows = read_jsonl(Path(path) if path else self._path(split))
        if split == "val":
            _, rows = _stratified_split(rows, val_ratio, seed)
        elif split == "train":
            rows, _ = _stratified_split(rows, val_ratio, seed)

        if limit is not None:
            rows = rows[:limit]

        self.items: list[Sample] = []
        for lineno, r in enumerate(rows, 1):
            text, cat = r.get("product_name"), r.get("category")
            if not text or cat not in self.label_map:
                raise ValueError(f"{split}:{lineno} 数据非法: {r}")
            self.items.append(Sample(text=text, label=self.label_map[cat], category=cat))

    @staticmethod
    def _path(split: str) -> Path:
        key = "train" if split in ("train", "val") else split
        if key not in FILES:
            raise ValueError(f"split 必须是 train/val/test，得到 {split!r}")
        return DATA_DIR / FILES[key]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Sample:
        return self.items[index]

    # ---------- 便于数据总览 / 训练 ---------- #
    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def class_counts(self) -> dict[str, int]:
        c = Counter(s.category for s in self.items)
        return {name: c.get(name, 0) for name in self.label_map}

    def describe(self) -> dict:
        lens = [len(s.text) for s in self.items]
        counts = self.class_counts()
        return {
            "split": self.split,
            "条数": len(self.items),
            "类别数": self.num_classes,
            "平均长度": round(sum(lens) / len(lens), 1) if lens else 0,
            "最长": max(lens) if lens else 0,
            "最多类": max(counts, key=counts.get) if counts else None,
            "最少类": min(counts, key=counts.get) if counts else None,
            "不均衡比": round(max(counts.values()) / max(1, min(counts.values())), 1) if counts else 0,
        }


if __name__ == "__main__":
    lm = build_label_map()
    print(f"类别 {len(lm)} 个: {lm}\n")
    total = 0
    for split in ("train", "val", "test"):
        ds = ProductDataset(split)
        total += len(ds)
        print(ds.describe())
    print(f"\n合计 {total} 条（应为 20,000）")
    s = ProductDataset("train")[0]
    print(f"\n首条: text={s.text!r} label={s.label} category={s.category!r}")
