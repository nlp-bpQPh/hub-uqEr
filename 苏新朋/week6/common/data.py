"""CNews 清洗、读写与固定分层切分数据加载。"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"


def read_tsv(path):
    """读取 label<TAB>text 格式的数据。"""
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                records.append({"label": row[0].strip(), "text": " ".join(row[1:]).strip()})
    return records


def clean_source(source):
    """清洗原始 CNews，并统计剔除原因。"""
    records, seen = [], set()
    audit = Counter()
    with Path(source).open("r", encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip():
                audit["blank_lines"] += 1
                continue
            if "\t" not in line:
                audit["malformed_lines"] += 1
                continue
            label, text = line.split("\t", 1)
            label = label.strip()
            text = re.sub(r"\s+", " ", text).strip()
            if not label or not text:
                audit["empty_label_or_text"] += 1
                continue
            key = (label, text)
            if key in seen:
                audit["duplicate_records"] += 1
                continue
            seen.add(key)
            records.append({"label": label, "text": text})
            audit["kept_records"] += 1
    if not records:
        raise ValueError(f"清洗后没有有效数据：{source}")
    return records, dict(audit)


def save_tsv(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows((row["label"], row["text"]) for row in records)


def load_or_prepare_data(source=None):
    """默认加载本地固定切分；传入 source 时重新清洗、分层切分并保存。"""
    if source is None:
        split_paths = {name: DATA_DIR / f"{name}.tsv" for name in ("train", "valid", "test")}
        if not all(path.is_file() for path in split_paths.values()):
            raise FileNotFoundError(
                "找不到 week6/data 中的 train/valid/test.tsv；请用 --source 指定原始数据。"
            )
        splits = {name: read_tsv(path) for name, path in split_paths.items()}
        cleaned_path = DATA_DIR / "cnews.cleaned.tsv"
        cleaned = read_tsv(cleaned_path)
        metrics_path = RESULT_DIR / "metrics.json"
        audit = {}
        if metrics_path.is_file():
            try:
                audit = json.loads(metrics_path.read_text(encoding="utf-8")).get("cleaning_audit", {})
            except (json.JSONDecodeError, OSError):
                audit = {}
        if not audit:
            audit = {"kept_records": len(cleaned), "note": "由本目录已保存的清洗数据加载"}
        note = (
            "从 week6/data 中读取预先保存的去重分层切分；数据源最初仅提供 cnews.test.txt，"
            "因此这是单文件内部留出评测，不是官方划分。"
        )
        return splits["train"], splits["valid"], splits["test"], cleaned, audit, note

    records, audit = clean_source(source)
    from sklearn.model_selection import train_test_split

    train, remainder = train_test_split(
        records, test_size=0.2, random_state=SEED,
        stratify=[row["label"] for row in records],
    )
    valid, test = train_test_split(
        remainder, test_size=0.5, random_state=SEED,
        stratify=[row["label"] for row in remainder],
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_tsv(DATA_DIR / "cnews.cleaned.tsv", records)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        save_tsv(DATA_DIR / f"{name}.tsv", rows)
    note = f"从 {Path(source).name} 清洗后按 80/10/10 分层切分；这是单文件内部留出评测，不是官方 train/test 划分。"
    return train, valid, test, records, audit, note
