"""Utilities for XFUND's OCR-block annotations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLASSES = ["other", "header", "question", "answer"]
ENTITY_CLASSES = ["header", "question", "answer"]
TAGS = ["O"] + [tag for label in ENTITY_CLASSES for tag in (f"B-{label}", f"I-{label}")]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}
TAG_TO_CLASS = {"O": "other"}
for _label in ENTITY_CLASSES:
    TAG_TO_CLASS[f"B-{_label}"] = _label
    TAG_TO_CLASS[f"I-{_label}"] = _label


def load_xfund(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("lang") != "zh":
        raise ValueError(f"Expected Chinese XFUND data, got lang={data.get('lang')!r}")
    docs = data.get("documents")
    if not isinstance(docs, list):
        raise ValueError("XFUND JSON has no documents list")
    return docs


def block_label(block: dict[str, Any]) -> str:
    label = block.get("label", "other")
    return label if label in CLASSES else "other"


def clean_text(text: str) -> str:
    return "".join(ch for ch in str(text) if not ch.isspace())


def document_char_units(doc: dict[str, Any]) -> tuple[list[str], list[int], list[int]]:
    """Return compact characters, BIO tag ids and source block indices.

    A special boundary marker (None) separates XFUND blocks. Whitespace is
    omitted because it is not consistently represented in the OCR annotations.
    """
    chars: list[str] = []
    tag_ids: list[int] = []
    block_ids: list[int] = []
    for block_idx, block in enumerate(doc.get("document", [])):
        text = clean_text(block.get("text", ""))
        label = block_label(block)
        if label == "other" or not text:
            chars.extend(text)
            tag_ids.extend([TAG_TO_ID["O"]] * len(text))
        else:
            chars.append(text[0])
            tag_ids.append(TAG_TO_ID[f"B-{label}"])
            if len(text) > 1:
                chars.extend(text[1:])
                tag_ids.extend([TAG_TO_ID[f"I-{label}"]] * (len(text) - 1))
        block_ids.extend([block_idx] * len(text))
        # A visible boundary prevents two neighboring entities from merging.
        if block_idx < len(doc.get("document", [])) - 1:
            chars.append("\u0001")
            tag_ids.append(TAG_TO_ID["O"])
            block_ids.append(-1)
    return chars, tag_ids, block_ids


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_normalized_splits(train_path: str | Path, val_path: str | Path,
                            out_dir: str | Path) -> dict[str, int]:
    """Create transparent JSONL copies, retaining blocks, boxes and links."""
    out_dir = Path(out_dir)
    train_docs = load_xfund(train_path)
    val_docs = load_xfund(val_path)
    for split, docs in (("train", train_docs), ("val", val_docs)):
        write_jsonl(out_dir / f"{split}.jsonl", docs)
    return {"train_documents": len(train_docs), "val_documents": len(val_docs)}
