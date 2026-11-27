"""Block-level evaluation for XFUND's four OCR entity classes."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from xfund_data import CLASSES, ENTITY_CLASSES, block_label


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def evaluate_blocks(docs: list[dict[str, Any]],
                    predictions: dict[str, list[str]]) -> dict[str, Any]:
    counts = {c: [0, 0, 0] for c in CLASSES}  # tp, fp, fn
    confusion = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    total = correct = exact_docs = 0
    entity_tp = entity_fp = entity_fn = 0
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        blocks = doc.get("document", [])
        pred = predictions.get(doc_id, [])
        doc_ok = len(pred) == len(blocks)
        for i, block in enumerate(blocks):
            gold = block_label(block)
            guess = pred[i] if i < len(pred) and pred[i] in CLASSES else "other"
            confusion[gold][guess] += 1
            total += 1
            correct += int(gold == guess)
            doc_ok = doc_ok and gold == guess
            for cls in CLASSES:
                if guess == cls and gold == cls:
                    counts[cls][0] += 1
                elif guess == cls:
                    counts[cls][1] += 1
                elif gold == cls:
                    counts[cls][2] += 1
            if guess in ENTITY_CLASSES and guess == gold:
                entity_tp += 1
            else:
                if guess in ENTITY_CLASSES:
                    entity_fp += 1
                if gold in ENTITY_CLASSES:
                    entity_fn += 1
        exact_docs += int(doc_ok)

    per_class = {c: _prf(*counts[c]) for c in CLASSES}
    result = {
        "documents": len(docs),
        "blocks": total,
        "block_accuracy": correct / total if total else 0.0,
        "document_all_blocks_exact": exact_docs / len(docs) if docs else 0.0,
        "macro_f1_all_four_classes": sum(per_class[c]["f1"] for c in CLASSES) / len(CLASSES),
        "entity_micro": _prf(entity_tp, entity_fp, entity_fn),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "link_pairs_evaluated": False,
        "note": "Scores classify the supplied OCR blocks; XFUND question-answer links are retained but not scored.",
    }
    return result


def save_predictions(path: str | Path, docs: list[dict[str, Any]],
                     predictions: dict[str, list[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            doc_id = str(doc.get("id", ""))
            pred = predictions.get(doc_id, [])
            rows = []
            for i, block in enumerate(doc.get("document", [])):
                rows.append({
                    "block_index": i,
                    "source_id": block.get("id"),
                    "text": block.get("text", ""),
                    "box": block.get("box"),
                    "gold": block_label(block),
                    "predicted": pred[i] if i < len(pred) else "other",
                    "links": block.get("linking", []),
                })
            f.write(json.dumps({"doc_id": doc_id, "blocks": rows}, ensure_ascii=False) + "\n")


def save_report(out_dir: str | Path, docs: list[dict[str, Any]],
                predictions: dict[str, list[str]], method: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"method": method, "metrics": evaluate_blocks(docs, predictions)}
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(out_dir / "predictions.jsonl", docs, predictions)
    return report
