"""Character-level BIO features for XFUND blocks, following the course setup."""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils.data import Dataset

from xfund_data import ID_TO_TAG, TAG_TO_CLASS, TAG_TO_ID, block_label, clean_text

BOUNDARY = "\u0001"


def _block_units(doc: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    """Chunk at OCR-block boundaries where possible; retain source block IDs."""
    result: list[dict[str, Any]] = []
    current: list[tuple[str, int, int]] = []
    for block_idx, block in enumerate(doc.get("document", [])):
        text = clean_text(block.get("text", ""))
        if not text:
            continue
        label = block_label(block)
        if label == "other":
            ids = [TAG_TO_ID["O"]] * len(text)
        else:
            ids = [TAG_TO_ID[f"B-{label}"]] + [TAG_TO_ID[f"I-{label}"]] * (len(text) - 1)
        block_units = [(ch, tag_id, block_idx) for ch, tag_id in zip(text, ids)]

        if len(block_units) > max_chars:
            if current:
                result.append(_pack(current, doc))
                current = []
            for start in range(0, len(block_units), max_chars):
                part = block_units[start:start + max_chars]
                if start and label != "other":
                    part[0] = (part[0][0], TAG_TO_ID[f"B-{label}"], block_idx)
                result.append(_pack(part, doc))
            continue

        needed = len(block_units) + (1 if current else 0)
        if current and len(current) + needed > max_chars:
            result.append(_pack(current, doc))
            current = []
        if current:
            current.append((BOUNDARY, TAG_TO_ID["O"], -1))
        current.extend(block_units)

    if current:
        result.append(_pack(current, doc))
    return result


def _pack(units: list[tuple[str, int, int]], doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": str(doc.get("id", "")),
        "chars": [u[0] for u in units],
        "tag_ids": [u[1] for u in units],
        "block_ids": [u[2] for u in units],
    }


def make_char_vocab(docs: list[dict[str, Any]]) -> dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1, BOUNDARY: 2}
    chars = sorted({ch for doc in docs for block in doc.get("document", [])
                    for ch in clean_text(block.get("text", ""))})
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def make_bilstm_examples(docs: list[dict[str, Any]], vocab: dict[str, int],
                         max_chars: int = 240) -> list[dict[str, Any]]:
    examples = []
    for doc in docs:
        for chunk in _block_units(doc, max_chars):
            examples.append({
                **chunk,
                "input_ids": [vocab.get(ch, vocab["<UNK>"]) for ch in chunk["chars"]],
                "first_token": [ch != BOUNDARY for ch in chunk["chars"]],
            })
    return examples


def make_bert_examples(docs: list[dict[str, Any]], tokenizer: Any,
                       max_chars: int = 240, max_len: int = 256) -> list[dict[str, Any]]:
    examples = []
    for doc in docs:
        for chunk in _block_units(doc, max_chars):
            split_words = [tokenizer.sep_token if ch == BOUNDARY else ch for ch in chunk["chars"]]
            encoded = tokenizer(split_words, is_split_into_words=True,
                                add_special_tokens=True, truncation=False)
            word_ids = encoded.word_ids()
            if len(encoded["input_ids"]) > max_len:
                raise ValueError(f"A character chunk expanded to {len(encoded['input_ids'])} tokens; reduce max_chars")

            labels_crf: list[int] = []
            labels_ce: list[int] = []
            token_block_ids: list[int] = []
            first_token: list[bool] = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    labels_crf.append(TAG_TO_ID["O"])
                    labels_ce.append(-100)
                    token_block_ids.append(-1)
                    first_token.append(False)
                    continue
                is_first = wid != prev_word
                tag_id = chunk["tag_ids"][wid]
                tag = ID_TO_TAG[tag_id]
                if not is_first and tag.startswith("B-"):
                    tag_id = TAG_TO_ID["I-" + tag[2:]]
                labels_crf.append(tag_id)
                labels_ce.append(tag_id if is_first else -100)
                token_block_ids.append(chunk["block_ids"][wid])
                first_token.append(is_first and chunk["block_ids"][wid] >= 0)
                prev_word = wid

            examples.append({
                "doc_id": chunk["doc_id"],
                "input_ids": encoded["input_ids"],
                "labels_crf": labels_crf,
                "labels_ce": labels_ce,
                "block_ids": token_block_ids,
                "first_token": first_token,
            })
    return examples


def make_collate(pad_id: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(x["input_ids"]) for x in batch)
        n = len(batch)
        input_ids = torch.full((n, width), pad_id, dtype=torch.long)
        attention = torch.zeros((n, width), dtype=torch.bool)
        labels_crf = torch.zeros((n, width), dtype=torch.long)
        labels_ce = torch.full((n, width), -100, dtype=torch.long)
        block_ids = torch.full((n, width), -1, dtype=torch.long)
        first = torch.zeros((n, width), dtype=torch.bool)
        for i, row in enumerate(batch):
            length = len(row["input_ids"])
            input_ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention[i, :length] = True
            labels_crf[i, :length] = torch.tensor(row.get("labels_crf", row.get("tag_ids")), dtype=torch.long)
            labels_ce[i, :length] = torch.tensor(row.get("labels_ce", row.get("tag_ids")), dtype=torch.long)
            block_ids[i, :length] = torch.tensor(row["block_ids"], dtype=torch.long)
            first[i, :length] = torch.tensor(row["first_token"], dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels_crf": labels_crf,
            "labels_ce": labels_ce,
            "block_ids": block_ids,
            "first_token": first,
            "doc_ids": [x["doc_id"] for x in batch],
            "example_indices": [x.get("_index", -1) for x in batch],
        }
    return collate


def predictions_to_blocks(docs: list[dict[str, Any]], examples: list[dict[str, Any]],
                          predicted_tags: list[list[int]]) -> dict[str, list[str]]:
    """Majority-vote decoded character labels back to original OCR blocks."""
    votes: dict[str, dict[int, Counter[str]]] = {}
    for ex, tags in zip(examples, predicted_tags):
        doc_id = ex["doc_id"]
        doc_votes = votes.setdefault(doc_id, {})
        for block_idx, is_first, tag_id in zip(ex["block_ids"], ex["first_token"], tags):
            if block_idx < 0 or not is_first:
                continue
            tag = ID_TO_TAG.get(int(tag_id), "O")
            cls = "other" if tag == "O" else tag[2:]
            doc_votes.setdefault(block_idx, Counter())[cls] += 1

    output: dict[str, list[str]] = {}
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        doc_votes = votes.get(doc_id, {})
        labels = []
        for idx, _ in enumerate(doc.get("document", [])):
            tally = doc_votes.get(idx)
            if not tally:
                labels.append("other")
            else:
                labels.append(max(("other", "header", "question", "answer"),
                                  key=lambda cls: tally.get(cls, 0)))
        output[doc_id] = labels
    return output


class FeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = dict(self.rows[i])
        row["_index"] = i
        return row
