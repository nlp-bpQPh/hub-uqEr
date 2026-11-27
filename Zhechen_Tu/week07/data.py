"""Load the pinned source files and align BIO tags to character offsets."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

import requests


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"
LABELS = ("PER", "LOC")
Span = tuple[int, int, str]

CONLL_URL = (
    "https://huggingface.co/datasets/lhoestq/conll2003/resolve/"
    "2aecdeb6176d9b2b1ae7c8672e1d4ff48566747c/default"
)
WNUT_URL = (
    "https://raw.githubusercontent.com/leondz/emerging_entities_17/"
    "e52a2d2a71ac1a051ca2d532eef28653c5c02603/emerging.test.annotated"
)
MODEL_ID = "dslim/distilbert-NER"
MODEL_REV = "dfa2838a127384aabb82ed7719e16dab84c42a2a"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
CONLL_TAGS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC")
TARGET_LABELS = {"PER": "PER", "LOC": "LOC", "person": "PER", "location": "LOC"}
API_SEED = 20261127
API_SAMPLE_SIZE = 48


@dataclass(frozen=True)
class Record:
    id: str
    tokens: tuple[str, ...]
    starts: tuple[int, ...]
    text: str
    gold: frozenset[Span]


def download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(25, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)
    partial.replace(path)
    return path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def make_record(record_id: str, tokens: list[str], tags: list[str]) -> Record:
    if len(tokens) != len(tags) or not tokens or any(not token for token in tokens):
        raise ValueError(f"Invalid token/tag sequence: {record_id}")
    starts = []
    offset = 0
    for token in tokens:
        starts.append(offset)
        offset += len(token) + 1
    spans = set()
    active_start = None
    active_label = None
    for index, tag in enumerate([*tags, "O"]):
        prefix, _, raw_label = tag.partition("-")
        label = TARGET_LABELS.get(raw_label)
        if prefix == "I" and label is not None and label == active_label:
            continue
        if active_start is not None:
            end = starts[index - 1] + len(tokens[index - 1])
            spans.add((starts[active_start], end, active_label))
        active_start = index if label is not None and prefix in {"B", "I"} else None
        active_label = label if active_start is not None else None
    return Record(record_id, tuple(tokens), tuple(starts), " ".join(tokens), frozenset(spans))


def load_conll(split: str) -> tuple[list[Record], dict]:
    import pyarrow.parquet as pq

    url = f"{CONLL_URL}/{split}/0000.parquet"
    path = download(url, CACHE / f"conll-{split}.parquet")
    table = pq.read_table(path, columns=["id", "tokens", "ner_tags"])
    rows = []
    for record in table.to_pylist():
        tokens = record["tokens"]
        if tokens and tokens[0] != "-DOCSTART-":
            tags = [CONLL_TAGS[tag] for tag in record["ner_tags"]]
            rows.append(make_record(f"conll-{split}-{record['id']}", tokens, tags))
    return rows, {"url": url, "sha256": sha256(path), "records": len(rows), "source_rows": len(table)}


def load_wnut() -> tuple[list[Record], dict]:
    path = download(WNUT_URL, CACHE / "wnut-test.annotated")
    rows = []
    tokens = []
    tags = []
    for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
        if line.strip():
            token, tag = line.rsplit("\t", maxsplit=1)
            tokens.append(token)
            tags.append(tag)
        elif tokens:
            rows.append(make_record(f"wnut-test-{len(rows)}", tokens, tags))
            tokens, tags = [], []
    return rows, {"url": WNUT_URL, "sha256": sha256(path), "records": len(rows)}


def load_data() -> tuple[list[Record], dict[str, list[Record]], dict]:
    train, train_meta = load_conll("train")
    conll, conll_meta = load_conll("test")
    wnut, wnut_meta = load_wnut()
    return train, {"conll": conll, "wnut": wnut}, {
        "conll_train": train_meta, "conll_test": conll_meta, "wnut_test": wnut_meta,
    }


def api_sample(records: list[Record]) -> list[Record]:
    indices = sorted(Random(API_SEED).sample(range(len(records)), API_SAMPLE_SIZE))
    return [records[index] for index in indices]


def fetch_model() -> Path:
    target = CACHE / "distilbert-ner"
    for filename in MODEL_FILES:
        download(f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REV}/{filename}", target / filename)
    return target


if __name__ == "__main__":
    import sys

    if "--model-only" in sys.argv:
        print(fetch_model())
    else:
        _, _, manifest = load_data()
        print(json.dumps(manifest, indent=2))
