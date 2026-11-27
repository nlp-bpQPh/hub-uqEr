"""Check span logic and replay a deterministic slice of each implementation."""

import argparse
import hashlib
import json
from collections import Counter

from data import CACHE, MODEL_FILES, RESULTS, api_sample, fetch_model, load_data, make_record, sha256
from evaluate import counts, metrics, read_json, read_predictions
from extract import API_BATCH, API_METHOD, Gazetteer, ground_api_items, make_distilbert_predictor, make_spacy_predictor


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_span_logic():
    record = make_record("toy", ["Zoë", "visited", "New", "York", "."],
                         ["B-PER", "O", "B-LOC", "I-LOC", "O"])
    require(record.gold == {(0, 3, "PER"), (12, 20, "LOC")}, "BIO offsets disagree with the hand-labeled example")
    partial = {(0, 3, "PER"), (12, 15, "LOC"), (12, 20, "PER")}
    require(counts(record.gold, partial) == (1, 2, 1) and metrics(1, 2, 1)["f1"] == 0.4,
            "Exact boundary and type scoring accepted a partial match")
    raw = '{"items":[{"id":0,"entities":[{"text":"New York","label":"LOC"},{"text":"Zo","label":"PER"}]}]}'
    predictions, diagnostics = ground_api_items([record], raw)
    require(predictions == [{(12, 20, "LOC")}] and diagnostics["ungrounded_entities"] == 1,
            "Grounding does not reject a partial word")
    require(ground_api_items([record], "{")[1]["missing_records"] == 1, "Missing API output was not counted")
    print("PASS: hand-checked Unicode offsets, strict scoring, and invalid API output handling")


def verify_metrics(scores, records, predictions):
    gold = sum(len(record.gold) for record in records)
    predicted = sum(len(predictions[record.id]) for record in records)
    hit = sum(len(record.gold.intersection(predictions[record.id])) for record in records)
    require(scores["records"] == len(records) and scores["tp"] == hit and scores["fp"] == predicted - hit
            and scores["fn"] == gold - hit, "Published record or entity counts disagree")
    for key in ("tp", "fp", "fn"):
        require(sum(values[key] for values in scores["per_label"].values()) == scores[key], "Per-label totals disagree")
    f1 = 2 * hit / (gold + predicted) if gold + predicted else 0.0
    require(abs(scores["f1"] - f1) < 1e-12, "Published F1 disagrees")


def verify_local(sources, test, api_run):
    core_run = read_json(RESULTS / "core_run.json")
    model_path = fetch_model()
    require(core_run["distilbert"]["files_sha256"] == {name: sha256(model_path / name) for name in MODEL_FILES},
            "DistilBERT files disagree with the recorded hashes")
    for method, predict in (("spacy", make_spacy_predictor()),
                            ("distilbert", make_distilbert_predictor(core_run["torch_threads"]))):
        for dataset, records in test.items():
            saved = read_predictions(method, dataset, records)
            sample = api_sample(records)[:16]
            require(predict(sample) == [saved[record.id] for record in sample], f"{method} replay changed for {dataset}")
    print("PASS: model file hashes and 64 fresh spaCy/DistilBERT predictions across both datasets")
    settings = {key: value for key, value in api_run.items() if key != "timing"}
    cache_key = hashlib.sha256(json.dumps([settings, sources], sort_keys=True).encode()).hexdigest()[:16]
    run_dir = CACHE / f"api-{cache_key}"
    for dataset, records in test.items():
        sample = api_sample(records)
        saved = read_predictions(API_METHOD, dataset, sample)
        replayed = []
        counters = Counter()
        for start in range(0, len(sample), API_BATCH):
            batch = sample[start:start + API_BATCH]
            raw = read_json(run_dir / f"{dataset}-{start:03}.json")
            require(raw["ids"] == [record.id for record in batch], "Cached raw API IDs changed")
            spans, diagnostics = ground_api_items(batch, raw["raw"])
            replayed.extend(spans)
            counters.update(diagnostics)
        require(replayed == list(saved.values()), "Grounding the cached API outputs changed predictions")
        require(all(api_run["timing"][dataset]["diagnostics"][key] == value for key, value in counters.items()),
                "API diagnostic counts changed")
    print("PASS: replayed all 24 cached HTTP responses and recovered all 96 grounded predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-local", action="store_true", help="also rerun local models and replay cached API payloads")
    args = parser.parse_args()
    verify_span_logic()
    train, test, sources = load_data()
    require(sources == read_json(RESULTS / "sources.json"), "Dataset sources or hashes changed")
    scores = read_json(RESULTS / "scores.json")
    sample_ids = read_json(RESULTS / "api_sample_ids.json")
    train_texts = {record.text for record in train}
    dictionary = Gazetteer(train)
    for dataset, records in test.items():
        sample = api_sample(records)
        require(sample_ids[dataset] == [record.id for record in sample], "Matched sample changed")
        for method, full_scores in scores["full"][dataset].items():
            predictions = read_predictions(method, dataset, records)
            verify_metrics(full_scores, records, predictions)
            verify_metrics(scores["matched"][dataset][method], sample, predictions)
            if method == "gazetteer":
                require(dictionary.predict(records) == list(predictions.values()), "Dictionary replay changed")
        verify_metrics(scores["matched"][dataset][API_METHOD], sample, read_predictions(API_METHOD, dataset, sample))
        print(f"PASS: {dataset} source hash, aligned sample, local baseline replay and all metric counts; "
              f"exact test records also in CoNLL train={sum(record.text in train_texts for record in records)}")
    if args.replay_local:
        verify_local(sources, test, read_json(RESULTS / "api_run.json"))


if __name__ == "__main__":
    main()
