"""Run full test splits for local NER, and a matched sample for the HTTP API."""

import argparse
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from time import perf_counter

from data import API_SAMPLE_SIZE, API_SEED, CACHE, MODEL_FILES, MODEL_ID, MODEL_REV, RESULTS, api_sample, fetch_model, load_data, sha256, write_json
from extract import API_BATCH, API_METHOD, API_MODEL, API_OPTIONS, API_PROMPT, API_SCHEMA, OllamaExtractor, Gazetteer, ground_api_items, make_distilbert_predictor, make_spacy_predictor


def save_predictions(method: str, dataset: str, records, predictions) -> None:
    if len(records) != len(predictions):
        raise ValueError(f"Prediction count mismatch for {method}/{dataset}")
    path = RESULTS / "predictions" / f"{dataset}-{method}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record, spans in zip(records, predictions, strict=True):
            output.write(json.dumps({"id": record.id, "spans": sorted(spans)}, separators=(",", ":")) + "\n")


def run_core(train, test, threads: int) -> None:
    started = perf_counter()
    dictionary = Gazetteer(train)
    dictionary_fit = perf_counter() - started
    timing = {}
    for name, predictor_factory in (
        ("gazetteer", lambda: dictionary.predict),
        ("spacy", make_spacy_predictor),
        ("distilbert", lambda: make_distilbert_predictor(threads)),
    ):
        predict = predictor_factory()
        predict(test["conll"][:12])  # Warm up the implementation before measuring.
        timing[name] = {}
        for dataset, records in test.items():
            started = perf_counter()
            predictions = predict(records)
            seconds = perf_counter() - started
            save_predictions(name, dataset, records, predictions)
            timing[name][dataset] = {"seconds": seconds, "records": len(records)}
            print(f"{name} / {dataset}: {len(records)} records in {seconds:.2f}s", flush=True)
        del predict
    model_path = fetch_model()
    write_json(RESULTS / "core_run.json", {
        "system": platform.platform(), "python": platform.python_version(), "torch_threads": threads,
        "versions": {name: version(name) for name in ("torch", "transformers", "spacy", "en-core-web-sm", "numpy", "pyarrow")},
        "distilbert": {"model": MODEL_ID, "revision": MODEL_REV, "aggregation": "first", "stride": 64,
                       "batch_size": 16, "files_sha256": {name: sha256(model_path / name) for name in MODEL_FILES}},
        "gazetteer": {"entries": len(dictionary.entries), "fit_seconds": dictionary_fit}, "timing": timing,
    })


def run_api(test, sources) -> None:
    extractor = OllamaExtractor()
    settings = {"model": API_MODEL, "model_digest": extractor.digest, "ollama_version": extractor.version,
                "endpoint": "/api/generate", "think": False, "batch_size": API_BATCH, "options": API_OPTIONS,
                "seed": API_SEED, "sample_per_dataset": API_SAMPLE_SIZE, "prompt": API_PROMPT, "schema": API_SCHEMA}
    cache_key = hashlib.sha256(json.dumps([settings, sources], sort_keys=True).encode()).hexdigest()[:16]
    run_dir = CACHE / f"api-{cache_key}"
    timing = {}
    for dataset, full_records in test.items():
        records = api_sample(full_records)
        predictions = []
        elapsed = []
        counters = Counter()
        for start in range(0, len(records), API_BATCH):
            batch = records[start:start + API_BATCH]
            path = run_dir / f"{dataset}-{start:03}.json"
            if path.exists():
                call = json.loads(path.read_text(encoding="utf-8"))
            else:
                call = extractor.predict_batch(batch)
                partial = path.with_suffix(".part")
                write_json(partial, call)
                partial.replace(path)
            if call["ids"] != [record.id for record in batch]:
                raise ValueError(f"Cached API batch does not match {dataset}/{start}")
            batch_predictions, diagnostics = ground_api_items(batch, call["raw"])
            predictions.extend(batch_predictions)
            counters.update(diagnostics)
            counters.update({f"done_{call['done_reason']}": 1, "generated_tokens": call["eval_count"]})
            elapsed.append({key: call[key] for key in ("elapsed_seconds", "load_duration", "total_duration")})
            print(f"{API_METHOD} / {dataset}: {start + len(batch)}/{len(records)} ({call['elapsed_seconds']:.2f}s)", flush=True)
        save_predictions(API_METHOD, dataset, records, predictions)
        timing[dataset] = {"seconds": sum(item["elapsed_seconds"] for item in elapsed), "records": len(records),
                           "batches": elapsed, "diagnostics": dict(counters)}
    write_json(RESULTS / "api_run.json", {**settings, "timing": timing})


def run_wordpiece_ablation(train, test, threads: int) -> None:
    from evaluate import summarize

    predict = make_distilbert_predictor(threads, aggregation="simple")
    dictionary = Gazetteer(train)
    comparison = {}
    fields = ("records", "tp", "fp", "fn", "precision", "recall", "f1", "error_counts")
    for dataset, records in test.items():
        predictions = {record.id: spans for record, spans in zip(records, predict(records), strict=True)}
        summary, _ = summarize(records, predictions, dictionary)
        comparison[dataset] = {field: summary[field] for field in fields}
        print(f"simple WordPiece aggregation / {dataset}: F1={summary['f1']:.2%}", flush=True)
    write_json(RESULTS / "wordpiece_ablation.json", {"aggregation": "simple", "full": comparison})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "api", "wordpiece", "all"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    train, test, sources = load_data()
    write_json(RESULTS / "sources.json", sources)
    print(f"Loaded train={len(train)}, " + ", ".join(f"{key}={len(value)}" for key, value in test.items()), flush=True)
    if args.stage in {"core", "all"}:
        run_core(train, test, args.threads)
    if args.stage in {"api", "all"}:
        run_api(test, sources)
    if args.stage == "wordpiece":
        run_wordpiece_ablation(train, test, args.threads)


if __name__ == "__main__":
    main()
