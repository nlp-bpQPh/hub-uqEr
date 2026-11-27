"""Verify saved predictions, score exact spans, and generate comparison artifacts."""

import csv
import itertools
import json
from collections import Counter

import numpy as np

from data import LABELS, RESULTS, api_sample, load_data, write_json
from extract import API_METHOD, CORE_METHODS, Gazetteer

METHODS = (*CORE_METHODS, API_METHOD)
DATASET_NAMES = {"conll": "CoNLL-2003 (news)", "wnut": "WNUT-17 (forum posts)"}
METHOD_NAMES = {"gazetteer": "Gazetteer", "spacy": "spaCy", "distilbert": "DistilBERT", "gemma_api": "Gemma API"}
COLORS = {"gazetteer": "#687684", "spacy": "#16796f", "distilbert": "#335dc0", "gemma_api": "#ae5728"}
BOOTSTRAP_SEED = 7341
BOOTSTRAP_REPEATS = 2000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(method, dataset, records):
    path = RESULTS / "predictions" / f"{dataset}-{method}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row["id"] for row in rows] != [record.id for record in records]:
        raise ValueError(f"Prediction IDs are missing, duplicated, or out of order: {path.name}")
    predictions = {}
    for record, row in zip(records, rows, strict=True):
        spans = set()
        for start, end, label in row["spans"]:
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(record.text) or label not in LABELS:
                raise ValueError(f"Invalid predicted span for {record.id}")
            if (start, end, label) in spans:
                raise ValueError(f"Duplicate predicted span for {record.id}")
            spans.add((start, end, label))
        predictions[record.id] = spans
    return predictions


def counts(gold, predicted):
    tp = len(gold & predicted)
    return tp, len(predicted) - tp, len(gold) - tp


def metrics(tp, fp, fn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0}


def error_type(span, other):
    start, end, label = span
    if any(a == start and b == end and other_label != label for a, b, other_label in other):
        return "type_only"
    if any(max(start, a) < min(end, b) and other_label == label for a, b, other_label in other):
        return "boundary_only"
    if any(max(start, a) < min(end, b) for a, b, _ in other):
        return "boundary_and_type"
    return "no_overlap"


def summarize(records, predictions, dictionary):
    rows = []
    by_type = {label: np.zeros(3, dtype=int) for label in LABELS}
    gold_stats = {name: Counter() for name in ("seen", "unseen")}
    errors = {kind: Counter() for kind in ("false_negative", "false_positive")}
    examples = {kind: {} for kind in errors}
    for record in records:
        gold = record.gold
        predicted = predictions[record.id]
        rows.append(counts(gold, predicted))
        for label in LABELS:
            by_type[label] += counts({span for span in gold if span[2] == label},
                                     {span for span in predicted if span[2] == label})
        for start, end, label in gold:
            key = tuple(record.text[start:end].casefold().split(" "))
            group = "seen" if key in dictionary.entries else "unseen"
            gold_stats[group]["gold"] += 1
            gold_stats[group]["hit"] += (start, end, label) in predicted
        for kind, candidates, other in (("false_negative", gold - predicted, predicted),
                                        ("false_positive", predicted - gold, gold)):
            for span in sorted(candidates):
                reason = error_type(span, other)
                errors[kind][reason] += 1
                entry = examples[kind].setdefault(reason, [])
                if len(entry) < 4:
                    entry.append({"id": record.id, "span": span, "other": sorted(other)})
    rows = np.asarray(rows, dtype=int)
    slices = {group: {"gold": stats["gold"], "hit": stats["hit"],
                      "recall": stats["hit"] / stats["gold"] if stats["gold"] else 0.0}
              for group, stats in gold_stats.items()}
    return {"records": len(records), **metrics(*rows.sum(axis=0)),
            "per_label": {label: metrics(*values) for label, values in by_type.items()},
            "by_lexicon_coverage": slices, "error_counts": errors, "example_offsets": examples}, rows


def bootstrap(rows_by_method):
    """Paired sentence-level percentile intervals for F1 differences."""
    count = len(next(iter(rows_by_method.values())))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {method: [] for method in rows_by_method}
    for _ in range(0, BOOTSTRAP_REPEATS, 100):
        indices = rng.integers(0, count, size=(100, count))
        for method, rows in rows_by_method.items():
            tp, fp, fn = rows[indices].sum(axis=1).T
            denominator = 2 * tp + fp + fn
            samples[method].extend(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
    intervals = []
    for left, right in itertools.combinations(rows_by_method, 2):
        diff = np.array(samples[right]) - np.array(samples[left])
        low, high = np.percentile(diff, (2.5, 97.5))
        point = metrics(*rows_by_method[right].sum(axis=0))["f1"] - metrics(*rows_by_method[left].sum(axis=0))["f1"]
        intervals.append({"difference": f"{right} - {left}", "point": point, "low": float(low), "high": float(high)})
    return intervals


def draw_chart(summary):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
    fig.suptitle("Person and location extraction", fontsize=18, fontweight="bold")
    for i, (scope, datasets) in enumerate(summary.items()):
        for j, (dataset, results) in enumerate(datasets.items()):
            ax = axes[i, j]
            methods = list(results)
            bars = ax.bar([METHOD_NAMES[method] for method in methods],
                          [results[method]["f1"] for method in methods],
                          color=[COLORS[method] for method in methods], width=0.64)
            ax.bar_label(bars, labels=[f"{results[method]['f1']:.1%}" for method in methods], padding=3, fontsize=9)
            n = results[methods[0]]["records"]
            scope_label = "Full test split" if scope == "full" else "Matched API sample"
            ax.set_title(f"{DATASET_NAMES[dataset]}  ·  {scope_label}, n={n}", fontsize=11)
            ax.set_ylabel("Exact-span micro-F1")
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, .25, .50, .75, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "comparison.png", dpi=165)
    plt.close(fig)


def main():
    train, test, sources = load_data()
    if sources != read_json(RESULTS / "sources.json"):
        raise ValueError("Source hashes or counts do not match the recorded run")
    if {record.id for record in train} & {record.id for records in test.values() for record in records}:
        raise ValueError("Training and test IDs overlap")
    api_run = read_json(RESULTS / "api_run.json")
    core_run = read_json(RESULTS / "core_run.json")
    dictionary = Gazetteer(train)
    if len(dictionary.entries) != core_run["gazetteer"]["entries"]:
        raise ValueError("Rebuilt lexicon count does not match the recorded run")
    scores = {"full": {}, "matched": {}}
    comparisons = {"full": {}, "matched": {}}
    samples = {}
    train_texts = {record.text for record in train}
    deduplicated = {}
    for dataset, records in test.items():
        sample = api_sample(records)
        samples[dataset] = [record.id for record in sample]
        all_predictions = {method: read_predictions(method, dataset, records) for method in CORE_METHODS}
        all_predictions[API_METHOD] = read_predictions(API_METHOD, dataset, sample)
        if dictionary.predict(records) != list(all_predictions["gazetteer"].values()):
            raise ValueError(f"Rebuilt lexicon predictions disagree for {dataset}")
        if core_run["timing"]["gazetteer"][dataset]["records"] != len(records) or api_run["timing"][dataset]["records"] != len(sample):
            raise ValueError(f"Timing and prediction counts disagree for {dataset}")
        train_disjoint = [record for record in records if record.text not in train_texts]
        deduplicated[dataset] = {"excluded": len(records) - len(train_disjoint), "records": len(train_disjoint),
                                 "methods": {method: metrics(*np.asarray([
                                     counts(record.gold, all_predictions[method][record.id]) for record in train_disjoint
                                 ]).sum(axis=0)) for method in CORE_METHODS}}
        for scope, inputs, methods in (("full", records, CORE_METHODS), ("matched", sample, METHODS)):
            scope_results = {method: summarize(inputs, all_predictions[method], dictionary) for method in methods}
            scores[scope][dataset] = {method: result[0] for method, result in scope_results.items()}
            comparisons[scope][dataset] = bootstrap({method: result[1] for method, result in scope_results.items()})
    write_json(RESULTS / "scores.json", scores)
    write_json(RESULTS / "paired_bootstrap.json", {"seed": BOOTSTRAP_SEED, "repeats": BOOTSTRAP_REPEATS, "comparisons": comparisons})
    write_json(RESULTS / "api_sample_ids.json", samples)
    write_json(RESULTS / "without_train_text_matches.json", deduplicated)
    with (RESULTS / "scores.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ("scope", "dataset", "method", "records", "tp", "fp", "fn", "precision", "recall", "f1")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for scope, datasets in scores.items():
            for dataset, results in datasets.items():
                for method, result in results.items():
                    writer.writerow({"scope": scope, "dataset": dataset, "method": method, **result})
                    print(f"{scope:7} {dataset:5} {method:10} P={result['precision']:.2%} R={result['recall']:.2%} F1={result['f1']:.2%}")
    draw_chart(scores)


if __name__ == "__main__":
    main()
