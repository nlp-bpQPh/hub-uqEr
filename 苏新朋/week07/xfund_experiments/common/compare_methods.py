from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "01_rules",
    "02_bilstm_crf",
    "03_bert_linear",
    "04_bert_crf",
    "05_bert_crf_hard_decode",
    "06_llm_api_zero_shot",
    "07_llm_api_few_shot",
]


def main():
    rows = []
    for folder in METHODS:
        path = ROOT / "methods" / folder / "outputs" / "metrics.json"
        if not path.exists():
            rows.append((folder, None))
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        m = report["metrics"]
        rows.append((folder, {
            "blocks": m["blocks"],
            "block_accuracy": m["block_accuracy"],
            "macro_f1": m["macro_f1_all_four_classes"],
            "entity_precision": m["entity_micro"]["precision"],
            "entity_recall": m["entity_micro"]["recall"],
            "entity_f1": m["entity_micro"]["f1"],
            "answer_f1": m["per_class"]["answer"]["f1"],
            "question_f1": m["per_class"]["question"]["f1"],
            "header_f1": m["per_class"]["header"]["f1"],
            "other_f1": m["per_class"]["other"]["f1"],
        }))

    md = [
        "# Method comparison (official XFUND zh.val)",
        "",
        "All scores classify the same OCR blocks. Entity micro-F1 includes header/question/answer; other is excluded from this metric. Question-answer links are not scored.",
        "",
        "| Method | Blocks | Accuracy | 4-class macro-F1 | Entity micro P | Entity micro R | Entity micro F1 | Header F1 | Question F1 | Answer F1 | Other F1 | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, v in rows:
        if v is None:
            md.append(f"| {method} | - | - | - | - | - | - | - | - | - | - | not run |")
        else:
            md.append("| " + " | ".join([
                method, str(v["blocks"]), f'{v["block_accuracy"]:.4f}', f'{v["macro_f1"]:.4f}',
                f'{v["entity_precision"]:.4f}', f'{v["entity_recall"]:.4f}', f'{v["entity_f1"]:.4f}',
                f'{v["header_f1"]:.4f}', f'{v["question_f1"]:.4f}', f'{v["answer_f1"]:.4f}',
                f'{v["other_f1"]:.4f}', "completed",
            ]) + " |")
    out = ROOT / "reports/method_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
