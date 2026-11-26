from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from xfund_data import block_label, load_xfund
from metrics import save_report


KEY_PATTERN = re.compile(r"(名称|姓名|编号|代码|日期|时间|金额|地址|电话|账号|类别|类型|单位|联系人|证件|标准号|标准名称)")
VALUE_PATTERN = re.compile(r"(\d{4}[年./-]\d{1,2}|\d+[年月日号%％]|GB\s*/?T?\s*[-\w.]+|\d{6,}|[A-Z]{2,}\d+)", re.I)


def predict(doc: dict) -> list[str]:
    blocks = doc.get("document", [])
    max_y = max((b.get("box", [0, 0, 0, 0])[3] for b in blocks), default=1) or 1
    questions = []
    labels = ["other"] * len(blocks)
    for i, block in enumerate(blocks):
        text = str(block.get("text", "")).strip()
        box = block.get("box", [0, 0, 0, 0])
        if not text:
            continue
        if len(text) <= 35 and (text.endswith((":", "：", "?", "？")) or KEY_PATTERN.search(text)):
            labels[i] = "question"
            questions.append((i, box, text))
        elif box[1] <= max_y * 0.12 and len(text) >= 5:
            labels[i] = "header"

    for i, block in enumerate(blocks):
        if labels[i] != "other":
            continue
        text = str(block.get("text", "")).strip()
        box = block.get("box", [0, 0, 0, 0])
        if not text or len(text) > 80:
            continue
        # Approximate key-value alignment: a compact value near a question,
        # or a date/number/code-shaped value.
        near_question = False
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        for _, qbox, _ in questions:
            qcx = (qbox[0] + qbox[2]) / 2
            qcy = (qbox[1] + qbox[3]) / 2
            h = max(qbox[3] - qbox[1], box[3] - box[1], 1)
            same_row_right = cx >= qcx and abs(cy - qcy) <= 1.5 * h
            below_aligned = cy > qcy and abs(cx - qcx) <= 4 * h and abs(cy - qcy) <= 4 * h
            if same_row_right or below_aligned:
                near_question = True
                break
        if near_question or VALUE_PATTERN.search(text):
            labels[i] = "answer"
    return labels


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--val", default=str(ROOT / "data/source/zh.val.json"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs"))
    args = p.parse_args()
    docs = load_xfund(args.val)
    preds = {str(doc["id"]): predict(doc) for doc in docs}
    report = save_report(args.out, docs, preds, {
        "name": "rules_regex_geometry",
        "description": "Keyword/format rules plus OCR-box proximity heuristics; no training.",
        "split": "official zh.val",
    })
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
