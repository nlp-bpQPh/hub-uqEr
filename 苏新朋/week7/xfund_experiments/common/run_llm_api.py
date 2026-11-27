from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from metrics import save_report
from xfund_data import load_xfund


SYSTEM = """你是中文表单字段抽取器。对输入的 OCR 文本块逐块分类，标签只能是 header、question、answer、other。不要改写文本或遗漏 block id。question-answer 关系按 linking 输出。只返回 JSON：{\"labels\":[{\"id\":1,\"label\":\"question\"}],\"links\":[[1,2]]}。"""


def compact(doc):
    return [{"id": b.get("id"), "text": b.get("text", ""), "box": b.get("box")}
            for b in doc.get("document", [])]


def example_payload(doc):
    labels = [{"id": b.get("id"), "label": b.get("label", "other")}
              for b in doc.get("document", [])]
    links = []
    for b in doc.get("document", []):
        if b.get("label") == "question":
            links.extend(b.get("linking", []))
    return {"blocks": compact(doc), "gold": {"labels": labels, "links": links}}


def parse_reply(content: str, blocks: list[dict]) -> list[str]:
    parsed = json.loads(content)
    by_id = {str(x.get("id")): x.get("label", "other") for x in parsed.get("labels", [])}
    allowed = {"header", "question", "answer", "other"}
    return [by_id.get(str(b.get("id")), "other") if by_id.get(str(b.get("id")), "other") in allowed else "other"
            for b in blocks]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero", "few"], required=True)
    p.add_argument("--train", default=str(ROOT / "data/source/zh.train.json"))
    p.add_argument("--val", default=str(ROOT / "data/source/zh.val.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--examples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set LLM_API_KEY (or OPENAI_API_KEY) to run this API method; no credential is stored in the project.")
    if not os.environ.get("LLM_MODEL"):
        raise SystemExit("Set LLM_MODEL; optionally set LLM_BASE_URL for an OpenAI-compatible endpoint.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("Install the openai Python package to run this API method.") from e

    client = OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL"))
    train_docs = load_xfund(args.train)
    val_docs = load_xfund(args.val)
    examples = []
    if args.mode == "few":
        candidates = [d for d in train_docs if 8 <= len(d.get("document", [])) <= 35]
        random.Random(args.seed).shuffle(candidates)
        examples = candidates[:args.examples]
    preds = {}
    raw_path = Path(args.out) / "raw_responses.jsonl"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw_file:
        for index, doc in enumerate(val_docs, 1):
            messages = [{"role": "system", "content": SYSTEM}]
            for ex in examples:
                messages.append({"role": "user", "content": json.dumps(example_payload(ex), ensure_ascii=False)})
                messages.append({"role": "assistant", "content": json.dumps(
                    {"labels": [{"id": b.get("id"), "label": b.get("label", "other")} for b in ex.get("document", [])],
                     "links": [pair for b in ex.get("document", []) for pair in b.get("linking", [])]},
                    ensure_ascii=False)})
            messages.append({"role": "user", "content": json.dumps({"blocks": compact(doc)}, ensure_ascii=False)})
            response = client.chat.completions.create(
                model=os.environ["LLM_MODEL"], messages=messages, temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            try:
                labels = parse_reply(content, compact(doc))
            except Exception:
                labels = ["other"] * len(doc.get("document", []))
            preds[str(doc.get("id", ""))] = labels
            raw_file.write(json.dumps({"doc_id": doc.get("id"), "raw": content}, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(f"{index}/{len(val_docs)} {doc.get('id')}")

    save_report(args.out, val_docs, preds, {
        "name": f"api_{args.mode}_shot",
        "model": os.environ["LLM_MODEL"],
        "base_url": os.environ.get("LLM_BASE_URL", "provider default"),
        "temperature": 0,
        "few_shot_examples": len(examples),
        "official_split": "zh.val",
        "note": "Per-block labels scored. Key-answer relation links are requested but not evaluated.",
    })


if __name__ == "__main__":
    main()
