# -*- coding: utf-8 -*-
r"""用法：复制这一行就能跑（PowerShell）
D:\Miniconda3\envs\py312\python.exe "D:\AI Program\hub-uqEr\林赫洋\week07\作业\实验.py"
末尾可加 --quick（默认）或 --full。输出和数据路径均相对于本脚本。
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

try:
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, BertTokenizerFast, logging
except ImportError as exc:
    raise SystemExit(r"请用 D:\Miniconda3\envs\py312\python.exe 实验.py" + f"\n缺少依赖：{exc}")

ROOT = Path(__file__).resolve().parent
SEED = 42
LIMIT = 190  # 三路均看原文前 190 字，完整金标仍参与评测。
BATCH = 16


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def read_split(split):
    path = ROOT / "data" / f"{split}_extract.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def subset(rows, count):
    selected, _ = train_test_split(rows, train_size=count, random_state=SEED,
                                   stratify=[row["event"]["type"] for row in rows])
    return selected


def make_labels(schema):
    names = ["触发/" + kind for kind in sorted(schema)]
    names += sorted({"论元/" + role for roles in schema.values() for role in roles})
    return ["O"] + [prefix + name for name in names for prefix in ("B-", "I-")]


def encode(row, tokenizer, label_ids, training):
    # 两个检查点都是中文 BERT 字表。显式每个字符查一个 ID，空格也占一格。
    text = row["text"][:LIMIT]
    tokens = tokenizer.convert_tokens_to_ids([ch.lower() for ch in text])
    ids = [tokenizer.cls_token_id, *tokens, tokenizer.sep_token_id]
    tags = [0] * len(text)
    if training:
        event = row["event"]
        spans = [("触发/" + event["type"], event["trigger"])]
        spans += [("论元/" + a["role"], a) for a in sorted(
            event["args"], key=lambda a: -len(a["text"]))]
        for name, span in spans:
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(text) or any(tags[start:end]):
                continue
            tags[start:end] = [label_ids["B-" + name]] + [label_ids["I-" + name]] * (end-start-1)
    return torch.tensor(ids), torch.tensor([-100, *tags, -100])


def batchify(examples):
    ids, tags = zip(*examples)
    return (nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0),
            nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100))


def decode(text, tags, confidence):
    # I 出现在开头或换了角色时，按新片段处理。
    spans = []
    for pos, tag in enumerate(tags):
        if tag == "O":
            continue
        name = tag[2:]
        if tag.startswith("I-") and spans and spans[-1][0] == name and spans[-1][2] == pos:
            spans[-1][2] = pos + 1
        else:
            spans.append([name, pos, pos + 1])
    triggers, args = [], []
    for name, start, end in spans:
        span = dict(text=text[start:end], start=start, end=end)
        family, role = name.split("/", 1)
        if family == "触发":
            triggers.append((sum(confidence[start:end]) / (end-start), role, span))
        else:
            args.append(dict(role=role, **span))
    if not triggers:
        return None
    _, kind, trigger = max(triggers, key=lambda item: item[0])
    return dict(type=kind, trigger=trigger, args=args)


def neural(model_name, train, dev, labels, epochs, device):
    seed_everything()
    model_dir = ROOT / "models" / model_name
    if not model_dir.is_dir():
        model_dir = ROOT.parents[1] / "week06" / "作业" / "models" / model_name
    if not model_dir.is_dir():
        raise SystemExit(f"未找到本地模型，请检查 {model_dir}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, num_labels=len(labels), local_files_only=True).to(device)
    label_ids = {name: i for i, name in enumerate(labels)}
    examples = [encode(row, tokenizer, label_ids, True) for row in train]
    loader = DataLoader(examples, batch_size=BATCH, shuffle=True, collate_fn=batchify,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW([{"params": model.base_model.parameters(), "lr": 5e-5},
                                   {"params": model.classifier.parameters(), "lr": 8e-4}])
    weights = torch.ones(len(labels), device=device)
    weights[0] = 0.12  # 减轻大量背景字对损失的支配。
    criterion = nn.CrossEntropyLoss(weight=weights)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(epochs):
        model.train()
        losses = []
        for ids, tags in loader:
            ids, tags = ids.to(device), tags.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(ids, attention_mask=ids.ne(0), return_dict=True).logits
                loss = criterion(logits.flatten(0, 1), tags.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"{model_name} 第 {epoch+1}/{epochs} 轮，平均 batch loss {np.mean(losses):.4f}", flush=True)
    model.eval()
    prediction_start = perf_counter()
    examples = [encode(row, tokenizer, label_ids, False) for row in dev]
    predictions = []
    with torch.inference_mode():
        for ids, _ in DataLoader(examples, batch_size=BATCH, collate_fn=batchify):
            ids = ids.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(ids, attention_mask=ids.ne(0), return_dict=True).logits
            probs, indices = logits.float().softmax(-1).max(-1)
            for seq, conf in zip(indices.tolist(), probs.tolist()):
                text = dev[len(predictions)]["text"][:LIMIT]
                tags = [labels[i] for i in seq[1:1+len(text)]]
                predictions.append(decode(text, tags, conf[1:1+len(text)]))
    return predictions, sum(p.numel() for p in model.parameters()), perf_counter() - prediction_start



# 同一句可能有多个关键词，先看原文中最早出现的；同位置优先长词。
TRIGGERS = {
    "解除质押": r"解除质押|解押", "质押": r"质押",
    "股份回购": r"回购", "股东增持": r"增持", "股东减持": r"减持",
    "中标": r"中标", "亏损": r"亏损|亏蚀", "企业收购": r"收购|并购",
    "高管变动": r"辞职|辞任|离任|聘任|辞去|任命",
    "企业破产": r"破产", "被约谈": r"约谈", "企业融资": r"融资|领投",
    "公司上市": r"上市(?!公司)|挂牌|IPO",
}
NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
FIELD_PATTERNS = [
    (r"(?:\d{4}年)?\d{1,2}月\d{1,2}[日号](?:上午|下午|晚间)?",
     ("披露时间", "披露日期")),
    (NUMBER + r"(?:万亿|亿|万|千)?(?:港元|美元|欧元|元)",
     ("交易金额", "融资金额", "中标金额", "净亏损", "债务规模", "募资金额")),
    (NUMBER + r"(?:亿|万|千)?(?=股(?!东))",
     ("回购股份数量", "交易股票/股份数量", "质押股票/股份数量")),
]


def rule_predict(text, schema):
    """返回一句的单个事件。数值规则只填写该事件类型允许的角色。"""
    hits = []
    for event_type, pattern in TRIGGERS.items():
        hit = re.search(pattern, text)
        if hit:
            hits.append((hit.start(), -len(hit.group()), event_type, hit))
    if not hits:
        return None
    _, _, event_type, trigger = min(hits)
    args = []
    for pattern, roles in FIELD_PATTERNS:
        role = next((r for r in roles if r in schema[event_type]), None)
        match = re.search(pattern, text)
        if role and match:
            args.append({"role": role, "text": match.group(),
                         "start": match.start(), "end": match.end()})
    return {"type": event_type,
            "trigger": {"text": trigger.group(), "start": trigger.start(), "end": trigger.end()},
            "args": args}


def scores(dev, predictions, schema):
    """一句一个金标事件。先对齐事件类型，再按文本给触发词和论元计分。"""

    def counts(items):
        return Counter((item["role"].strip(), item["text"].strip()) for item in items)

    def f1(hits, predicted, expected):
        return 2 * hits / (predicted + expected) if predicted + expected else 0.0

    gold_types, pred_types, correct_types = Counter(), Counter(), Counter()
    trigger_hits = trigger_pred = trigger_gold = 0
    arg_hits = arg_pred = arg_gold = 0
    event_hits = event_pred = 0
    for row, guess in zip(dev, predictions, strict=True):
        gold = row["event"]
        kind = gold["type"]
        gold_types[kind] += 1
        gold_trigger = gold["trigger"]["text"].strip()
        gold_args = counts(gold["args"])
        trigger_gold += bool(gold_trigger)
        arg_gold += gold_args.total()
        if not guess:
            continue
        pred_kind = guess["type"]
        pred_types[pred_kind] += 1
        pred_trigger = (guess.get("trigger") or {}).get("text", "").strip()
        pred_args = counts(guess.get("args", []))
        trigger_pred += bool(pred_trigger)
        arg_pred += pred_args.total()
        event_pred += 1
        if kind == pred_kind:
            correct_types[kind] += 1
            trigger_hits += bool(pred_trigger) and pred_trigger == gold_trigger
            arg_hits += (gold_args & pred_args).total()
            event_hits += pred_trigger == gold_trigger and pred_args == gold_args
    macro = sum(f1(correct_types[t], pred_types[t], gold_types[t]) for t in schema) / len(schema)
    return (correct_types.total() / len(dev), macro, f1(trigger_hits, trigger_pred, trigger_gold),
            f1(arg_hits, arg_pred, arg_gold), f1(event_hits, event_pred, len(dev)))


def summarize(event):
    if event is None:
        return "未抽出事件"
    fields = "，".join(a["role"] + "=" + a["text"] for a in event["args"])
    return f"{event['type']} | {event['trigger']['text']} | {fields or '无论元'}"


def main():
    parser = argparse.ArgumentParser(description="DuEE-Fin：三种方法在同一批公告上抽取事件")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true", help="固定分层子集，默认")
    modes.add_argument("--full", action="store_true", help="全部去泄漏训练数据和完整验证集")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    seed_everything()
    logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = json.loads((ROOT / "data/event_schema.json").read_text(encoding="utf-8"))
    schema.pop("__enums__", None)
    labels = make_labels(schema)
    train, dev = read_split("train"), read_split("dev")
    held_out_text = {r["text"] for r in dev}
    train = [r for r in train if r["text"] not in held_out_text]
    if not args.full:
        train, dev = subset(train, 1040), subset(dev, 208)
    epochs = 3 if args.full else 2
    print(f"{mode} | {device} | train={len(train)} dev={len(dev)} | epochs={epochs} seed={SEED}", flush=True)
    results, guesses = [], {}
    for method in ("词典+正则", "roberta-l4-h256", "bert-base-chinese"):
        start = perf_counter()
        if method == "词典+正则":
            predictions = [rule_predict(row["text"][:LIMIT], schema) for row in dev]
            prediction_time, params = perf_counter()-start, 0
        else:
            predictions, params, prediction_time = neural(method, train, dev, labels, epochs, device)
        elapsed = perf_counter()-start
        metrics = scores(dev, predictions, schema)
        results.append([method, params, *metrics, prediction_time*100/len(dev), elapsed])
        guesses[method] = predictions

    # 优先展示三路均认对类型、有数字且有多个字段的一条短句，便于逐字核对。
    def example_rank(i):
        row = dev[i]
        examples = [pred[i] for pred in guesses.values()]
        recognized = sum(bool(e) and e["type"] == row["event"]["type"] for e in examples)
        variants = len({summarize(e) for e in examples})
        fields = row["event"]["args"]
        has_amount = any("金额" in a["role"] for a in fields)
        return (len(row["text"]) <= LIMIT, recognized, variants, has_amount,
                len(fields) >= 2, -len(row["text"]))

    index = max(range(len(dev)), key=example_rank)
    sample = dev[index]
    print("\n同句对照：" + sample["text"])
    print("标准答案：" + summarize(sample["event"]))
    for method, predictions in guesses.items():
        print(method + "：" + summarize(predictions[index]))

    header = ["方法", "参数量", "准确率", "Macro-F1", "触发词F1", "论元F1", "整事件F1", "推理秒/百句", "总耗时(s)"]
    print("\n" + "\t".join(header))
    for method, params, *metrics, latency, elapsed in results:
        print("\t".join([method, f"{params:,}", *(f"{m:.4f}" for m in metrics),
                         f"{latency:.3f}", f"{elapsed:.3f}"]))
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    report = {"mode": mode, "seed": SEED, "train": len(train), "dev": len(dev), "epochs": epochs,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
              "table": [dict(zip(header, row)) for row in results],
              "example": {**sample, "predictions": {m: rows[index] for m, rows in guesses.items()}}}
    (out / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
