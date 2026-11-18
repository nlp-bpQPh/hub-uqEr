# -*- coding: utf-8 -*-
"""
第 3 步：评估「SFT 前 vs SFT 后」。

指标：
  - val_loss / ppl / token_acc：teacher forcing 下，只在回答段计算
  - 生成质量：贪心解码，算完全匹配率（EM）和字符级 F1
  - 抽样对比：同一批问题，base 模型 vs SFT 模型的回答，落成 markdown 表格

用法：python src/evaluate.py [--num_samples 40]
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_jsonl(path, limit=0):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def load_val_tensors():
    return torch.load(config.PROCESSED_DIR / "val.pt", weights_only=False)


@torch.no_grad()
def val_metrics(model, tokenizer, samples, device, batch_size=8):
    """在验证集上算 masked loss / ppl / token 准确率。"""
    model.eval()
    pad_id = tokenizer.pad_token_id
    total_loss, total_tok, correct = 0.0, 0, 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        max_len = max(len(s["input_ids"]) for s in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, s in enumerate(chunk):
            n = len(s["input_ids"])
            input_ids[i, :n] = torch.tensor(s["input_ids"])
            labels[i, :n] = torch.tensor(s["labels"])
            attn[i, :n] = 1
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device), labels=labels.to(device))
        n = (labels != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tok += n
        pred = out.logits[:, :-1].argmax(-1).cpu()
        target = labels[:, 1:]
        mask = target != -100
        correct += (pred[mask] == target[mask]).sum().item()
    loss = total_loss / max(total_tok, 1)
    return {"val_loss": round(loss, 4), "ppl": round(math.exp(min(loss, 20)), 2),
            "token_acc": round(correct / max(total_tok, 1), 4)}


@torch.no_grad()
def generate(model, tokenizer, instructions, device, max_new_tokens=config.MAX_NEW_TOKENS, batch_size=8):
    """按 ChatML 模板生成回答（贪心解码，遇到 <|im_end|> 停）。"""
    model.eval()
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    im_end_id = tokenizer.convert_tokens_to_ids(config.IM_END)
    outputs = []
    for start in range(0, len(instructions), batch_size):
        chunk = instructions[start:start + batch_size]
        prompts = [config.build_prompt(x) for x in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=im_end_id,
        )
        for i, seq in enumerate(gen):
            new_tokens = seq[enc["input_ids"].shape[1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    tokenizer.padding_side = prev_side
    return outputs


def char_f1(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    p, r = overlap / len(pred), overlap / len(gold)
    return 0.0 if p + r == 0 else round(2 * p * r / (p + r), 4)


def score(gens, golds):
    em = sum(1 for g, t in zip(gens, golds) if g.strip() == t.strip()) / max(len(golds), 1)
    f1 = sum(char_f1(g, t) for g, t in zip(gens, golds)) / max(len(golds), 1)
    return {"exact_match": round(em, 4), "char_f1": round(f1, 4)}


def main():
    parser = argparse.ArgumentParser(description="对比 SFT 前后的模型")
    parser.add_argument("--model", default=str(config.MODEL_DIR))
    parser.add_argument("--adapter", default=str(config.ADAPTER_DIR))
    parser.add_argument("--num_samples", type=int, default=40, help="生成对比用的验证集样本数")
    parser.add_argument("--show", type=int, default=8, help="写进对比表格的条数")
    args = parser.parse_args()

    if not (Path(args.adapter) / "adapter_config.json").exists():
        raise SystemExit(f"找不到 LoRA 适配器：{args.adapter}\n请先运行 python src/train_sft.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    val_samples = load_val_tensors()
    val_rows = load_jsonl(config.DATA_DIR / "sft_val.jsonl", args.num_samples)
    golds = [r["output"] for r in val_rows]
    print(f"设备 {device} | 验证集 {len(val_samples)} 条 | 生成对比 {len(val_rows)} 条")

    # ---------- 1) base 模型 ----------
    print("\n[1/2] 评测基座模型（SFT 前）…")
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    base_metrics = val_metrics(base, tokenizer, val_samples, device)
    base_gens = generate(base, tokenizer, [r["instruction"] for r in val_rows], device)
    base_metrics.update(score(base_gens, golds))
    print("   ", base_metrics)
    del base
    torch.cuda.empty_cache()

    # ---------- 2) SFT 模型 ----------
    print("\n[2/2] 评测 SFT 模型（基座 + LoRA）…")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    sft_metrics = val_metrics(model, tokenizer, val_samples, device)
    sft_gens = generate(model, tokenizer, [r["instruction"] for r in val_rows], device)
    sft_metrics.update(score(sft_gens, golds))
    print("   ", sft_metrics)

    # ---------- 结果 ----------
    result = {"base": base_metrics, "sft": sft_metrics,
              "n_val": len(val_samples), "n_gen": len(val_rows)}
    (config.OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"{'指标':<14}{'SFT 前':>14}{'SFT 后':>14}")
    print("-" * 78)
    for k, name in [("val_loss", "验证集 loss"), ("ppl", "困惑度 PPL"), ("token_acc", "回答 token 准确率"),
                    ("exact_match", "完全匹配率 EM"), ("char_f1", "字符级 F1")]:
        print(f"{name:<14}{base_metrics[k]:>14}{sft_metrics[k]:>14}")
    print("=" * 78)

    # 抽样对比表
    lines = ["# SFT 前后对比（验证集抽样）", "",
             "| # | 指令 | 标准答案 | SFT 前（基座模型） | SFT 后（LoRA） |",
             "|---|---|---|---|---|"]
    for i in range(min(args.show, len(golds))):
        fmt = lambda s: s.replace("\n", " ").replace("|", "\\|")[:80]
        lines.append(f"| {i + 1} | {fmt(val_rows[i]['instruction'])} | {fmt(golds[i])} | "
                     f"{fmt(base_gens[i])} | {fmt(sft_gens[i])} |")
    (config.OUTPUT_DIR / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    all_rows = [{"instruction": r["instruction"], "gold": g, "base": b, "sft": s}
                for r, g, b, s in zip(val_rows, golds, base_gens, sft_gens)]
    (config.OUTPUT_DIR / "generations.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[写出] {config.OUTPUT_DIR / 'metrics.json'}")
    print(f"[写出] {config.OUTPUT_DIR / 'compare.md'}")
    print(f"[写出] {config.OUTPUT_DIR / 'generations.json'}")
    print("\n下一步：python src/chat.py   # 和 SFT 后的模型对话")


if __name__ == "__main__":
    main()
