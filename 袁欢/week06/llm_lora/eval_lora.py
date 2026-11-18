"""
方法五（推理）：加载 LoRA adapter，在干净验证集上评测

教学重点：
  1. 推理必须和训练用**同一个 chat 模板 + 同一个 enable_thinking=False**，
     否则格式错位，准确率会莫名掉一半。
  2. 对比 three-way：zero-shot 0.098 → LoRA SFT 0.484 → BERT 微调 0.534。
     SFT 把塌缩修好了，但在「封闭标签 + 短文本」上仍不如判别式——
     这就是选型的答案：不是谁更强，是谁更合适。
  3. 解析失败率要单独统计（-1 的条数）。生成式模型总会有不听话的时候，
     这个数字直接决定你要不要加约束解码。

用法：
  python llm_lora/eval_lora.py                       # 默认 outputs/lora_adapter
  python llm_lora/eval_lora.py --n 500 --adapter outputs/lora_adapter
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent

# 与 train_lora.py 必须逐字符一致：推理与训练模板不同，格式错位直接掉分
SYSTEM = ("You are a URL security classifier. Classify the given URL into exactly one "
          "of these categories:\n"
          "- benign: a normal, legitimate website URL\n"
          "- phishing: a fake site impersonating a legitimate one, aimed at stealing credentials\n"
          "- malware: a URL that serves malicious software or leads to a malicious download\n"
          "- defacement: a hacked/vandalized legitimate website\n"
          "Answer with only the category name, nothing else.")


def parse_label(text: str) -> int:
    """精确等值优先，边界匹配兜底（与 zero-shot / api 同一实现）；
    边界用 (?<![a-z-])：连字符不算边界，non-benign 不会被 benign 命中"""
    t = text.strip().lower()
    for i, name in enumerate(LABELS):
        if t == name:
            return i
    hits = [n for n in LABELS if re.search(rf"(?<![a-z-]){re.escape(n)}(?![a-z])", t)]
    return LABELS.index(max(hits, key=len)) if hits else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(CODE_ROOT / "outputs" / "lora_adapter"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default="outputs/lora_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    ckpt_dir = Path(args.adapter)
    if (ckpt_dir / "adapter_config.json").exists():
        # LoRA：先载基座，再挂 adapter
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(str(qwen3_path()), dtype=torch.float16 if device == "cuda" else torch.float32)
        model = PeftModel.from_pretrained(base, str(ckpt_dir)).to(device).eval()
        kind = "lora"
    else:
        # 全量微调：目录里就是完整权重，直接 from_pretrained（约 1.2 GB）
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        kind = "full_ft"
    print(f"[eval] 加载方式：{kind}（{ckpt_dir}）", flush=True)

    samples, meta = load_clean_val(args.n)
    preds, raws = [], []
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(parse_label(raw))
        if i % 50 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report["method"] = "qwen3_lora_sft"
    report["pred_distribution"] = pred_distribution(preds)

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集）")
    print(to_markdown_table({f"Qwen3 + {kind}": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()


