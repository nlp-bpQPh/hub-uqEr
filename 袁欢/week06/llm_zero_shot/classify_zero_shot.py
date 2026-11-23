"""
方法四：大模型零样本（L3，不训练）

教学重点：
  1. 零样本 = 把分类当成「生成」来做：写一句 prompt，让模型直接吐类别名。
     不用数据、不用训练，冷启动最快——这是它的唯一优势。
  2. 但 15 类中文新闻标题上它**基本不可用**：实测 Macro-F1 0.098，
     相对随机基线（1/15=0.067）只高一点。原因是**预测塌缩**：
     15 类里只输出过 5 类，「文化」一个类就占了七成。
  3. Qwen3 必须传 enable_thinking=False。不传的话模型会先吐一段
     <think>...</think> 推理链，把「只输出类别名」的格式彻底打乱。
     这是 Qwen2 → Qwen3 迁移时最容易踩的坑。

用法：
  python llm_zero_shot/classify_zero_shot.py              # 默认 200 条
  python llm_zero_shot/classify_zero_shot.py --n 500
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val
from common.labels import LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent

# Prompt 三件套：角色 + 类别定义 + 输出约束
# 类别名不在 prompt 里列全，模型就会自己发明标签；phishing 与 malware 是 LLM 最易混
# 的两个类，定义必须写进 prompt（英文 URL 用英文 prompt + 英文标签名）
SYSTEM = ("You are a URL security classifier. Classify the given URL into exactly one "
          "of these categories:\n"
          "- benign: a normal, legitimate website URL\n"
          "- phishing: a fake site impersonating a legitimate one, aimed at stealing credentials\n"
          "- malware: a URL that serves malicious software or leads to a malicious download\n"
          "- defacement: a hacked/vandalized legitimate website\n"
          "Answer with only the category name, nothing else.")


def parse_label(text: str) -> int:
    """
    从生成文本里抽出类别名；抽不到返回 -1（记为解析失败）。

    为什么「精确等值优先、边界匹配兜底」而不是简单包含：
      英文标签互不为子串（benign/phishing/malware/defacement 无嵌套），但模型可能输出
      带前缀/解释的句子（"The category is: phishing."）；严格相等会误判为失败，
      裸包含又会被 non-benign 这类词误伤——先等值再边界，两头都稳。
      边界用 (?<![a-z-])：连字符不算边界内字符，non-benign 不会被 benign 命中。
    """
    t = text.strip().lower()
    for i, name in enumerate(LABELS):
        if t == name:
            return i
    hits = [n for n in LABELS if re.search(rf"(?<![a-z-]){re.escape(n)}(?![a-z])", t)]
    if not hits:
        return -1
    return LABELS.index(max(hits, key=len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default="outputs/zero_shot_pred.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    model = AutoModelForCausalLM.from_pretrained(
        str(qwen3_path()), dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    samples, meta = load_clean_val(args.n)
    preds, raws, t0 = [], [], time.time()
    for i, d in enumerate(samples):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": d["sentence"]}]
        # enable_thinking=False 必须在模板阶段传：Qwen3 会在 assistant 前插入空 think 块
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(parse_label(raw))
        if i % 50 == 0:
            print(f"  {i}/{len(samples)} {time.time()-t0:.0f}s", flush=True)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report["method"] = "qwen3_zero_shot"
    report["sec_per_item"] = round((time.time() - t0) / len(samples), 2)
    report["pred_distribution"] = pred_distribution(preds)   # 塌缩的证据

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条 | {report['sec_per_item']}s/条")
    print(to_markdown_table({"Qwen3 zero-shot": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
