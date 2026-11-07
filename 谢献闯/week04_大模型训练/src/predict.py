# -*- coding: utf-8 -*-
"""
第四步：用训练好的模型识别语言。

用法：
    python src/predict.py "今天天气很好，我们去公园散步吧"
    python src/predict.py "Bonjour, comment allez-vous ?" "Hola, ¿cómo estás?"
    python src/predict.py --file texts.txt          # 每行一条文本
    python src/predict.py --demo                    # 12 种语言各来一句
    python src/predict.py --interactive             # 交互式，输入一行识别一行
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根目录（config.py）
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))  # 同目录（train.py）
import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEMO_TEXTS = [
    "今天天气很好，我们一起去公园散步吧。",
    "The weather is lovely today, so let's take a walk in the park.",
    "今日はいい天気ですね、一緒に公園を散歩しましょう。",
    "오늘 날씨가 정말 좋네요. 같이 공원에 산책하러 갑시다.",
    "Il fait très beau aujourd'hui, allons nous promener dans le parc.",
    "Das Wetter ist heute wunderschön, lass uns im Park spazieren gehen.",
    "Hoy hace muy buen tiempo, vamos a pasear por el parque.",
    "Сегодня прекрасная погода, давайте прогуляемся по парку.",
    "الطقس جميل اليوم، دعنا نتمشى في الحديقة.",
    "O tempo está ótimo hoje, vamos passear no parque.",
    "Oggi il tempo è bellissimo, andiamo a fare una passeggiata nel parco.",
    "Bugün hava çok güzel, hadi parkta yürüyüşe çıkalım.",
    # 混一句"没有明显特征"的，看看模型会不会乱猜
    "OK, 1234567890 !!!",
]


def load(model_dir: Path, device: torch.device):
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"找不到模型：{model_dir}\n请先运行 python src/train.py 完成训练。")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def predict(texts, tokenizer, model, device, max_len=config.MAX_LEN, topk=3, batch_size=32):
    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        for text, prob in zip(chunk, probs):
            values, indices = prob.topk(min(topk, prob.numel()))
            results.append([(config.LABELS[i], config.LABEL_NAMES_ZH[config.LABELS[i]], float(v))
                            for v, i in zip(values, indices)])
    return results


def show(text, top):
    code, zh, prob = top[0]
    snippet = text if len(text) <= 70 else text[:70] + "…"
    print(f"\n文本 : {snippet}")
    print(f"识别 : {zh} ({code})   置信度 {prob:.2%}")
    for rank, (c, z, p) in enumerate(top, 1):
        bar = "█" * max(1, int(p * 30))
        print(f"   {rank}. {z:<5} {p:7.2%}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="12 国语言识别推理")
    parser.add_argument("texts", nargs="*", help="要识别的文本，可传多条")
    parser.add_argument("--model_dir", default=str(config.SAVE_DIR))
    parser.add_argument("--file", help="文本文件，每行一条")
    parser.add_argument("--demo", action="store_true", help="跑一遍内置的 12 国语言样例")
    parser.add_argument("--interactive", action="store_true", help="交互式输入")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load(Path(args.model_dir), device)
    print(f"模型：{args.model_dir} | 设备：{device}")

    if args.interactive:
        print("输入文本回车识别，输入 q 退出。")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() in {"q", "quit", "exit"}:
                break
            show(line, predict([line], tokenizer, model, device, topk=args.topk)[0])
        return

    texts = list(args.texts)
    if args.file:
        texts += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.demo or not texts:
        texts = DEMO_TEXTS

    for text, top in zip(texts, predict(texts, tokenizer, model, device, topk=args.topk)):
        show(text, top)


if __name__ == "__main__":
    main()
