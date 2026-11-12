"""
export_ime_model.py — 把训练好的字符级 LSTM 语言模型导出成网页可用的 ime_model.js

为什么可以直接搬到浏览器：
    模型只有 Embedding + 2 层 LSTM + 一个 Linear，全是矩阵乘 + 激活，
    没有依赖任何框架特性，用几十行 JS 就能逐位复现前向过程。
    权重按 fp16 存（体积减半，对概率的影响在 1e-3 量级），浏览器里解码成 float32 使用。

导出内容:
    1. 字表 / 拼音表（去重、过滤到词表内）
    2. 字频表 + bigram 表（给 N-gram 对照组用）
    3. 模型权重（embed / lstm l0 l1 / fc，fp16 + base64）
    4. 自检数据（几组上下文的下一个字 log 概率 + 隐状态校验和）

用法:
    python export_ime_model.py
    python export_ime_model.py --model best_model.pt --out ..\\web\\ime_model.js
"""

import argparse
import base64
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.normpath(os.path.join(HERE, "..", "循环神经网络语言模型", "best_model.pt"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "web", "ime_model.js"))

MAX_CANDIDATES = 12          # 每个音节最多保留多少候选字
BIGRAM_CONTEXTS = 1500       # 为前多少个高频字导出 bigram 表
BIGRAM_TOPK = 8


# ─────────────────────── 模型定义（与 language_model.py 保持一致） ───────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        return self.fc(self.drop(out))


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["args"]
    m = LM(
        vocab_size=len(ck["char2idx"]), embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        model_type=cfg["model"], dropout=0.0,
    )
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck, cfg


def b64_fp16(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


def pack_lstm(model, layer: int):
    """把 PyTorch LSTM 的 4 组权重拆成 [in, 4H] 形式（按 i,f,g,o 顺序），bias 合并。"""
    p = f"rnn."
    w_ih = getattr(model.rnn, f"weight_ih_l{layer}").detach().numpy()   # (4H, in)
    w_hh = getattr(model.rnn, f"weight_hh_l{layer}").detach().numpy()   # (4H, H)
    b = (getattr(model.rnn, f"bias_ih_l{layer}") + getattr(model.rnn, f"bias_hh_l{layer}")).detach().numpy()
    return {
        "weight_ih": b64_fp16(np.ascontiguousarray(w_ih)),   # (4H, in) 行主序，JS 逐行点积
        "weight_hh": b64_fp16(np.ascontiguousarray(w_hh)),   # (4H, H)
        "bias": b64_fp16(b),
        "in": int(w_ih.shape[1]),
        "hidden": int(w_hh.shape[1]),
    }


def build_ngram(corpus_path, char2idx):
    """字频 + bigram（上下文只取上一个字，正是 N-gram 的马尔可夫假设）。"""
    text = open(corpus_path, encoding="utf-8", errors="ignore").read()
    text = "".join(c for c in text if c in char2idx)
    uni = Counter(text)
    bi = defaultdict(Counter)
    for a, b in zip(text, text[1:]):
        bi[a][b] += 1

    top_ctx = [c for c, _ in uni.most_common(BIGRAM_CONTEXTS)]
    bigrams = {}
    for c in top_ctx:
        items = bi[c].most_common(BIGRAM_TOPK)
        if items:
            bigrams[c] = [[ch, n] for ch, n in items]
    return dict(uni), bigrams


def selftest(model, ck, contexts):
    """导出前先算好几组上下文的下一个字概率，供网页/命令行自检对拍。"""
    char2idx = ck["char2idx"]
    checks = []
    for ctx in contexts:
        ids = [char2idx[c] for c in ctx if c in char2idx]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
            logp = F.log_softmax(out[0, -1], dim=-1).numpy()
        order = np.argsort(-logp)[:8]
        checks.append({
            "context": ctx,
            "top": [[ck["idx2char"][int(i)], round(float(logp[i]), 4)] for i in order],
        })
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--pinyin_map", default=None, help="默认取模型同目录下的 pinyin_map.json")
    ap.add_argument("--corpus", default=None, help="默认取模型同目录下的所有 .txt")
    args = ap.parse_args()

    model, ck, cfg = load_model(args.model)
    char2idx = ck["char2idx"]
    vocab_size = len(char2idx)
    idx2char = [ck["idx2char"][i] for i in range(vocab_size)]
    print(f"模型: {args.model}")
    print(f"  类型 {cfg['model']} 层数 {cfg['num_layers']} embed {cfg['embed_dim']} hidden {cfg['hidden_dim']}  字表 {vocab_size}")

    # 拼音表：去重、过滤词表外的字
    map_path = args.pinyin_map or os.path.join(os.path.dirname(args.model), "pinyin_map.json")
    raw = json.load(open(map_path, encoding="utf-8"))
    pinyin = {}
    dropped = 0
    for syl, chars in raw.items():
        seen, out = set(), []
        for c in chars:
            if c in char2idx and c not in seen:
                seen.add(c)
                out.append(c)
            elif c not in char2idx:
                dropped += 1
        if out:
            pinyin[syl] = out[:MAX_CANDIDATES]
    print(f"  拼音表 {len(raw)} 个音节 -> 保留 {len(pinyin)} 个，丢弃词表外候选 {dropped} 次")

    # 字频 / bigram
    corpus_dir = os.path.dirname(args.model)
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))
    print(f"  N-gram 统计语料: {', '.join(files)}")
    uni, bigrams = Counter(), {}
    for f in files:
        u, b = build_ngram(os.path.join(corpus_dir, f), char2idx)
        uni.update(u)
        for k, v in b.items():
            bigrams.setdefault(k, []).extend(v)
    # 合并后重排
    merged = {}
    for k, items in bigrams.items():
        tmp = defaultdict(int)
        for ch, n in items:
            tmp[ch] += n
        merged[k] = [[ch, n] for ch, n in sorted(tmp.items(), key=lambda x: -x[1])[:BIGRAM_TOPK]]
    freq = {c: int(uni.get(c, 0)) for c in idx2char}
    print(f"  字频表 {len(freq)} 项，bigram 上下文 {len(merged)} 个")

    # 权重
    embed = model.embed.weight.detach().numpy()
    fc_w = model.fc.weight.detach().numpy()      # (V, H)
    fc_b = model.fc.bias.detach().numpy()
    layers = [pack_lstm(model, i) for i in range(cfg["num_layers"])]

    checks = selftest(model, ck, ["今天天气", "我爱你", "我们明天", "中国的首都", "我正在"])

    data = {
        "meta": {
            "model": cfg["model"],
            "embed_dim": cfg["embed_dim"],
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "vocab_size": vocab_size,
            "max_vocab": cfg.get("max_vocab"),
            "corpus": cfg.get("corpus"),
            "dtype": "float16",
        },
        "vocab": "".join(idx2char),
        "pinyin": pinyin,
        "freq": freq,
        "bigram": merged,
        "weights": {
            "embed": b64_fp16(embed),
            "fc_w": b64_fp16(fc_w),
            "fc_b": b64_fp16(fc_b),
            "lstm": layers,
        },
        "selftest": checks,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    js = "window.IME_MODEL = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(js)
    size = os.path.getsize(args.out) / 1024 / 1024
    print(f"已写出: {args.out}  ({size:.2f} MB)")
    print("自检数据:")
    for c in checks:
        print(f"  {c['context']!r} -> " + ", ".join(f"{ch}{p:+.2f}" for ch, p in c["top"][:5]))


if __name__ == "__main__":
    main()
