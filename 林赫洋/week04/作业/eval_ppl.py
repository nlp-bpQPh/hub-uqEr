# coding:utf8
"""
eval_ppl.py — 复算 best_model.pt 的验证集 PPL，并在同一划分上算 1/2/3-gram 的 PPL

训练脚本 language_model.py 只打印训练过程，不落盘验证指标。想写梳理、做汇报时
需要一组能重跑的数字，于是补了这个小脚本。

切分口径与训练完全一致：读 checkpoint 里存下的 args（max_chars / seq_len / stride /
val_ratio / seed / max_vocab），按同样的步骤重建语料、字表、按行打散、后 5% 做验证集。
所以这里算出的 "Val PPL" 就是训练时打印过的那个数，不是另起一套口径。

顺带算 1/2/3-gram 的 PPL（拉普拉斯平滑），于是「统计模型 vs 神经模型」的对比
是同一份语料、同一个验证集上的数字，可以直接放进报告。

用法：
    python eval_ppl.py
"""
import math
import os
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import language_model as lm

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "best_model.pt")

SMOOTH = 1.0  # 拉普拉斯平滑，和课堂 tools/eval_ngram.py 保持一致


def ngram_ppl(train_text, val_text, vset, n):
    """在 val_text 上算 n-gram 的平均负对数似然 → PPL。含字表外字符的 n-gram 跳过。"""
    def grams(s):
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            if all(c in vset for c in g):
                yield g

    V = len(vset) + 1  # +1 给「未登录字」
    train_cnt = Counter(grams(train_text))
    if n == 1:
        ctx_cnt = None
    else:
        m = n - 1
        ctx_cnt = Counter(train_text[i:i + m] for i in range(len(train_text) - m + 1)
                          if all(c in vset for c in train_text[i:i + m]))

    logp, cnt = 0.0, 0
    for g in grams(val_text):
        ctx = g[:-1]
        c_ctx = len(train_text) if n == 1 else ctx_cnt[ctx]
        c_all = train_cnt[g]
        p = (c_all + SMOOTH) / (c_ctx + SMOOTH * V)
        logp += math.log(p)
        cnt += 1
    return math.exp(-logp / cnt), cnt


def main():
    ck = torch.load(CKPT, map_location="cpu")
    a = ck["args"]
    print("checkpoint 里的训练参数：")
    for k in ("max_chars", "epochs", "seq_len", "stride", "batch_size",
              "embed_dim", "hidden_dim", "num_layers", "nhead", "dropout",
              "lr", "warmup_ratio", "max_vocab", "tie_weights", "val_ratio"):
        print(f"  {k:<12} {a.get(k)}")
    print(f"  {'参数量':<10} {sum(v.numel() for v in ck['model_state'].values()):,}")

    # ── 1. 按训练时同样的口径重建数据 ──
    text = lm.load_corpus(os.path.join(HERE, "*.txt"))
    if a["max_chars"]:
        text = text[: a["max_chars"]]
    char2idx, idx2char = lm.build_vocab(text, a["max_vocab"] or None)

    lines = text.splitlines()
    random.seed(a["seed"])
    random.shuffle(lines)
    split = int(len(lines) * (1 - a["val_ratio"]))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])
    print(f"\n语料 {len(text):,} 字符 → 字表 {len(char2idx)}")
    print(f"训练集 {len(train_text):,} 字符 / 验证集 {len(val_text):,} 字符")

    # ── 2. 神经模型 PPL ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lm.LM(
        vocab_size=len(char2idx),
        embed_dim=a["embed_dim"],
        hidden_dim=a["hidden_dim"],
        num_layers=a["num_layers"],
        model_type="transformer",
        dropout=a["dropout"],
        nhead=a["nhead"],
        max_len=max(a.get("max_len", 1024), a["seq_len"]),
        tie_weights=a["tie_weights"],
    ).to(device)
    model.load_state_dict(ck["model_state"])

    val_ds = lm.CharDataset(val_text, char2idx, a["seq_len"], a["stride"])
    loader = DataLoader(val_ds, batch_size=a["batch_size"], shuffle=False, drop_last=False)
    loss, ppl = lm.run_epoch(model, loader, nn.CrossEntropyLoss(), None, device, train=False)
    print(f"\n验证集  loss = {loss:.4f}   PPL = {ppl:.2f}")

    # ── 3. 同一划分上的 n-gram PPL ──
    vset = set(char2idx)
    print("\n同一份语料、同一个验证集上的对照：")
    print(f"  Transformer(字符级)  PPL = {ppl:8.2f}")
    for n in (1, 2, 3):
        p, cnt = ngram_ppl(train_text, val_text, vset, n)
        print(f"  {n}-gram             PPL = {p:8.2f}   ({cnt:,} 个位置)")
    print("\n（1-gram 相当于完全不看上文，只会按字频猜；3-gram 因为数据稀疏反而比 2-gram 差，"
          "这正是「统计模型靠穷举、神经模型靠压缩」的差别）")


if __name__ == "__main__":
    main()
