"""
eval_ngram.py — 在同一份语料、同一个验证划分上，算 N-gram 与神经网络语言模型的困惑度

目的：PPT 里「N-gram / RNN / LSTM 的 PPL 对比」原来用的是 Penn Treebank 的二手数字，
换成我们自己语料上的实测值 —— 而且五个数字必须来自同一份语料、同一个划分，否则没法比。

口径与 language_model.py 一致：
    - 语料：当前目录 *.txt 拼起来
    - 划分：按行随机打散（random.seed(42)），最后 5% 做验证
    - 字表：按频率取前 4000 个字，字表外的字丢弃
    - 神经模型：字符级、seq_len=64、stride=64 不重叠切块，与训练时评估一致
用法:
    python eval_ngram.py                               # 只算 1/2/3-gram
    python eval_ngram.py --ckpt best_model.pt --ckpt best_model_rnn.pt   # 顺带评估模型
    python eval_ngram.py --json lm_stats.json          # 把结果写成 PPT 用的 json
"""

import glob
import math
import os
import random
from collections import Counter

SEED = 42
VAL_RATIO = 0.05
MAX_VOCAB = 4000
SMOOTH = 1.0     # 拉普拉斯平滑


def eval_checkpoint(path, val_text, char2idx, seq_len=64, batch=128):
    """用与训练时相同的切块口径，在验证集上算 PPL。"""
    import torch
    import torch.nn as nn

    ck = torch.load(path, map_location="cpu")
    cfg = ck["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(ck["char2idx"])

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, cfg["embed_dim"])
            rnn_cls = nn.LSTM if cfg["model"] == "lstm" else nn.RNN
            self.rnn = rnn_cls(cfg["embed_dim"], cfg["hidden_dim"],
                               num_layers=cfg["num_layers"], batch_first=True,
                               dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
            self.drop = nn.Dropout(cfg["dropout"])
            self.fc = nn.Linear(cfg["hidden_dim"], vocab_size)

        def forward(self, x):
            e = self.drop(self.embed(x))
            out, _ = self.rnn(e)
            return self.fc(self.drop(out))

    model = LM().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ids = [ck["char2idx"][c] for c in val_text if c in ck["char2idx"]]
    data = torch.tensor(ids, dtype=torch.long)
    chunks = [(data[i:i + seq_len], data[i + 1:i + seq_len + 1])
              for i in range(0, len(data) - seq_len - 1, seq_len)]
    chunks = chunks[: (len(chunks) // batch) * batch]      # 与训练时 drop_last 一致
    crit = nn.CrossEntropyLoss()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            x = torch.stack([c[0] for c in part]).to(device)
            y = torch.stack([c[1] for c in part]).to(device)
            logits = model(x)
            loss = crit(logits.reshape(-1, vocab_size), y.reshape(-1))
            total += loss.item() * y.numel()
            tokens += y.numel()
    ppl = math.exp(total / tokens)
    return cfg["model"], ppl, tokens


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="要一并评估的 checkpoint，可重复")
    ap.add_argument("--json", default=None, help="把结果写成 json（PPT 第 11/12 页的 PPL 表用）")
    args = ap.parse_args()

    text = ""
    for path in glob.glob("*.txt"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text += f.read()
    print(f"语料字符数: {len(text):,}")

    vocab = [c for c, _ in Counter(text).most_common(MAX_VOCAB)]
    vset = set(vocab)
    V = len(vocab) + 1          # +1 给「未登录字」
    print(f"字表: {V}（含 1 个 OOV）")

    lines = text.splitlines()
    random.seed(SEED)
    random.shuffle(lines)
    split = int(len(lines) * (1 - VAL_RATIO))
    val_text = "\n".join(lines[split:])
    train_text = "\n".join(lines[:split])
    print(f"训练集字符 {len(train_text):,} / 验证集字符 {len(val_text):,}")

    oov = sum(1 for c in val_text if c not in vset)
    print(f"验证集 OOV 字符占比: {oov / max(1, len(val_text)) * 100:.2f}%")

    def ngrams(s, n):
        out = []
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            if any(c not in vset for c in g):
                continue
            out.append(g)
        return out

    report = []
    for n in (1, 2, 3):
        train_cnt = Counter(ngrams(train_text, n))
        total_ctx = Counter(ngrams(train_text, n - 1)) if n > 1 else None
        total_train = len(train_text) - n + 1
        logp_sum = 0.0
        count = 0
        for g in ngrams(val_text, n):
            ctx = g[:-1]
            c_ctx = total_ctx[ctx] if n > 1 else total_train
            c_all = train_cnt[g]
            # P = (count + α) / (c_ctx + α·V)
            p = (c_all + SMOOTH) / (c_ctx + SMOOTH * V)
            logp_sum += math.log(p)
            count += 1
        ppl = math.exp(-logp_sum / count)
        report.append((n, ppl, count))
        print(f"{n}-gram  PPL = {ppl:8.2f}   ({count:,} 个待预测位置)")

    d = {n: p for n, p, _ in report}
    print("\n写入 PPT 用的对照（字符级、同一份语料、同一个划分）：")
    print(f"  1-gram(unigram)  {d[1]:.1f}")
    print(f"  2-gram           {d[2]:.1f}")
    print(f"  3-gram           {d[3]:.1f}")

    stats = {"corpus_chars": len(text), "vocab": len(vocab),
             "ngram": {str(n): round(p, 1) for n, p, _ in report}}
    for path in args.ckpt:
        kind, ppl, tokens = eval_checkpoint(path, val_text, {c: i for i, c in enumerate(vocab)})
        print(f"  {kind.upper():<5} PPL = {ppl:8.2f}   ({tokens:,} 个 token)")
        stats[f"{kind}_ppl"] = round(ppl, 2)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
