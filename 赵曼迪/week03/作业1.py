# gen_v12_vocab.py
import os, csv, unicodedata
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "v12", "train.csv")
OUT_PATH  = os.path.join(HERE, "vocab", "vocab_v12.txt")  # 和你 v8 放一起

def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))

freq = Counter()
with open(TRAIN_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        freq.update(normalize(row["text"]))

chars = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
for ch, _ in chars:
    vocab[ch] = len(vocab)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("# 字符\tid\t出现次数\n")
    for ch, idx in vocab.items():
        f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

print(f"完成：{len(vocab)} 个字符 → {OUT_PATH}")
