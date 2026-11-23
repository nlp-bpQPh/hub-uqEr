"""
准备 THUCNews 10 类新闻分类数据（体育/娱乐/家居/房产/教育/时尚/时政/游戏/科技/财经）。

来源：https://hf-mirror.com/datasets/spiritx2023/ThuCnews
  cnews.val.txt  ～5,000 条（10 类 × 500）
  cnews.test.txt ～10,000 条（10 类 × 1,000）

用法：
  - 从 val 里按类抽 400 条做 train、100 条做 dev（共 4,000 / 1,000）
  - 从 test 里按类抽 150 条做 test（共 1,500）
  - 保存为 _data/thucnews/{train,dev,test}.json，字段 text / label / label_name
"""
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "_data" / "thucnews"
BASE = "https://hf-mirror.com/datasets/spiritx2023/ThuCnews/resolve/main/"


def download(fname):
    p = DATA / fname
    if p.exists() and p.stat().st_size > 1000:
        return p
    print("download", fname)
    r = requests.get(BASE + fname, timeout=300)
    r.raise_for_status()
    p.write_bytes(r.content)
    return p


def parse(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        text = text.strip()
        if text:
            rows.append((label.strip(), text))
    return rows


def split_per_class(rows, n_train, n_dev, seed=42):
    by = defaultdict(list)
    for label, text in rows:
        by[label].append(text)
    rng = random.Random(seed)
    train, dev = [], []
    for label in sorted(by):
        items = by[label][:]
        rng.shuffle(items)
        train.extend((label, t) for t in items[:n_train])
        dev.extend((label, t) for t in items[n_train:n_train + n_dev])
    return train, dev


def sample_per_class(rows, n, seed=42):
    by = defaultdict(list)
    for label, text in rows:
        by[label].append(text)
    rng = random.Random(seed)
    out = []
    for label in sorted(by):
        items = by[label][:]
        rng.shuffle(items)
        out.extend((label, t) for t in items[:n])
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    val_rows = parse(download("cnews.val.txt"))
    test_rows = parse(download("cnews.test.txt"))
    labels = sorted({x[0] for x in val_rows} | {x[0] for x in test_rows})
    label2id = {name: i for i, name in enumerate(labels)}
    print("classes:", len(labels), labels)

    train, dev = split_per_class(val_rows, 400, 100)
    test = sample_per_class(test_rows, 150)

    def dump(name, rows):
        out = [{"text": t, "label": label2id[l], "label_name": l} for l, t in rows]
        (DATA / f"{name}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        lens = sorted(len(x["text"]) for x in out)
        print(f"{name}: n={len(out)} classes={len(set(x['label'] for x in out))} "
              f"len median={lens[len(lens)//2]} p95={lens[int(.95*len(lens))]}")

    dump("train", train)
    dump("dev", dev)
    dump("test", test)
    (DATA / "labels.json").write_text(
        json.dumps({name: i for i, name in enumerate(labels)}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    seen = {t for _, t in train}
    print("test overlap train:", sum(1 for _, t in test if t in seen), "/", len(test))
    print("train classes:", Counter(l for l, _ in train).most_common())
    print("→", DATA)


if __name__ == "__main__":
    main()
