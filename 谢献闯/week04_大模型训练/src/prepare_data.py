# -*- coding: utf-8 -*-
"""
第一步：准备 12 国语言数据集。

数据来源：WiLI-2018（235 种语言的维基百科段落，语言识别任务的标准公开数据集）
  https://huggingface.co/datasets/wili_2018
本脚本做三件事：
  1) 下载原始 parquet（若本地没有，自动走 hf-mirror 镜像）
  2) 只保留 config.LANGUAGES 里的 12 种语言，做最基础的清洗
  3) 官方 train 再按 9:1 分层切成 train/dev，test 保持官方划分
输出：data/processed/{train,dev,test}.csv，列 = text, label, label_id, label_zh
"""
import argparse
import os
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根目录（config.py）
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))  # 同目录（train.py）
import config

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
SPLITS = {
    "train": ("wili_train.parquet", f"{MIRROR}/datasets/wili_2018/resolve/main/WiLI-2018%20dataset/train-00000-of-00001.parquet"),
    "test": ("wili_test.parquet", f"{MIRROR}/datasets/wili_2018/resolve/main/WiLI-2018%20dataset/test-00000-of-00001.parquet"),
}


def download(url: str, dest: Path):
    """带进度条的简单下载，已存在则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[跳过] 已存在 {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[下载] {url}\n    -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    print(f"\r    {done / 1e6:6.1f}/{total / 1e6:.1f} MB ({pct:5.1f}%)", end="")
        print()
        tmp.replace(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载失败：{e}\n可手动下载后放到 {dest}，或设置 HF_ENDPOINT 环境变量切换镜像。"
        ) from e


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """去空白、去重、丢掉过短的样本（语言识别需要一定长度的文本）。"""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= 30]
    df = df.drop_duplicates(subset=["text"])
    return df


def read_split(path: Path) -> pd.DataFrame:
    """读一个 parquet，并把整数 label 还原成 ISO 639-3 语言码。

    该数据集在 HF 上的列名是 sentence / label(int)，
    语言码列表存在 parquet 的 huggingface 元数据里（235 种，按字母序）。
    """
    import json
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    meta = schema.metadata or {}
    if b"huggingface" not in meta:
        raise SystemExit("parquet 里没有 huggingface 元数据，无法还原语言码")
    info = json.loads(meta[b"huggingface"].decode("utf-8"))
    names = info["info"]["features"]["label"]["names"]

    df = pd.read_parquet(path)
    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    df["label"] = df["label"].astype(int).map(lambda i: names[i])
    return df


def main():
    parser = argparse.ArgumentParser(description="准备 12 国语言识别数据集")
    parser.add_argument("--force", action="store_true", help="重新生成已存在的 csv")
    args = parser.parse_args()

    config.ensure_dirs()
    random.seed(config.SEED)

    # 1) 原始数据
    raw = {}
    for split, (fname, url) in SPLITS.items():
        path = config.DATA_RAW / fname
        download(url, path)
        raw[split] = read_split(path)
        print(f"[读取] {split}: {len(raw[split])} 行, 语言数={raw[split]['label'].nunique()}")

    # 2) 过滤 12 种语言 + 清洗
    frames = {}
    for split in ("train", "test"):
        df = raw[split]
        missing = set(config.LABELS) - set(df["label"].unique())
        if missing:
            raise SystemExit(f"数据里缺少这些语言：{missing}")
        df = df[df["label"].isin(config.LABELS)]
        df = clean(df)
        df["label_id"] = df["label"].map(config.LABEL2ID)
        df["label_zh"] = df["label"].map(config.LABEL_NAMES_ZH)
        frames[split] = df.sort_values(["label_id", "text"]).reset_index(drop=True)
        print(f"[筛选] {split}: {len(df)} 条 / {df['label'].nunique()} 种语言")

    # 3) 分层切分 train / dev
    train_full = frames["train"]
    train_df, dev_df = train_test_split(
        train_full, test_size=config.DEV_RATIO,
        stratify=train_full["label_id"], random_state=config.SEED, shuffle=True,
    )
    out = {
        "train": train_df.reset_index(drop=True),
        "dev": dev_df.reset_index(drop=True),
        "test": frames["test"],
    }

    # 4) 落盘 + 统计
    print("\n" + "=" * 62)
    print(f"{'语言':<8}{'代码':<8}{'train':>8}{'dev':>7}{'test':>7}{'平均字符':>10}")
    print("-" * 62)
    for code in config.LABELS:
        row = []
        for split in ("train", "dev", "test"):
            row.append(int((out[split]["label"] == code).sum()))
        avg_len = out["train"].loc[out["train"]["label"] == code, "text"].str.len().mean()
        print(f"{config.LABEL_NAMES_ZH[code]:<8}{code:<8}{row[0]:>8}{row[1]:>7}{row[2]:>7}{avg_len:>10.0f}")
    print("-" * 62)
    print(f"{'合计':<16}{len(out['train']):>8}{len(out['dev']):>7}{len(out['test']):>7}")
    print("=" * 62)

    for split, df in out.items():
        path = config.DATA_PROCESSED / f"{split}.csv"
        if path.exists() and not args.force:
            print(f"[跳过] {path.name} 已存在（--force 可覆盖）")
            continue
        df[["text", "label", "label_id", "label_zh"]].to_csv(
            path, index=False, encoding="utf-8-sig", quoting=1  # QUOTE_ALL，文本里有逗号/引号也安全
        )
        print(f"[写出] {path}  ({len(df)} 行)")

    print("\n数据准备完成。下一步：python src/train.py")


if __name__ == "__main__":
    main()
