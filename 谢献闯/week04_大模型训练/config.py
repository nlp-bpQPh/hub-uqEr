# -*- coding: utf-8 -*-
"""
全局配置：12 种语言的标签映射、数据/模型路径、训练超参数。

想换语言集合，只改 LANGUAGES 即可（代码用 ISO 639-3 语言码，与 WiLI-2018 数据集一致）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- 路径 ----------
DATA_RAW = ROOT / "data" / "raw"              # 原始 parquet
DATA_PROCESSED = ROOT / "data" / "processed"  # 切分后的 train/dev/test.csv
PRETRAINED_DIR = ROOT / "models" / "pretrained" / "bert-base-multilingual-cased"
SAVE_DIR = ROOT / "models" / "bert-langid-12"  # 微调后保存的位置
OUTPUT_DIR = ROOT / "outputs"                  # 指标、混淆矩阵、日志

# ---------- 12 种国家语言 ----------
# (语言码, 中文名, 英文名)
LANGUAGES = [
    ("zho", "中文", "Chinese"),
    ("eng", "英语", "English"),
    ("jpn", "日语", "Japanese"),
    ("kor", "韩语", "Korean"),
    ("fra", "法语", "French"),
    ("deu", "德语", "German"),
    ("spa", "西班牙语", "Spanish"),
    ("rus", "俄语", "Russian"),
    ("ara", "阿拉伯语", "Arabic"),
    ("por", "葡萄牙语", "Portuguese"),
    ("ita", "意大利语", "Italian"),
    ("tur", "土耳其语", "Turkish"),
]

LABELS = [code for code, _, _ in LANGUAGES]
ID2LABEL = {i: code for i, code in enumerate(LABELS)}
LABEL2ID = {code: i for i, code in enumerate(LABELS)}
LABEL_NAMES_ZH = {code: zh for code, zh, _ in LANGUAGES}
LABEL_NAMES_EN = {code: en for code, _, en in LANGUAGES}
NUM_LABELS = len(LABELS)

# ---------- 训练超参数 ----------
SEED = 42
MAX_LEN = 128          # 截断长度；语言识别只需前 128 个 token
BATCH_SIZE = 16
LEARNING_RATE = 2e-5   # BERT 微调常用学习率
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 2   # 验证集 macro-F1 连续多少轮不提升就停
DEV_RATIO = 0.1           # 从官方训练集里再切 10% 做验证集


def ensure_dirs():
    for d in (DATA_RAW, DATA_PROCESSED, PRETRAINED_DIR.parent, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
