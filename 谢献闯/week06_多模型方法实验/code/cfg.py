"""实验全局配置：路径、模型注册表、数据集标签。

本作业在课程代码基础上做两处「更换」：
  1) 换预训练模型：BERT-base-chinese → RoBERTa-wwm-ext / MacBERT-base
  2) 换数据集：TNEWS 15 类 → THUCNews 10 类 / ChnSentiCorp 情感二分类
所有方法共用同一套数据切分、同一套指标，保证横向可比。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"

# 课程原始数据（TNEWS 与规则基线都在这里）
COURSE_CODE = Path(r"E:\大模型学习\学习资料\第六周 分类任务\week6-数据接入与分流\code")
COURSE_TNEWS_TRAIN = COURSE_CODE / "data" / "train.json"
COURSE_TNEWS_VAL = COURSE_CODE / "data" / "val.json"

# 三个编码器：参数量同为 ~102M（BERT-base 结构），只换预训练权重与语料
MODEL_REGISTRY = {
    "bert-base": {
        "path": Path(r"E:\大模型学习\学习资料\第四周 语言模型\bert-base-chinese"),
        "display": "BERT-base-chinese",
        "pretrain": "Google 中文维基 + 通用语料，字级 MLM",
    },
    "roberta-wwm-ext": {
        "path": MODELS_DIR / "chinese-roberta-wwm-ext",
        "display": "RoBERTa-wwm-ext",
        "pretrain": "哈工大讯飞，5.4B tokens，全词掩码 + 动态掩码，去 NSP",
    },
    "macbert-base": {
        "path": MODELS_DIR / "chinese-macbert-base",
        "display": "MacBERT-base",
        "pretrain": "哈工大讯飞，MLM as correction（相似词替换），约 200 亿字",
    },
}

# 统一标签口径（顺序必须固定，混淆矩阵/per-class 都按这个顺序）
DATASET_SPECS = {
    "tnews": {
        "display": "TNEWS 15 类新闻标题",
        "labels": ["故事", "文化", "娱乐", "体育", "财经",
                   "房产", "汽车", "教育", "科技", "军事",
                   "旅游", "国际", "证券", "农业", "电竞"],
        "num_labels": 15,
        "max_length": 64,
    },
    "thucnews": {
        "display": "THUCNews 10 类新闻标题",
        "labels": ["教育", "娱乐", "时尚", "财经", "游戏",
                   "时政", "社会", "体育", "股票", "科技"],
        "num_labels": 10,
        "max_length": 64,
    },
    "chnsenticorp": {
        "display": "ChnSentiCorp 情感二分类",
        "labels": ["负面", "正面"],
        "num_labels": 2,
        "max_length": 128,
    },
}
