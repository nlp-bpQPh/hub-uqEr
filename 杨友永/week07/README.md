# Week7 · 中文序列标注基准（People's Daily NER）

以 `data/peoples_daily/` 为数据，对比三种序列标注方案：

| 方案 | 模型 | 解码 |
|---|---|---|
| `bert-linear` | bert-base-chinese + Linear | 逐 token argmax |
| `bert-crf` | bert-base-chinese + Linear + CRF | 整句 Viterbi |
| `llm-lora` | Qwen2.5-0.5B-Instruct + LoRA + Linear | 逐 token argmax |

## 数据

- `train.json` 20,864 句 / `validation.json` 2,318 句 / `test.json` 4,636 句
- 字符级 BIO，7 个标签：`O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC`
- 平均句长 47 字，最长 577 字

## 目录

```
week7_text_tag/
├── common/
│   ├── config.py     # 路径解析（本地/云端自动探测）、超参数
│   ├── data.py       # 读取 + 字符级 BIO 对齐 + DataLoader
│   ├── metrics.py    # 实体级 P/R/F1、分类型统计、token acc
│   ├── eval.py       # 通用 argmax 评估（BERT-linear / LLM 共用）
│   └── utils.py      # 种子、设备、计时、落盘
├── bert/
│   ├── crf.py        # 手写批量 CRF（前向算法 + Viterbi）
│   ├── model.py      # BertTagger / BertCRFTagger
│   └── train.py      # --head linear|crf
├── llm/
│   ├── model.py      # Qwen + LoRA 标注头
│   └── train.py
├── run_all.py        # 一键跑三组实验
├── report.py         # 生成 outputs/REPORT.md
└── outputs/          # 每组实验一个目录（metrics.json / log.txt / best.pt / adapter）
```

## 运行

```bash
# 冒烟（200 条，1 轮）
python run_all.py --limit 200 --eval-limit 100 --epochs-bert 1 --epochs-llm 1

# 全量
python run_all.py

# 只跑 BERT
python run_all.py --only bert-linear,bert-crf
```

单个实验：

```bash
python -m bert.train --head linear --epochs 3 --batch-size 32 --lr 3e-5
python -m bert.train --head crf    --epochs 3 --batch-size 16 --lr 3e-5
python -m llm.train  --epochs 2 --batch-size 8 --grad-accum 4 --lr 2e-4 --lora-r 16
```

## 路径约定

默认读 `./data/peoples_daily` 与 `./pretrain_models`；也可以用环境变量覆盖：

```bash
export NER_DATA_DIR=/path/to/peoples_daily
export BERT_PATH=/path/to/bert-base-chinese
export LLM_PATH=/path/to/Qwen2.5-0.5B-Instruct
export NER_OUTPUT_DIR=/path/to/outputs
```

云端会自动探测 `/root/text_class/pretrain_models/...`（复用上一个工程的模型）。

## 指标口径

- **实体级 micro P/R/F1**：BIO 解码成 `(类型, 起点, 终点)`，边界+类型全对才算 TP
- **token 准确率**：逐字符标签准确率（辅助指标，实体会被"部分正确"高估）
- 分类型 F1：PER / ORG / LOC
