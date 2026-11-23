# 换预训练模型对比实验（TNEWS 15 类）

> 任务：数据集不变（CLUE TNEWS），把项目里的 **BERT-base（12 层 / 102.3M）** 换成
> **StructBERT-tiny（4 层 / 256 维 / 8.77M）**，在同一批 **500 条干净验证样本（seed=42）** 上
> 对比规则 / TF-IDF / 小模型 / BERT-base 的效果与成本。

## 为什么换小模型

本机 **无 NVIDIA GPU**（torch 2.11.0+cpu，i5-1155G7 / 4 核 8 线程）：

- BERT-base 全量微调 53k×3 epoch 预计 3–5 天，跑不动；
- StructBERT-tiny 单步约 1.3 s，3 epoch 约 1 小时，可以在这台机器上跑完；
- 结论仍然有效：同一份训练数据、同一批评测样本，只换模型容量。

## 目录

```
experiments/
├── train_small_bert.py     # 小模型微调 + 评测（3 epoch，类别加权，[CLS] 池化）
├── run_baselines.py        # 规则 / TF-IDF+LinearSVC / TF-IDF+SGD(balanced)
├── REPORT.md               # 最终对比报告（静态文件，已随实验生成）
└── outputs/
    ├── rule_pred.json
    ├── tfidf_pred.json
    ├── tfidf_sgd_pred.json
    ├── small_bert_pred.json
    ├── small_bert_log.json
    ├── small_bert_ckpt.pt
    ├── summary.md
    └── comparison.png
```

## 模型来源

- 模型：`iic/nlp_structbert_backbone_tiny_std`（ModelScope）
- 本地目录：`_models/structbert-tiny/`
  - `config.json`、`vocab.txt`、`pytorch_model.bin`（35 MB）
- 加载方式：`transformers.BertConfig` + `BertModel`，把预训练权重的 `bert.` 前缀去掉后
  `load_state_dict(strict=False)`（`cls.*` 是 MLM 头，丢弃）。
- Tokenizer：`BertTokenizerFast(vocab_file=...)`；已与 `bert-base-chinese` 的 tokenizer
  逐条比对过，**前 5 条样本的 input_ids 完全一致**。

> 网络说明：HuggingFace 直连被自签证书拦截，hf-mirror 可用但偶发限流，
> 所以模型从 ModelScope 下载。下载脚本没有放进仓库（只需一次）。

## 运行

```powershell
# 1) 轻量基线（约 30 秒）
python experiments/run_baselines.py

# 2) 小模型微调（CPU，约 1 小时）
python experiments/train_small_bert.py --epochs 3
```

报告与图已生成：`REPORT.md`、`outputs/summary.md`、`outputs/comparison.png`；需要改数字时直接编辑这些文件。

冒烟（不落盘）：

```powershell
python experiments/train_small_bert.py --epochs 1 --max_steps 3 --no_save
```

## 口径

| 项 | 说明 |
|---|---|
| 训练集 | TNEWS train 53,360 条 |
| 评测集 | `common.data.load_clean_val(500)`：去重后 8,901 条中按 seed=42 抽样 |
| 指标 | Accuracy（含失败）、Macro-F1、4 组主题准确率、相关性 F1、per-class recall |
| 小模型超参 | 3 epoch、batch 64、max_length 64、lr 5e-5（分类头 ×2）、类别加权、warmup 10% |
| BERT-base | 项目已有结果（`code/outputs/bert_pred.json`），未重训 |
| 随机种子 | 42（数据抽样、torch/numpy/random） |

## 限制

- 小模型只做了 3 epoch、lr 5e-5，没有超参搜索；换 lr / 更长训练可能再涨 1–2 个点。
- 500 条样本的统计噪声约 ±2 个点，1–2 个点的差异不要过度解读。
- BERT-base 的训练耗时 828 s 来自 `code/outputs/bert_train_log.json`（GPU）。
- 没有跑 LLM zero-shot / LoRA / 全量微调：本机无 GPU 或需要 API key；
  这些结果见 `code/outputs/summary.md`，本报告只在需要时引用。
