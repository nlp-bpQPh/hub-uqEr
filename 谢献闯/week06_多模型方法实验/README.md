# 第六周作业：更换预训练模型与数据集后的分类方法对比实验

本目录是课程「第六周 分类任务 · week6-数据接入与分流」的作业实现。作业要求：
**更换预训练模型或数据集，做分类任务各种方法的对比实验**。这里两个变量都做了控制变量实验。

## 实验设计

| 实验 | 被测变量 | 控制变量 |
|---|---|---|
| 实验一 | 编码器预训练权重：BERT-base-chinese / RoBERTa-wwm-ext / MacBERT-base | TNEWS 15 类、同一套超参、同一切分 |
| 实验二 | 数据集：TNEWS 15 类 / THUCNews 10 类 / ChnSentiCorp 情感二分类 | 每个数据集内方法、配方、切分一致 |

三个编码器参数量均为 102.3M（12 层 / 768 维），差异只来自预训练语料与掩码策略。

## 方法

- 规则 / 关键词（L0）
- TF-IDF + LinearSVC（L1）
- 冻结编码器特征 + 线性分类器（预训练表示，不微调）
- BERT-base-chinese / RoBERTa-wwm-ext / MacBERT-base 全参数微调

## 数据与评测

- train / dev / test 三段切分；dev 只用于选 epoch，test 只评一次；
- TNEWS 验证集去重后 8,901 条作 test（原验证集 11% 与训练集重复）；dev 从 train 分层抽 3,000；
- THUCNews：50,000 / 5,000 / 10,000；ChnSentiCorp：9,600 / 1,200 / 1,200；
- 主指标 Macro-F1（去重 test），辅以 Accuracy、per-class recall、ECE、训练耗时、显存峰值。

## 复现

```bash
# 0) 依赖：torch 2.6 + transformers 4.55 + scikit-learn + matplotlib（py312 环境）
python code/make_stats.py

# 1) 无预训练基线（CPU，秒级）
python code/run_rule.py
python code/run_tfidf.py --dataset tnews
python code/run_tfidf.py --dataset thucnews
python code/run_tfidf.py --dataset chnsenticorp

# 2) 三个编码器 × 三个数据集（GPU，单卡 3060 约 1 小时）
python code/train_encoder.py --model bert-base       --dataset tnews
python code/train_encoder.py --model roberta-wwm-ext --dataset tnews
python code/train_encoder.py --model macbert-base    --dataset tnews
# ……其余数据集同理，或直接运行 code/run_queue.ps1

# 3) 冻结特征 + 线性分类器
python code/run_probe.py --model bert-base --dataset tnews

# 4) 汇总 + 图表 + 报告
python code/collect.py
python code/analysis.py
python code/make_report.py
```

## 目录

```
code/            实验脚本（数据集/模型可配置）
data/            THUCNews、ChnSentiCorp（TNEWS 复用课程 data/）
models/          chinese-roberta-wwm-ext、chinese-macbert-base
outputs/         每个组合的 JSON、results.csv、summary.md
figures/         报告用图
报告_第六周_分类方法对比实验.docx
```

## 主要结果（去重测试集 Macro-F1）

| 数据集 | TF-IDF | BERT-base | RoBERTa-wwm-ext | MacBERT-base |
|---|---|---|---|---|
| TNEWS 15 类 | 0.5104 | 0.5357 | 0.5494 | **0.5508** |
| THUCNews 10 类 | 0.9513 | 0.9632 | 0.9650 | **0.9685** |
| ChnSentiCorp 情感 | 0.9032 | 0.9354 | **0.9423** | 0.9375 |

## 主要结论（详见报告）

1. 同参数量的 RoBERTa-wwm-ext / MacBERT-base 相对 BERT-base 在 TNEWS 上 +1.4~1.5 个点，训练成本几乎不变；
2. 预训练表示本身已有价值：冻结特征 + 线性分类器在 TNEWS 上 0.5389，比 TF-IDF 高 2.9 个点，但仍低于微调 1~2 个点；在 THUCNews / ChnSentiCorp 上与 TF-IDF 持平甚至略低；
3. 换数据集后方法排序稳定（编码器 > TF-IDF > 规则），但绝对差距随任务变化：
   THUCNews 上 TF-IDF 已到 0.951，编码器只高 1.7 个点；ChnSentiCorp 上高 3.9 个点；
4. ChnSentiCorp 测试集 27.6% 与训练集完全重复，去重后所有方法掉 1~3 个点——换数据集先洗数据。
