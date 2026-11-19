# Week06 新闻分类

目录按任务拆分：

```text
week06/
├── analysis/          # 数据分析图表
├── bert/              # BERT 训练、消融、预测
├── qwen/              # Qwen LoRA 预留目录
├── data/              # 原始数据，不修改
├── labels.py          # 15 类标签
└── outputs/
    ├── analysis/      # 数据分析图和报告
    ├── bert/
    │   ├── cls/       # 已有 cls 指标和混淆矩阵
    │   ├── mean/      # 各实验独立结果
    │   ├── max/
    │   ├── warmup/
    │   ├── focal/
    │   ├── fgm/
    │   ├── best_model/   # 只保存一份最优模型权重
    │   └── summary.csv
    └── qwen/
```

## 数据分析

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python analysis/analyze_data.py
```

输出在 `outputs/analysis/figures/`。

## BERT

直接运行默认使用 CLS：

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python bert/bert_train.py
```

运行其余消融实验。该脚本会复用已有 `cls` 结果，不重复训练 cls：

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python bert/run_all_experiments.py
```

一条命令会依次训练 Mean、Max、Warmup、Focal、FGM，每个实验固定 1 epoch。
已有 cls 结果只用于比较，不会重复训练。每个实验的指标和混淆矩阵保存到
独立目录，模型候选先放在临时目录，只有 Macro-F1 更高的模型才会替换
`outputs/bert/best_model/`，因此最终只保留一份权重。

单句预测：

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python bert/bert_predict.py \
  --model-dir outputs/bert/best_model \
  "中国乒乓球队在世界杯决赛中夺得冠军"
```

## Qwen 对比

千问比较三个方法：

1. 全量训练集 LoRA 微调，1 epoch。
2. zero-shot in-context learning，不训练。
3. few-shot in-context learning，不训练，每类提供 1 个示例。

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python qwen/run_experiments.py
```

指标分别写入 `outputs/qwen/lora/`、`outputs/qwen/zero_shot_icl/` 和
`outputs/qwen/few_shot_icl/`。ICL 不产生模型权重，LoRA adapter 保存在
`outputs/qwen/best_model/`。
