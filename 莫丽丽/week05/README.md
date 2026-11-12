# Transformer SFT（纯监督微调版）

**自定义 Decoder-only Transformer + SFT 数据处理 + Assistant-only Loss Mask + 训练/验证 + 纯 SFT 推理**。

> 从随机初始化参数开始训练，不是对 Qwen/Llama 等预训练模型进行工业级 Fine-Tuning。3,000 条混合任务样本通常不足以训练出通用中文问答能力，不能保证达到通用大模型的准确率。

## 1. 项目结构

```text
transformer_sft_only/
├── transformer_lm.py       # Pre-LN Decoder-only Transformer 与自回归生成
├── dataset.py              # JSONL、字符级 Tokenizer、Prompt 和 Label Loss Mask
├── train_sft.py            # SFT 训练、验证、调度器、梯度累积、早停和权重保存
├── inference.py            # 纯 SFT 单次问答、交互问答、生成准确率评估
├── README.md
├── requirements.txt
├── data/
│   ├── sft_train.jsonl     # 原始 3,000 条训练样本
│   └── sft_val.jsonl       # 原始 200 条验证样本
├── tests/
│   └── test_sft_only.py
├── checkpoints/            # 自动生成 best.pt / latest.pt
└── logs/                   # 自动生成 metrics.csv 和预测文件
```

## 2. 开始训练

```powershell
python train_sft.py --epochs 20 --patience 10 --batch-size 16 --grad-accum-steps 2 --max-seq-len 256 --max-target-chars 128 --d-model 256 --heads 8 --layers 4 --dropout 0.05 --lr 3e-4 --warmup-ratio 0.03 --min-lr-ratio 0.1 --device cuda
```

有效 Batch Size 约为 32（16 × 2）。数据集里的问题和答案平均长度较短，`max-seq-len=256` 通常足够；启动训练时程序会报告截断样本数量。如果出现截断，再提高长度。

### 显存不足

```powershell
python train_sft.py --epochs 50 --patience 10 --batch-size 8 --grad-accum-steps 4 --max-seq-len 256 --d-model 192 --heads 4 --layers 4 --dropout 0.05 --lr 2e-4 --device cuda
```

### 训练结果

```text
checkpoints/
├── best.pt       # 验证 Loss 最低的模型，推理优先使用它
└── latest.pt     # 最后一轮模型

logs/
└── metrics.csv   # 每轮 Train Loss、Val Loss、PPL、学习率和耗时
```

训练参数改变后建议使用新的输出目录，或先备份旧 checkpoint 和日志。该脚本每次从随机初始化开始，不会自动续训旧权重。

## 3.问答

单条问题：

```powershell
python inference.py --checkpoint checkpoints/best.pt --question "给定两个数字，请选择其中的较大数。数字1：238 数字2：564" --device cuda
```

交互式问答：

```powershell
python inference.py --checkpoint checkpoints/best.pt --device cuda
```

输入 `exit`、`quit` 或 `q` 退出。准确率测试建议保持默认贪心解码（`temperature=0`），不要使用随机采样。

## 4. 测试真实生成准确率

Teacher Forcing 验证 Loss 并不等于生成准确率。建议单独测试训练集和验证集真实逐 Token 生成结果。

测试训练集：

```powershell
python inference.py --checkpoint checkpoints/best.pt --eval-data data/sft_train.jsonl --prediction-file logs/train_predictions.jsonl --device cuda
```

测试验证集：

```powershell
python inference.py --checkpoint checkpoints/best.pt --eval-data data/sft_val.jsonl --prediction-file logs/val_predictions.jsonl --device cuda
```

输出包含：

- Exact Match：生成文本完全一致的比例。
- Normalized Exact Match：统一空白和标点后完全一致的比例。
- 字符 F1：字符多重集的表面重合度，**不等于语义正确率**。
- 逐条预测保存在 `logs/*_predictions.jsonl`，可以逐条检查错误类型。

