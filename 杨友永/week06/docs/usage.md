# 使用文档

## 一、安装

```bash
pip install torch transformers numpy matplotlib
```

不需要 `peft`（LoRA 自带实现）、不需要 `sklearn`（指标自带实现）。

## 二、模型准备

两种模型按需放置，`paths.py` 会自动查找：

```bash
# 方式 1：放项目内（推荐）
pretrain_models/
├── bert-base-chinese/
└── Qwen3-0.6B/

# 方式 2：环境变量指定
export CLASSIFY_BERT_DIR=/path/to/bert-base-chinese
export CLASSIFY_QWEN_DIR=/path/to/Qwen3-0.6B
```

检查是否找到：

```bash
python paths.py
# bert_dir: /path/.../pretrain_models/bert-base-chinese
# qwen_dir: /path/.../pretrain_models/Qwen3-0.6B
```

只跑 BERT 实验时，**不需要 Qwen3**——`qwen_dir()` 只在 LLM 入口被调用。

## 三、命令总览

```bash
python main.py overview                     # 数据总览
python main.py bert --exp all               # 7 个 BERT 实验
python main.py bert --exp cls_fgm           # 单个实验
python main.py llm --mode all               # LoRA + 全参 SFT
python main.py llm --mode lora              # 只跑 LoRA
python main.py report                       # 生成对比报告
```

也可以直接调子模块：

```bash
python -m bert.run_bert --exp mean --epochs 2
python -m llm.run_llm --mode full --limit 2000
```

## 四、BERT 实验

### 4.1 七个预设

| `--exp` | 池化 | 额外配置 |
|---|---|---|
| `cls` | CLS | — （基线） |
| `mean` | Mean | |
| `max` | Max | |
| `cls_mean` | CLS+Mean 拼接 | |
| `cls_fgm` | CLS | FGM，`eps=1.0` |
| `cls_ls` | CLS | Label Smoothing，`0.1` |
| `cls_focal` | CLS | Focal Loss，`gamma=2.0` |

### 4.2 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--exp` | `all` | 实验名或 `all` |
| `--epochs` | 3 | |
| `--lr` | 3e-5 | |
| `--batch-size` | 32 | |
| `--device` | 自动 | `cuda` / `mps` / `cpu` |

### 4.3 默认超参（`bert/train.BertConfig`）

```python
pooling="cls"      lr=3e-5          epochs=3        batch_size=32
max_len=32         weight_decay=0.01 warmup_ratio=0.1
max_grad_norm=1.0  dropout=0.1      seed=42
label_smoothing=0  focal_gamma=0    fgm_eps=0
```

> `label_smoothing / focal_gamma / fgm_eps` 三个都是 **> 0 才启用**，默认全关。

### 4.4 输出

```
outputs/bert_cls.json    完整结果（config、逐轮 history、best_macro_f1、train_s）
outputs/bert_cls.pt      最优轮次权重
outputs/results.csv      追加一行汇总
```

控制台：

```
[device] cuda (NVIDIA GeForce RTX 4090) · bfloat16 autocast
[run] bert_cls  pooling=cls loss=ce fgm=0.0
[model] {'total': 102275338, ...}
  ep1 loss=0.2134 val macro-F1=0.9812 acc=0.9875
  ep2 loss=0.0712 val macro-F1=0.9865 acc=0.9906
  ep3 loss=0.0431 val macro-F1=0.9871 acc=0.9912
[done] bert_cls  71.9s  best macro-F1=0.9871
```

## 五、LLM 实验

### 5.1 两种模式

| `--mode` | 训练参数 | 学习率 | 显存占用 |
|---|---|---|---|
| `lora` | 4.6 M（0.76%） | 2e-4 | ~6 GB |
| `full` | 596 M（100%） | 2e-5 | ~18 GB |

### 5.2 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `all` | `lora` / `full` / `all` |
| `--epochs` | LoRA 2 / 全参 1 | |
| `--batch-size` | LoRA 16 / 全参 4 | |
| `--limit` | 无 | 限制训练样本数，调试用 |

### 5.3 默认超参（`llm/train.LLMConfig`）

```python
max_len=128        weight_decay=0.0   warmup_ratio=0.03
max_grad_norm=1.0  lora_r=16          lora_alpha=32
seed=42            eval_batch_size=32
```

### 5.4 输出

```
outputs/llm_lora.json    结果
outputs/llm_lora.pt      只存 LoRA 的 A/B 矩阵（约 18 MB）
outputs/llm_full.pt      完整 state_dict（约 2.4 GB）
```

## 六、数据总览

```bash
python main.py overview                      # 控制台概览 + 生成报告
python data_overview.py                      # 只打控制台
python data_overview.py --report             # 生成 analysis/report.md + 三张图
```

报告包含：规模、类别分布、长度分布、关键词命中率、数据质量、难点分析、建模建议。

## 七、对比报告

```bash
python main.py report
```

读取 `outputs/*.json` 与 `outputs/seed_*.json`，生成：

- `compare_report.md`（工程根目录）
- `analysis/figures/compare_f1.png`

报告里的显著性判断是**按数据算的**：`min(FGM) > max(CLS)` 才判为「真实提升」。

## 八、部署到云主机

```bash
# 1. 打包（排除模型与输出）
tar czf /tmp/cls_code.tar --exclude='__pycache__' --exclude='outputs' \
    --exclude='pretrain_models' --exclude='analysis' *.py bert llm data

# 2. 上传
scp -P 23 /tmp/cls_code.tar root@HOST:/root/

# 3. 解压 + 链接模型
ssh -p 23 root@HOST 'cd /root/text_classify_test && tar xzf /root/cls_code.tar \
    && ln -sfn /root/text_class/pretrain_models/bert-base-chinese pretrain_models/bert-base-chinese'

# 4. 跑实验
ssh -p 23 root@HOST 'cd /root/text_classify_test && \
    python -m bert.run_bert --exp all > outputs/bert_all.log 2>&1'
```

云端用 conda 环境（如 `/usr/local/miniconda3/envs/py312/bin/python`）时，
把该路径写进脚本，不要依赖 `python` 指向。

## 九、故障排查

### 9.1 找不到模型

```
FileNotFoundError: 找不到模型 bert-base-chinese，已尝试:
  - .../pretrain_models/bert-base-chinese
  ...
```

报错会列出所有尝试路径。用环境变量显式指定：

```bash
export CLASSIFY_BERT_DIR=/root/text_class/pretrain_models/bert-base-chinese
```

### 9.2 找不到数据

```
FileNotFoundError: 找不到数据集目录，已尝试: ...
```

默认找 `data/product-classification-hiring-demo/`，也可用 `PRODUCT_DATA_DIR` 指定。

### 9.3 全参 SFT 显存不足

```
torch.OutOfMemoryError: CUDA out of memory
```

按顺序尝试：

1. **确认没有残留进程占用显存**（前一个实验未完全退出是常见原因）

   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv
   ```

2. 降 batch：`--batch-size 2`
3. 改用 LoRA：`--mode lora`（显存约 1/3）

### 9.4 MPS 上训练报 bf16 错误

`device.py` 已限制只有 CUDA 开 autocast，MPS/CPU 走 fp32。
如果自己改代码，注意 `torch.autocast("cuda", ...)` 在非 CUDA 设备上会构造失败。

### 9.5 LLM 训练不收敛 / 效果远低于预期

检查三处：

1. **LoRA 是否真的冻结了底座**

   ```
   [model] LoRA  替换 112 层  可训练 4,587,520 / 600,637,440 (0.76%)
   ```

   应该是 **0.76%** 左右。如果是 70%，说明只冻结了被替换的层。

2. **chat template 是否关了思考模式**

   Qwen3 默认开 `enable_thinking`，答案前会插入推理块。
   `llm/data.build_chat` 已自动处理；自己写 prompt 时要加 `enable_thinking=False`。

3. **tokenizer 是否左填充**

   `llm/model.load_llm` 设了 `padding_side="left"`。
   右填充时 `logits[:, -1]` 取到的是 PAD 位置，打分完全错乱（表现为 AUC/准确率低于随机）。

### 9.6 报告里数字对不上

`report.py` 取的是**最优轮次**的指标（`history[best_epoch-1]`），不是最后一轮。
FGM 的最优轮通常是第 2 轮，第 3 轮会有轻微回退。

## 十、加一个新实验

以「加一种 Attention 池化」为例：

```python
# 1. bert/model.py
POOLINGS = ("cls", "mean", "max", "cls_mean", "attn")

# 2. pool() 里加一个分支
if self.pooling == "attn":
    w = torch.softmax(self.attn(hidden).squeeze(-1).masked_fill(~mask.bool(), -1e9), -1)
    return (hidden * w.unsqueeze(-1)).sum(1)

# 3. __init__ 里定义 self.attn（注意 cls_mean 才翻倍维度）

# 4. bert/run_bert.py 的 PRESETS 加一行
"attn": BertConfig(pooling="attn"),

# 5. 跑
python main.py bert --exp attn
python main.py report
```

`report.py` 的 `LABELS` 里补一条中文名，报告就能正确显示。
