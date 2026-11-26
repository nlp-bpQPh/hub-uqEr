# 架构文档

## 一、设计目标

这是一个**对照实验工程**，不是生产服务。核心诉求：

1. **单变量可对照** —— 改一个因素（池化 / 损失 / 微调方式），其他全不变
2. **加实验成本低** —— 加一种池化只需改一个 `if`，加一个实验只需加一行 preset
3. **结果可汇总** —— 所有实验写同一份 `outputs/results.csv`，报告自动生成
4. **可复现** —— 固定种子、固定切分、固定评测口径

## 二、分层结构

```
┌──────────────────────────────────────────────┐
│ 入口层    main.py                             │
│           bert/run_bert.py  llm/run_llm.py   │
│           负责：解析参数、按预设组装 Config    │
├──────────────────────────────────────────────┤
│ 实验层    bert/train.py     llm/train.py     │
│           负责：训练循环、每轮评测、存最优      │
├──────────────────────────────────────────────┤
│ 组件层    bert/model.py     llm/model.py     │
│           bert/losses.py    llm/lora.py      │
│           bert/dataloader.py llm/data.py     │
│           负责：模型结构、损失、数据转换        │
├──────────────────────────────────────────────┤
│ 基础层    dataset.py  metrics.py  device.py   │
│           paths.py                          │
│           负责：数据加载、指标、设备、路径      │
└──────────────────────────────────────────────┘
```

**依赖方向严格向下**，基础层不 import 组件层，组件层不 import 实验层。
唯一的例外是 `llm/train.py` 复用了 `bert/train.py` 的 `record()` 写汇总表。

## 三、数据流

### 3.1 BERT 路径

```
data/*.jsonl
    │  dataset.ProductDataset（分层切分、标签映射）
    ▼
Sample(text, label, category)
    │  bert/dataloader.make_collate（分词 + padding，max_len=32）
    ▼
{input_ids, attention_mask, labels, token_type_ids}
    │  bert/model.BertClassifier
    ▼
   BERT-base → pool() → Dropout → Linear(768→10) → logits
    │  bert/losses：CE / LS / Focal
    ▼
   loss → (可选) FGM 二次前向 → 梯度累加 → step
```

### 3.2 LLM 路径

```
Sample(text, label, category)
    │  llm/data.build_chat（chat template + enable_thinking=False）
    ▼
"<|im_start|>system...<|im_start|>user...品类："  +  "即饮奶茶<|im_end|>"
    │  SFTDataset：labels 只在答案部分有值
    ▼
{input_ids, attention_mask, labels(前段=-100)}
    │  llm/model：Qwen3-0.6B（全参 or LoRA）
    ▼
   因果 LM 的交叉熵（只在答案 token 上算）
    │
    ▼
评测不走生成，而是取答案位置的 logits，比较 10 个类别首 token
```

**为什么评测用 verbalizer 而不是生成**：
- 生成 1600 条 × 数十 token，慢且不稳定
- verbalizer 一次前向就出结果，与 BERT 的 macro-F1 口径完全可比

## 四、关键设计决策

### 4.1 标签 id 按名称排序，不按频率

```python
names = sorted({r["category"] for r in rows})
```

频率排序会在数据变化后让 id 整体错位，历史 checkpoint 与报告全部对不上。

### 4.2 验证集从 train 分层切分

数据只有 train/test。直接切尾部会让稀有类比例失真（果汁 2,782 vs 口香糖 564）。
按类别分组后再切，保证 val 的不均衡比（5.0:1）与 train（4.9:1）一致。

```python
for items in by_cat.values():
    rng.shuffle(items)
    n_val = max(1, round(len(items) * ratio))
    val.extend(items[:n_val]); train.extend(items[n_val:])
```

### 4.3 池化统一在 `pool()` 一个方法里

```python
def pool(self, hidden, mask):
    if self.pooling == "cls":  return hidden[:, 0]
    ...
```

四种池化共用同一个分类头接口，分类头输入维度按 `cls_mean` 翻倍。
换池化只改一个字符串，不动任何其他代码。

**Max 池化必须屏蔽 PAD**：

```python
filled = hidden.masked_fill(~mask.bool().unsqueeze(-1), torch.finfo(hidden.dtype).min)
```

否则 PAD 位置的隐状态会主导 max。

### 4.4 FGM 的梯度累加，而不是两次更新

```python
loss.backward()          # ① 正常梯度
fgm.attack()             # ② 扰动 embedding
loss2.backward()         # ③ 对抗梯度累加到 .grad
fgm.restore()            # ④ 复原
opt.step()               # ⑤ 用梯度之和更新一次
```

只在 embedding 层扰动（`name="word_embeddings"`），不改其他参数。

### 4.5 LoRA 先冻结底座，再注入

```python
for p in model.parameters():
    p.requires_grad = False     # ← 必须先做
n = apply_lora(model, r, alpha)
```

**如果只依赖 `LoRALinear` 内部冻结被替换的那层**，embedding / LayerNorm / MLP 仍然是可训练的，
实测可训练参数会占到 **70.67%**，等于「LoRA + 大半全参微调」。修正后是 **0.76%**。

### 4.6 LoRA 自实现，不依赖 peft

```python
y = W x + (alpha / r) * B(A(x))
```

- `A` 随机初始化，`B` 置零 → 训练开始时增量恒为 0，不破坏预训练权重
- 目标层：`q_proj / k_proj / v_proj / o_proj`
- 省掉一个依赖，也避免 peft 与 transformers 版本耦合

### 4.7 路径解析而非写死

项目目录调整过多次（`pretrain/` → `pretrain_models/`、数据移入 `data/`），
所以 `paths.py` 与 `dataset.resolve_data_dir()` 都按优先级探测，并支持环境变量覆盖。

### 4.8 结果统一落一份 CSV

```python
bert/train.record(result)   # 训练结束调用
```

`outputs/results.csv` 是跨实验的唯一汇总源，`report.py` 只读 `outputs/*.json`。

## 五、扩展点

| 想加什么 | 改哪里 | 改动量 |
|---|---|---|
| 新池化方式 | `bert/model.py` 的 `POOLINGS` + `pool()` | 3 行 |
| 新损失函数 | `bert/losses.build_criterion()` 加分支 | 5 行 |
| 新 BERT 实验组合 | `bert/run_bert.py` 的 `PRESETS` 加一行 | 1 行 |
| 新 LLM 微调方式 | `llm/train.py` 的 `mode` 分支 | ~10 行 |
| 新指标 | `metrics.py` 加函数，`train.evaluate` 返回 | 5 行 |
| 换数据集 | 覆盖 `dataset.py` 的 `ProductDataset` | 视情况 |

## 六、产物布局

```
outputs/
├── bert_cls.json            每个实验一份完整结果（含逐轮 history）
├── bert_cls.pt              最优轮次的权重
├── llm_lora.pt              LoRA 只存 A/B 矩阵
├── llm_full.pt              全参存完整 state_dict（2.4 GB）
└── results.csv              跨实验汇总（run_id, family, macro_f1, train_s）

analysis/
├── report.md / report.pdf   数据总览报告
├── compare_report.md        方案对比报告（另有一份在工程根）
└── figures/                 全部图表
```
