# 商品名文本分类 · BERT 多配置 & LLM 微调对照

一个用于**横向对比训练方案**的实验工程：同一份商品名数据，跑 7 种 BERT 配置和 2 种 LLM 微调方式，输出可复现的 Macro-F1 对比。

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 模块划分、数据流、设计决策、扩展点 |
| [docs/usage.md](docs/usage.md) | 完整命令、参数、部署、故障排查 |
| [compare_report.md](compare_report.md) | 实验结果对比报告 |
| [analysis/report.md](analysis/report.md) | 数据总览报告 |

## 结果速览

| 方案 | Macro-F1 | 训练耗时（4090） |
|---|---|---|
| **BERT · CLS + FGM** | **0.9901** | 113 s |
| BERT · CLS + Focal Loss | 0.9883 | 70 s |
| BERT · Max 池化 | 0.9875 | 68 s |
| BERT · CLS 池化（基线） | 0.9871 | 72 s |
| BERT · CLS+Mean 拼接 | 0.9871 | 69 s |
| BERT · CLS + Label Smoothing | 0.9870 | 71 s |
| BERT · Mean 池化 | 0.9849 | 71 s |
| Qwen3-0.6B · SFT 全参 | 0.8393 | 707 s |
| Qwen3-0.6B · LoRA | 0.8364 | 340 s |

> 多随机种子验证：FGM 相对 CLS 基线 **+0.0034**（3 个种子全部胜出），Max 池化与基线**无差异**。
> 详见 [compare_report.md](compare_report.md)。

## 数据

商品名 → 10 个品类，来自 ModelScope `product-classification-hiring-demo`（CC-BY-NC-SA-4.0）。

| split | 条数 | 说明 |
|---|---|---|
| train | 14,397 | 原 15,997 条分层切出验证集后 |
| val | 1,600 | 从 train 分层切分（seed=42, 10%） |
| test | 4,003 | 官方测试集 |

- 类别不均衡 **4.9 : 1**（果汁 2,782 vs 口香糖 564）
- 商品名平均 **14.8 字**，P95 = 21 → `max_len=32` 足够
- **6 个类别的关键词命中率 > 91%**（「茶」「面」「牙膏」直接出现在商品名里）

数据放在 `data/product-classification-hiring-demo/`，路径由 `dataset.py` 自动解析。

## 快速开始

```bash
# 1. 数据总览（生成 analysis/report.md + 三张图）
python main.py overview

# 2. 跑全部 BERT 实验（7 个，4090 上约 8 分钟）
python main.py bert --exp all

# 3. 跑 LLM 实验（LoRA + 全参 SFT）
python main.py llm --mode all

# 4. 生成对比报告
python main.py report
```

单跑某一项：

```bash
python main.py bert --exp cls_fgm
python main.py llm --mode lora --epochs 2
```

## 环境

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.10+ | |
| torch | 2.6+ | CUDA 版用于训练 |
| transformers | 4.44+ | 需支持 `dtype=` 或回退 `torch_dtype=` |
| numpy / matplotlib | — | 指标与绘图 |

**不需要 peft**：LoRA 是本工程自己实现的（`llm/lora.py`，约 50 行）。

```bash
pip install torch transformers numpy matplotlib
```

## 模型

| 模型 | 用途 | 解析顺序 |
|---|---|---|
| `bert-base-chinese` | BERT 实验 | `CLASSIFY_BERT_DIR` → `pretrain_models/` → `homework/` → `/root/text_class/pretrain_models/` |
| `Qwen3-0.6B` | LLM 实验 | `CLASSIFY_QWEN_DIR` → `pretrain_models/` → `DL/model/` → `/root/pretrain_models/` |

路径解析在 `paths.py`，找不到会列出所有尝试过的路径。云端可复用其他项目的模型，不必重复下载。

## 目录结构

```
text_classify_test/
├── dataset.py            Dataset（分层切分 + 标签映射）
├── data_overview.py      数据总览（控制台 / 报告）
├── device.py             设备与混合精度
├── metrics.py            混淆矩阵 / macro-F1 / 分类报告
├── paths.py              模型路径解析
├── report.py             对比报告生成
├── main.py               统一入口
├── bert/                 BERT 方向（池化 / 损失 / 对抗训练）
├── llm/                  LLM 方向（SFT / LoRA）
├── data/                 数据集
├── outputs/              实验结果（json + csv + checkpoint）
└── analysis/             数据总览报告与图表
```

## 主要结论

1. **任务接近饱和**：BERT 全部配置在 98.5%~99.0%，相差不到 0.5 个点
2. **池化方式无所谓**：四种池化差异在种子噪声内
3. **FGM 是唯一稳定有效的技巧**：+0.34 点，训练慢 57%
4. **Label Smoothing / Focal 基本无效**：任务没有过拟合，这两个是治过拟合的药
5. **LoRA 性价比高**：0.76% 参数拿到全参 99.6% 的效果
6. **LLM 在此任务上不划算**：比 BERT 低 15 个点——短文本 + 关键词特征强是 BERT 的舒适区
