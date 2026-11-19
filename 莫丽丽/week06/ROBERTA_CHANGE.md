# BERT → RoBERTa 改动说明

## 1. 改动目标

本次实验只替换分类模型使用的预训练骨干：将原来的中文 BERT 预训练模型改为中文 RoBERTa，其他训练、数据和评测条件保持不变，用于比较预训练模型变化对分类效果的影响。

PPT 中原项目的实验约束是：同一份 TNEWS 数据、同一套 15 类标签映射、同一批去重验证样本、相同池化方式与损失函数，并使用 Macro-F1 作为主要指标。本次代码沿用这些约束。

## 2. 代码改动

| 文件                       | 改动                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `common/model_paths.py`    | 将本地预训练模型目录从 `bert-base-chinese` 改为 `chinese-roberta-wwm-ext`，新增 `roberta_path()`，仍然只查找本地权重，不联网下载或回退到 BERT。 |
| `bert/train_bert.py`       | 使用 `RobertaClassifier`，通过 `AutoModel` 加载 RoBERTa；保留原来的 CLS/Mean/Max 池化、类别加权、Label Smoothing、Focal Loss、FGM、分层学习率、Warmup、AMP 和训练轮数默认值。 |
| `bert/predict_bert.py`     | 加载 RoBERTa checkpoint，默认读取 `outputs/roberta_ckpt.pt`，默认输出 `outputs/roberta_pred.json`，结果方法名改为 `roberta_finetune`。 |
| `service/classify_news.py` | 服务改为加载 RoBERTa，默认 checkpoint 改为 `roberta_ckpt.pt`，模型标识改为 `roberta-cls-v1`。 |
| `service/classify_cli.py`  | 服务后端改为 RoBERTa，新增推荐参数 `--model roberta`；保留 `--model bert` 作为旧命令兼容别名，但实际加载的仍是 RoBERTa。 |
| `eval/eval_all.py`         | 汇总结果改为读取 `outputs/roberta_pred.json`，展示名称改为“RoBERTa 微调”。 |

代码目录仍保留 `bert/` 名称，是为了保持原有脚本入口和项目结构不变。目录名不代表继续使用 BERT，实际加载模型由 `roberta_path()` 决定。

## 3. 预训练权重放置

请将 RoBERTa 预训练文件放置为：

```text
项目上级目录/pretrain_models/chinese-roberta-wwm-ext/
```

目录内至少应包含 Transformers 可识别的配置和权重文件，例如 `config.json` 与 `pytorch_model.bin` 或 `*.safetensors`。当前工作区未包含该目录，因此没有在本机执行完整训练。

## 4. 运行方式

先做冒烟检查，不保存 checkpoint：

```bash
python bert/train_bert.py --epochs 1 --max_steps 3 --max_eval 300 --no_save
```

完整训练与预测：

```bash
python bert/train_bert.py
python bert/predict_bert.py --n 500
python eval/eval_all.py
```

服务调用：

```bash
python service/classify_cli.py --model roberta --text "贵州茅台：2023 年年度业绩预告" --json
```

如需兼容原命令，`--model bert` 仍可使用，但它只作为参数别名，不会加载 BERT 权重。

## 5. 保持不变的条件

- 训练集、验证集、去重逻辑、固定随机种子和 500 条评测抽样口径不变。
- 15 类标签、标签映射和类别加权方式不变。
- 默认 `pool=cls`、`epochs=3`、`batch_size=64`、模型学习率 `2e-5`、分类头学习率倍数 `5`、最大长度 `64` 不变。
- Warmup 比例、线性衰减、weight decay、AMP、评估指标和按 Macro-F1 保存最佳 checkpoint 的逻辑不变。
- 规则、TF-IDF、Qwen、LoRA 和服务接口字段不因本次模型替换而改变。
