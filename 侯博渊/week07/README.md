# 换数据集作业：DuEE 1.0 上的抽取方法对比

本目录是「第 7 周 · 信息抽取」课后作业的完整交付：**把课程项目整体换到 DuEE 1.0 数据集**，
在同一数据契约、同一评测脚本下重跑规则 / BiLSTM-CRF / BERT / CRF 硬约束 / 本地大模型，
并给出跨数据集对比与错误分析。

- 报告：`REPORT.md`、`作业报告_DuEE10.docx`
- 答辩 PPT：`答辩PPT_DuEE10.pptx`
- 代码：`code/`（课程项目的复制版，只改了数据、规则脚本、prompt 与少量路径/兼容性代码）
- 数据：`data/raw/DuEE_pytorch.zip`（原始下载）、`code/data/processed/`（转换后主实验数据）
- 模型：`pretrain_models/bert-base-chinese`、`pretrain_models/structbert-tiny`

---

## 1. 数据集：DuEE 1.0（不是课程用的 DuEE-Fin）

| 项 | DuEE-Fin（课程原数据） | **DuEE 1.0（本次换的）** |
|---|---|---|
| 来源 | 2021 千言·金融事件抽取 | 2020 千言·事件抽取（P100080/DuEE_pytorch 公开镜像） |
| 领域 | A 股公告 | 开放域新闻/资讯（体育、娱乐、司法、灾害…） |
| 事件类型 | 13 | **65** |
| 类型-角色槽位 / 去重角色 | 92 / 61 | **217 / 121** |
| 原始规模 | 5,747 训练句 / 930 dev | **11,908 句 / 1,492 句（13,860 / 1,783 事件）** |
| 句长（中位） | 161 字 | **35 字**（均 54） |
| 多事件句 | 篇章级 29% | **句子级 13.1%** |
| 无事件句 | 77.9% | **0%（发布数据全是事件句）** |

转换口径：
- **主实验只保留单事件句**（一行 = 一句 = 恰好 1 个事件），与课程评测脚本 `common/evaluate_events.py` 的口径一致；
  多事件句单独存 `dev_multi.jsonl` 做案例分析。
- offset 全部按「先信任标注下标、回读不一致则就近回找」重算，转换后硬校验 `text[start:end] == text`：
  **0 处失败**（1 处论元下标修复）。
- 主实验取样本量前 15 类（覆盖 62% 单事件训练句），标签空间 89，与课程 149 同量级；
  全 65 类版本另存 `code/data/processed_full65/`。

## 2. 方法矩阵（同一份数据、同一套评测）

| 层级 | 方法 | 脚本 | 配置（CPU 预算） | 状态 |
|---|---|---|---|---|
| L0 | 规则：触发词词典 + 时间/数字正则 + jieba 词性实体 | `rule/run_rule.py` | 0 训练 | ✅ |
| L1 | 字符级 BiLSTM-CRF | `bilstm_crf/train.py` | 3 epoch，1.30M 参数 | ✅ |
| L2a | BERT-tiny + Linear | `bert/train.py --decoder linear` | 4 层/256 隐层（8.8M），3 epoch，lr 1e-4 | ✅ |
| L2b | BERT-tiny + CRF（软约束） | `bert/train.py --decoder crf` | 同上 | ⏳ 未跑完（CPU 预算） |
| L2c | BERT-tiny + CRF + 解码期硬约束 | `bert/evaluate_hard_constraint.py` | 不重训，只改解码 | ⏳ 依赖 L2b |
| L3 | 本地 Qwen3-0.6B zero-shot / few-shot | `llm_zero_shot/llm_extract.py` | n=60，seed=42，贪心 | ✅ |

> 说明：课程原主线是 `bert-base-chinese`（102M）。本机只有 CPU（8 逻辑核），实测 BERT-base 4 层
> 训练约 20~26 s/step（一个 epoch 约 1 小时），无法在作业预算内训到收敛；因此主实验改用课程
> week6 的 `structbert-tiny`（中文预训练 BERT，4 层/256 隐层，8.8M），**训练到收敛**后再比较，
> 避免“BERT 欠拟合”造成的假结论。全部配置与预算写在报告 §3。

## 3 结果速览（DuEE 1.0，15 类子集，dev 793）

| 方法 | Trigger F1 | Argument F1 | Event F1 | type_macro | 训练(s) | 延迟(ms/句) |
|---|---|---|---|---|---|---|
| 规则 | 0.7150 | 0.1196 | 0.0191 | **0.8418** | 0 | 0.80 |
| BiLSTM-CRF | **0.7838** | **0.3419** | 0.1167 | 0.8720 | 475 | 5.87 |
| BERT-tiny + Linear | 0.7437 | 0.3320 | 0.1121 | 0.8564 | 1,420 | 14.64 |
| 本地 Qwen3-0.6B zero（n=60） | 0.2667 | 0.0316 | 0.0000 | 0.4712 | 0 | 28,123 |
| 本地 Qwen3-0.6B few（n=60） | 0.3761 | 0.1449 | 0.0513 | 0.3930 | 0 | 39,912 |

同一批 n=60 子集上：规则 .7797/.1161/0、BiLSTM .7407/.3024/.0926、BERT-Linear .7636/.3588/.1091。
完整分析见 `REPORT.md`。

## 3. 一键复现

```powershell
cd code

# 0) 数据：从原始 zip 重建 processed（含统计与 span 校验）
python data/scripts/build_duee10.py --max_types 15 --out_dir data/processed --stats_name duee10_stats_main15.json

# 1) 规则
python rule/run_rule.py

# 2) BiLSTM-CRF（CPU 约 8 分钟）
python bilstm_crf/train.py --decoder crf --epochs 3 --max_len 160 --device cpu --run_name bilstm_crf

# 3) BERT-tiny + Linear / CRF
python bert/train.py --decoder linear --bert_path ..\pretrain_models\structbert-tiny --max_len 160 \
    --epochs 3 --batch_size 32 --lr 1e-4 --device cpu --run_name berttiny_linear
python bert/train.py --decoder crf --bert_path ..\pretrain_models\structbert-tiny --max_len 160 \
    --epochs 3 --batch_size 32 --lr 1e-4 --device cpu --run_name berttiny_crf

# 4) 解码期硬约束（复用 CRF checkpoint，不重训）
python bert/evaluate_hard_constraint.py --ckpt outputs/checkpoints/berttiny_crf.pt \
    --run_name berttiny_crf_hard --soft_json berttiny_crf

# 5) 本地 Qwen3-0.6B zero/few-shot（CPU，n=100）
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 60 --gen_batch 8 --tag local_zero
python llm_zero_shot/llm_extract.py --backend local --mode few  --n 60 --gen_batch 8 --tag local_few

# 6) 汇总 + 图 + badcase
python eval/compare_duee10.py
python eval/badcase_duee10.py --pred outputs/disc/pred_berttiny_crf_hard.jsonl
```

所有指标写入 `code/outputs/metrics/`，预测写入 `code/outputs/disc/` 与 `code/outputs/llm/`，
图写入 `code/outputs/figures/`，汇总表写入 `code/outputs/report/`。

## 4. 与课程原项目的差异（改了什么）

1. 数据：`code/data/processed/` 换成 DuEE 1.0（`event_schema.json` + `train/dev_extract.jsonl`）。
2. `rule/run_rule.py`：触发词表改为纯统计（65 类不手写同义词）；论元从“金融公司词典”改为
   “正则 + jieba 词性（nr/ns/nt）”，适配开放域。
3. `llm_zero_shot/event_prompt.py`：system prompt 去掉“金融公告/13 类”硬编码；few-shot 示例改为
   **从 train 动态选取**（长尾类型各 1 例），换数据集自动跟随。
4. `llm_zero_shot/llm_extract.py`：新增 `--data_dir`。
5. `bert/*.py`：`BertTokenizer` → `BertTokenizerFast`（本机 transformers 4.55 下 fast tokenizer 才有 `word_ids()`）。
6. `bert/evaluate_hard_constraint.py`：新增 `--data_dir`。
7. `eval/compare_duee10.py`、`eval/badcase_duee10.py`、`eval/dataset_profile.py`：本作业新增的汇总/分析脚本。

其余训练与评测逻辑（`common/event_data.py`、`common/evaluate_events.py`、`common/crf.py`）**未改动**，
保证与课程同口径。
