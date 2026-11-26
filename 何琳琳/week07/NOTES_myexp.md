# NOTES —— week7 事件抽取（本地 CPU 复现与多骨干对比）

> 对应老师的 `NOTES_discriminative.md`，章节骨架一致，内容全部是本机实跑。
>
> **数据来源约定**：未标注者 = 本机 CPU 实测；标"老师（GPU）"者 = 老师原件，**只用于 §4.1/§4.2 校准与天花板参照，不参与排名、不与本机数字做减法**。整份文件唯一的跨来源减法已改为"老师对老师"（§4.3 的 0.5630 − 0.3495）。
> 数字口径：`outputs/metrics/disc/<run_name>.json`（同一份 `common/evaluate_events.py`）。
> 硬件：Windows + **纯 CPU**（`peak_gpu_mb` 全程 0）。

## 0. 一分钟结论（PPT 首页可用）

| # | 结论 | 数字 |
|---|---|---|
| 1 | 双向复现老师，评测链路可信 | BERT 侧论元差 **0.0006**；BiLSTM 侧差 **0.25 点** |
| 2 | 训练量是硬门槛，不是调参 | 69 步 → 486 步：论元 F1 **0.0000 → 0.3489** |
| 3 | 砍层数是伤害最大的一刀 | 老师同环境 12 层 0.5630 → 4 层 **0.3495**，**−21.35 点**；本机 4 层 0.3489 与之只差 0.0006 |
| 4 | 换领域（+9.0）远优于换架构（−15.6） | FinBERT 0.4390 / RoBERTa 0.1927 |
| 5 | 性价比冠军是从零训练的 BiLSTM-CRF | 1.31M / 300s / 3.92ms，论元 **0.4609** |
| 6 | 长尾：救得了 30 例，救不了 11 例 | `股东增持` 31 例 0.8966；`被约谈` 11 例 0.0000 |
| 7 | **CRF 买精度，硬约束买合法性** | CRF 论元 +3.51（非法仅 −1.5）；硬约束 0 训练：论元 **+14.46**、非法 84.6% → **6.1%** |

**一句话选型**：精度最高 → 4 层 + CRF + 硬约束（0.5286）；性价比 → BiLSTM-CRF（0.4609 / 1.31M / 300s / 3.92ms）；类型与召回优先 → FinBERT-4L；要天花板 → 12 层或 LLM SFT；只做事件发现 → 4 层 + max_len 128。

## 1. 目录与脚本职责

本次没有新写训练代码，全部复用老师工程：**新增 2 个脚本、修改 2 个脚本**（修改处都在文档里标了"已改"）：

| 文件 | 职责 |
|---|---|
| `bert/train.py` | E05~E08：BERT-4L + Linear，`--bert_path` 换骨干、`--max_len` 换截断 |
| `bilstm_crf/train.py` | E09：字符级 BiLSTM-CRF（从零，`--decoder crf`） |
| `rule/run_rule.py` | E01：词典 + 正则基线 |
| `eval/compare_methods.py` | 汇总表；**已在 `ABLATION_ORDER` 里补了 4 个本机 run_name** |
| `eval/make_figures.py` | 老师原版 6 张图（修了 `--curve_runs` 的参数解包 bug、非法 BIO 标题写死的问题） |
| **`eval/make_figures_myexp.py`** | **新增**：按 §8.3 清单把 6 张图重做成本机实验版 |
| **`eval/fig_backbone_compare.py`** | **新增**：骨干对比 + 成本-效果散点（报告专用） |
| `bert_pipeline/train.py` | E12（**本轮不做**，理由见 EXPERIMENTS §六）：两段式「检测 → 13 类分类 → 类型条件论元专家」，共 16 个模型 |
| `service/demo_extract.py` | 篇章级端到端演示（985 篇 `dev_docs.jsonl` 带金标，输出自带【与金标对照】；含正则兜底，**不参与 F1 评测**） |
| `EXPERIMENTS.md` | 人类实验日志（每条 6 项：run_name/命令/配置/结果/判断/决策） |

`outputs/` 目录约定（与老师一致）：`metrics/disc/` 成绩单、`logs/` 三类训练日志、`disc/` 逐句预测、`figures/` 6 张图、`checkpoints/` 权重。

## 2. 全部命令（按执行顺序，可直接复制）

```powershell
# ── 0. 每次都要先进入工程根，否则 bert\train.py 找不到、相对路径全错位 ──
# 进入解压出来的 code 目录（我本机是 D:\NLPEIGHT\homework\week7\code，请换成你的位置）
cd <解压目录>\code
#  下面统一写 python（PyCharm 终端已激活 py312）。若在普通 PowerShell 里跑，换成
#  & "D:\software\Anaconda3\envs\py312\python.exe"，否则会 ModuleNotFoundError: No module named 'numpy'。

# ── 1. L0 规则基线（秒级，纯 CPU）────────────────────────────────
python rule\run_rule.py
#   → outputs/metrics/disc/rule.json

# ── 2. 预训练权重准备 ──────────────────────────────────────
# 【位置】三个权重必须放在工程根目录下的 pretrain_models\（与 data\、outputs\ 同级），
#        因为后文所有 --bert_path 都是相对路径 pretrain_models\xxx。
# 【为何 zip 里没有】该目录 3.6GB，超出 GitHub 单文件 100MB 上限；且 bert-base-chinese
#        内部含 .git（会被识别成子模块），故有意排除，需按下述命令自备。
# 【不用手建目录】下面的 snapshot_download 与 New-Item -Force 会自动创建目录（已实测）。
# 期望结构：code\pretrain_models\{bert-base-chinese, roberta-base, finbert-zh}\
#            每个目录内应含 config.json + pytorch_model.bin
#
# 以下命令在 code 目录内执行；<CACHE> 换成你自己的 modelscope 缓存目录（我本机用的是 D:\modelscope_cache）

#   ① bert-base-chinese（HF 官方权重，392MB，走 LFS）
python -c "from huggingface_hub import snapshot_download; snapshot_download('google-bert/bert-base-chinese', local_dir=r'pretrain_models\bert-base-chinese')"
#      不要用 git clone：该仓库大文件走 git-lfs，未装 lfs 时只会拿到几十字节的指针文件，
#      训练时报权重加载失败。snapshot_download 会正确下实体。
#      若手头已有现成权重，直接复制到 pretrain_models\bert-base-chinese 也可，无需联网

#   ② roberta-base（来源：modelscope iic/nlp_roberta_backbone_base_std）
modelscope download --model iic/nlp_roberta_backbone_base_std --cache_dir <CACHE>
New-Item -ItemType Directory -Force pretrain_models\roberta-base | Out-Null
Copy-Item -Recurse "<CACHE>\models\iic--nlp_roberta_backbone_base_std\snapshots\master\*" pretrain_models\roberta-base

#   ③ finbert-zh（来源：modelscope qolaris/FinBert）
modelscope download --model qolaris/FinBert --cache_dir <CACHE>
New-Item -ItemType Directory -Force pretrain_models\finbert-zh | Out-Null
Copy-Item -Recurse "<CACHE>\models\qolaris--FinBert\snapshots\master\*" pretrain_models\finbert-zh

# ── 3. 验词表（决定可比性口径）────────────────────────────────────
python -c "from transformers import AutoTokenizer,AutoConfig; p=r'pretrain_models\finbert-zh'; print(AutoTokenizer.from_pretrained(p).vocab_size); c=AutoConfig.from_pretrained(p); print(c.model_type,c.num_hidden_layers,c.hidden_size)"
#   → 21128 / bert / 12 / 768  ⇒ 与 bert-base-chinese 完全同构，严格可比

# ── 4. 阶段 A：800 句子集，诊断 F1=0（E02~E04，三次）───────────────
python bert\train.py --data_dir data\myexp --bert_layers 4 --max_len 128 --epochs 1 --decoder linear --run_name bert4L_linear_128
python bert\train.py --data_dir data\myexp --bert_layers 4 --max_len 128 --epochs 3 --decoder linear --run_name bert4L_linear_128_3e
python bert\train.py --data_dir data\myexp --bert_layers 4 --max_len 256 --epochs 3 --decoder linear --run_name bert4L_linear_256_3e

# ── 5. 阶段 B：全量 5747，正式基线（E05，环境校准）─────────────────
python bert\train.py --data_dir data\processed --bert_layers 4 --max_len 256 --epochs 3 --decoder linear --run_name bert4L_linear_256_full_data

# ── 6. 换骨干：RoBERTa（E06）──────────────────────────────────────
python bert\train.py --data_dir data\processed --bert_path pretrain_models\roberta-base --bert_layers 4 --max_len 256 --epochs 3 --decoder linear --run_name roberta4L_linear_256_full_data

# ── 7. 截断消融（E07，只改 max_len）───────────────────────────────
python bert\train.py --data_dir data\processed --bert_layers 4 --max_len 128 --epochs 3 --decoder linear --run_name bert4L_linear_128_full_data

# ── 8. 换领域：FinBERT（E08）──────────────────────────────────────
python bert\train.py --data_dir data\processed --bert_path pretrain_models\finbert-zh --bert_layers 4 --max_len 256 --epochs 3 --decoder linear --run_name finbert4L_linear_256_full_data

# ── 9. 从零训练：BiLSTM-CRF（E09，照抄老师原件超参）────────────────
python bilstm_crf\train.py --data_dir data\processed --decoder crf --max_len 128 --epochs 3 --emb_dim 128 --hidden 256 --num_layers 1 --lr 1e-3 --run_name bilstm_crf_full_data --device cpu

# ── 10. 汇总 + 出图 ───────────────────────────────────────────────
python eval\compare_methods.py        # → outputs/metrics/disc/summary.md
python eval\make_figures_myexp.py     # → 6 张图（本机实验版）
python eval\fig_backbone_compare.py   # → 骨干对比 + 成本散点

# ── 11.（E10/E10b 已完成 11-25；E12 可选、未跑）────────────────────
python bert\train.py --data_dir data\processed --bert_layers 4 --max_len 256 --epochs 3 --decoder crf --run_name bert4L_crf_256_full_data   # E10 解码器 → 0.3840
python bert\evaluate_hard_constraint.py --ckpt outputs\checkpoints\bert4L_crf_256_full_data.pt --run_name bert4L_crf_256_full_data_hard --soft_json bert4L_crf_256_full_data --device cpu   # E10b → 0.5286
python bert_pipeline\train.py --device cpu --arg_epochs 3 --cls_epochs 2                                                              # E12 两段式：**本轮不做**，命令仅备查

# ── 12. 端到端篇章演示（985 篇带金标，自带【与金标对照】；不进 F1 评测）──
python service\demo_extract.py --n 4 --joint_ckpt outputs\checkpoints\finbert4L_linear_256_full_data.pt --device cpu
```

> 串行执行，**不要并行**：CPU 跑满会互相抢核，两组同时跑的时间大于分别跑。

## 3. 关键设计决策

### 3.1 为什么必须先做"双向校准"再比方法

对比实验最大的风险是"分数不准，结论全错"。只校预训练模型会掩盖标签/评测问题；只校从零模型会掩盖权重加载问题（E06 就差点被误判成"权重没加载"）。所以两侧各校一次（§4.1、§4.2），差值分别 0.0006 与 0.25 点，之后所有差异才能归因给被改的那个变量。

### 3.2 为什么 4 层要"加载完再切片"，不能用 `BertConfig(num_hidden_layers=4)`

`BertConfig(num_hidden_layers=4)` 会让 `from_pretrained` 只认前 4 层的 key，**其余层随机初始化**，等于丢掉预训练。正确做法（`common/models.py` 41~52 行）：先 `BertModel.from_pretrained()` 装满 12 层，再 `encoder.layer = ModuleList(list(...)[:4])`，只在内存里改 config。验证方式：ckpt 里 `encoder.layer.*` 的最大层号必须是 3。

### 3.3 为什么成本列一律用本机数字

老师那组 BiLSTM 是 1520.1s / 20.56ms，本机是 300s / 3.92ms，**差 5 倍且两边 `peak_gpu_mb` 都为 0** → 纯硬件差异。跨机比成本会系统性低估小模型优势，所以成本只在本机内部横比。

### 3.4 为什么训练曲线里不放 BiLSTM

CRF 的 train_loss 是负对数似然的序列累加（230 → 49），softmax-CE 是每 token 平均（1.7 → 0.45），**量级差两个数量级**。放同一面板会把其余 5 条曲线压成直线。两条曲线各自看趋势，但不可比大小。

### 3.5 为什么 `run_name` 不叫 `bilstm_crf`

会覆盖老师原件 `metrics/disc/bilstm_crf.json` 与 `checkpoints/bilstm_crf.pt`，之后就无法再核对"我复现得准不准"。

### 3.6 为什么用 `evaluate_hard_constraint.py`，而不是 `bert/train.py --hard_constraint`

两个东西名字像、机制不同：

| | `train.py --hard_constraint` | `evaluate_hard_constraint.py` |
|---|---|---|
| 什么时候屏蔽非法路径 | **训练期**（loss 里非法转移概率恒为 0） | **解码期**（权重不动，viterbi 前置 `-inf`） |
| 成本 | 要重训一整轮，4 层约 95 分钟 | **0 训练**，80 秒复评 |
| 与老师可比性 | 老师没有这个训练组 | 与老师 `bert_crf_256_hard` 同口径 |

所以选后者：既省 95 分钟，又能把"收益全部来自解码策略而非训练"这个结论说干净。**报告里凡说"0 训练成本"，指的都是这条路径。**

## 4. 实测结果

### 4.1 校准一：BERT 侧（E05 vs 老师 `bert4L_linear_256`）

| 指标 | 本机 CPU | 老师 GPU | 差值 |
|---|---|---|---|
| 触发词 F1 | 0.7846 | 0.7793 | +0.5 点 |
| **论元 F1** | **0.3489** | **0.3495** | **−0.06 点** |
| 事件级 F1 | 0.0599 | 0.0535 | +0.6 点 |
| 类型 macro | 0.7781 | 0.7617 | +1.6 点 |
| 非法序列率 | 86.1% | 86.67% | −0.6 点 |
| 内层验证 spanF1 | 0.4280 | 0.4251 | +0.3 点 |
| best_epoch | 3 | 3 | 同 |
| 推理延迟 | 79.49 ms | 7.31 ms | 慢 10.9× |
| 训练耗时 | 5022 s | 259 s | 慢 19.4× |

### 4.2 校准二：无预训练侧（E09 vs 老师 `bilstm_crf`）

| 指标 | 老师原件 | 本机复现 | 差值 |
|---|---|---|---|
| 触发词 F1 | 0.8165 | 0.8057 | −1.08 |
| **论元 F1** | 0.4634 | **0.4609** | **−0.25** |
| 事件级 F1 | 0.0885 | 0.0825 | −0.60 |
| 类型 macro | 0.7306 | 0.7206 | −1.00 |
| 非法序列率 | 48.82% | 47.90% | −0.92 |
| 训练耗时 | 1520.1 s | **300 s** | −1220 |
| 推理延迟 | 20.56 ms | **3.92 ms** | −16.6 |

**结论：精度两侧都在统计噪声内；差异只出现在速度。**

### 4.3 主表（dev_extract 930 句，统一口径）

| 方法 | 触发词 F1 | 论元 F1 | 事件 F1 | 类型 macro | 参数 M | 训练 s | 延迟 ms | 非法率 |
|---|---|---|---|---|---|---|---|---|
| **BERT-4L + CRF + 硬约束**（E10b） | **0.8121** | **0.5286** | **0.1165** | 0.7340 | 45.7 | **0** | 80.21 | **6.1%** |
| BiLSTM-CRF（从零，128） | 0.8057 | 0.4609 | 0.0825 | 0.7206 | **1.31** | **300** | **3.92** | 47.9% |
| FinBERT-4L + Linear（256） | 0.8067 | 0.4390 | 0.0791 | **0.8109** | 45.7 | 4894 | 78.50 | 84.3% |
| BERT-4L + **CRF**（256，E10） | 0.7893 | 0.3840 | 0.0672 | 0.7569 | 45.7 | 5669 | 84.49 | 84.6% |
| BERT-4L + Linear（256） | 0.7846 | 0.3489 | 0.0599 | 0.7781 | 45.7 | 5022 | 79.49 | 86.1% |
| BERT-4L + Linear（128） | 0.7971 | 0.3253 | 0.0520 | 0.8015 | 45.7 | 2448 | 40.03 | 85.3% |
| RoBERTa-4L + Linear（256） | 0.6990 | 0.1927 | 0.0012 | 0.5688 | 45.7 | 4977 | 81.37 | 88.2% |
| L0 规则 | 0.6168 | 0.1740 | 0.0161 | 0.8118 | — | — | 1.12 | — |

> 按论元 F1 降序。E10b 的"训练 0"是复用 E10 权重做解码期后处理，真实成本记在 E10 的 5669s。

> 天花板参照：老师 12 层 + CRF + 硬约束 = 论元 **0.6566** / 事件 0.2177（**非本机跑出**，只作高度参照）。完整六行表（12L linear / 12L CRF / 12L+硬约束 / 4L / 两段式 / oracle）见报告**附录 C**，本文件不重复维护，也不与本机数字混算。

### 4.4 训练量敏感性（E04 vs E05，只差数据量）

| run_name | 训练句 | 步数 | 末轮 train_loss | 内层 spanF1 | 论元 F1 |
|---|---|---|---|---|---|
| `bert4L_linear_256_3e` | 720 | 69 | 1.2626 | 0.0000 | 0.0000 |
| `bert4L_linear_256_full_data` | 5173 | 486 | 0.4498 | 0.4280 | 0.3489 |

**这是"F1=0 不是 bug"的直接证据**：除训练量外全部相同。另用 logit margin 复核（6830 个金标非 O 位置，`O 得分 − 最高非 O 得分` 中位 **+5.54**，负值 **0.00%**）—— 模型是"毫无倾向"，不是"差一点"。

### 4.5 截断消融（E07 vs E05，只差 max_len）

| 指标 | 128 | 256 | 差值 |
|---|---|---|---|
| 触发词 F1 | **0.7971** | 0.7846 | **+1.25** |
| 论元 F1 | 0.3253 | **0.3489** | **−2.36** |
| 事件级 F1 | 0.0520 | **0.0599** | −0.79 |
| 类型 macro | **0.8015** | 0.7781 | **+2.34** |
| 训练耗时 | **2448s** | 5022s | **−51.3%** |
| 推理延迟 | **40.03ms** | 79.49ms | **−49.6%** |

触发词集中在标题/首句，截断伤不到；论元散落全篇、常在文末，128 丢掉 36.1% 的论元。第 1 轮内层 spanF1 两者几乎相同（0.2364 vs 0.2338）→ 差距是**信息上限低**，不是学得慢。

### 4.6 骨干对比（E06 / E08 vs E05）

| 骨干 | 论元 F1 | 事件 F1 | 相对 BERT | 性质 |
|---|---|---|---|---|
| bert-base-chinese | 0.3489 | 0.0599 | 基准 | 通用 |
| iic roberta backbone | 0.1927 | 0.0012 | **−15.6** | 通用，换架构 |
| **FinBERT-zh** | **0.4390** | **0.0791** | **+9.0** | 同架构，换领域 |

FinBERT 的提升是真提升：`tp 1641→2138（+30.3%）` 而 `fp 3644→3481`（反而减少）；且第 1 轮起步就领先（内层 0.3007 vs 0.2338）。**召回驱动**：R 0.3981→0.5187，P 只从 0.3105 到 0.3805。

**RoBERTa 的正确表述**：486 步统一预算下收敛慢一个数量级（第 1 轮内层 spanF1 0.0019 vs 0.2338，差 100 倍），**不具竞争力，需更长预算才能公平比较**；不得写成"RoBERTa 劣于 BERT"。

### 4.7 从零训练（E09 vs 两个 4 层 Linear BERT）

| | BiLSTM-CRF | FinBERT-4L | BERT-4L |
|---|---|---|---|
| 论元 F1 | **0.4609** | 0.4390 | 0.3489 |
| 参数 | **1.31M** | 45.7M（35×） | 45.7M（35×） |
| 训练 | **300s** | 4894s（16×） | 5022s（17×） |
| 延迟 | **3.92ms** | 78.5ms（20×） | 79.5ms（20×） |
| 类型 macro | 0.7206 | **0.8109** | 0.7781 |
| 非法率 | **47.9%** | 84.3% | 86.1% |

1.31M 参数、5 分钟、每句 4 毫秒 → 论元 F1 还高 2.2 点。**原因**：完整结构（**1 层双向 LSTM**，前向/后向各 256 维 → 512 维，`--num_layers 1`；全序列 + CRF 转移约束）对上"残废结构"（预训练的前 4 层）。

**P/R 性格相反**：BiLSTM 高精低召（P 0.5893 / R 0.3785），FinBERT 低精高召（P 0.3805 / R 0.5187）。触发词上 BiLSTM 的 P 更是 **0.9071**（930 句只误报 69）。

**但它不是精度冠军**（11-25 新增 E10b 后）：4 层 + CRF + 硬约束论元 **0.5286**，比它高 6.77 点（§4.12）。BiLSTM-CRF 赢的是**性价比**——1/35 参数、1/19 训练时间、1/20 延迟。

### 4.8 非法 BIO 因果链

| 方法 | 含非法序列 | 成因 |
|---|---|---|
| 金标自身（重叠论元投影） | **18 条（1.9%）** | 不可消除的地板值（重叠论元投影） |
| BERT-4L 256 + Linear | 801 条（86.1%） | 逐 token 独立 softmax，无转移约束 |
| BERT-4L 128 + Linear | 793 条（85.3%） | 同上 |
| RoBERTa-4L + Linear | 820 条（88.2%） | 同上 |
| FinBERT-4L + Linear | 784 条（84.3%） | 同上 |
| **BERT-4L 256 + CRF（E10）** | **787 条（84.6%）** | 软约束学的是数据统计，而训练集 86.5% 本身非法 → 几乎无效 |
| **BiLSTM-CRF** | **445 条（47.9%）** | CRF 转移矩阵收敛充分，但**不约束首标签**（47 处非法开头） |
| **BERT-4L + 硬约束（E10b）** | **57 条（6.1%）** | 解码期注入 10878 个禁止转移 + 74 个非法起始 |
| 老师 12 层 + 硬约束复评 | **7.4%** | 同机制（0 训练成本） |

> 重要提醒：**"非法 0 条"不是好事，但也不能反推"有非法=模型在工作"**。E03/E04 是彻底全 `O`（fp=0、非法 0 条，天然合法）；而 E02 还没完全塌——报了 34 个错触发词、100 句非法（10.75%）。**判据是"有没有真的标对"，不是"非法率是多少"。**

### 4.9 逐角色论元 F1（BiLSTM-CRF，dev 出现 ≥20 次的角色里最难的 18 个）

最差 7 个角色 F1 **全为 0.000**：`高管职位`(37) / `高管姓名`(46) / `增持方`(34) / `破产公司`(30) / `变动类型`(42) / `质权方`(21) / `领投方`(24)。

→ 与 §4.10 呼应：`高管变动` 类整体只有 0.4262，就是被这三个角色拖垮的。

### 4.10 类型长尾（13 类逐项）

| 事件类型 | dev 实例 | BERT-4L | FinBERT-4L | BiLSTM-CRF |
|---|---|---|---|---|
| 股份回购 | 178 | 0.9803 | 0.9915 | 0.9605 |
| 中标 | 116 | 0.9735 | 0.9825 | 0.9358 |
| 企业收购 | 116 | 0.9339 | 0.9386 | 0.9140 |
| 亏损 | 106 | 0.9569 | 0.9717 | 0.9519 |
| 股东减持 | 83 | 0.8263 | **0.9024** | 0.7826 |
| 公司上市 | 58 | 0.8119 | **0.8627** | 0.7708 |
| 质押 | 57 | 0.7759 | 0.7222 | 0.7216 |
| 企业融资 | 53 | 0.8776 | **0.9200** | 0.8421 |
| 高管变动 | 48 | 0.7368 | 0.7952 | **0.4262** |
| 解除质押 | 40 | 0.6389 | 0.6329 | 0.5625 |
| 企业破产 | 33 | 0.9254 | 0.9254 | 0.9091 |
| **股东增持** | **31** | **0.5116** | **0.8966** | 0.5909 |
| **被约谈** | **11** | **0.1667** | **0.0000** | **0.0000** |

**分层结论**：≥100 实例谁都能学（全部 >0.91）；30~60 实例出现分化（FinBERT 在 31 例的 `股东增持` 拿到 0.8966，是 BERT 的 1.75 倍）；**11 例的 `被约谈` 三个模型全废 —— 领域预训练救得了 30 例，救不了 11 例。**

### 4.11 类型混淆（FinBERT-4L）

两个真实混淆：

- **`解除质押` ↔ `质押` 互混**：dev 上 `解除质押` 40 例、`质押` 57 例；**剔除被检测器判成"无事件"的句子后**（各 4 句），混淆矩阵里 `解除质押` 36 句中有 **11 句判成 `质押`**，`质押` 53 句中有 **13 句判成 `解除质押`**（另 1 句判成 `股东减持`）。这是两类 F1 只有 0.63 / 0.72 的根因——差在"解除"两个字有没有被抓住。
- `企业融资` → `企业破产` 2 条、`企业收购` → `企业融资` 2 条：资金类事件边界模糊。

### 4.12 解码器与后处理（E10 → E10b）

同一份 4 层权重，只换解码策略：

| 阶段 | 论元 F1 | 事件 F1 | 非法序列 | 成本 |
|---|---|---|---|---|
| Linear（E05） | 0.3489 | 0.0599 | 86.1% | 5022s |
| **+CRF（E10）** | 0.3840（**+3.51**） | 0.0672 | **84.6%（−1.5）** | 5669s（+13%） |
| **+CRF+硬约束（E10b）** | **0.5286（+14.46）** | **0.1165** | **6.1%（−78.5）** | **0（80 秒复评）** |

三条因果：

1. **CRF 压不住非法**：训练集 4476/5173 = 86.5% 的序列本身非法（重叠论元投影），软约束拟合的是数据统计，不是 BIO 规则。老师 12 层同样只从 66.88% 削到 62.47%。
2. **硬约束能归零**：把 10878 个禁止转移对 + 74 个非法起始标签的概率置 `-inf`，在合法空间重跑 viterbi——注入规则，不依赖训练。
3. **收益与基线非法率正相关**：老师基线 62.5% → +7.5；本机基线 84.6% → **+14.5（1.9 倍）**。

另：`type_macro` 从 0.7781 掉到 0.7569（再降到 0.7340），因为转移约束偏好高频路径，会把稀有类型往常见类型拉平。

**结论：0.5286 是本机最高论元 F1，已超过 BiLSTM-CRF（0.4609，+6.77）与 FinBERT（0.4390）；但仍低于老师 12 层 linear（0.5630，**跨来源天花板参照，只比高低不做减法**）→ 4 层的信息上限没被后处理完全补回。**

## 5. 数据事实（按实测为准，与任务描述有出入时以这里为准）

| 事实 | 实测值 | 影响 |
|---|---|---|
| 标签体系 | 149 = `O` + 2×（13 触发 + 61 论元角色） | 逐 token 149 选 1 分类，不是句分类 |
| 句子长度 | train 中位 166 / p90 270；dev 中位 161 | max_len 128 截断 69.6% 的句子 |
| 13 类实例分布 | dev 178 ~ 11（**差 16 倍**） | 每类 <50 必崩 |
| 金标自身非法 BIO | **18 / 930 句（1.9%）** | 不可消除的地板值（重叠论元投影） |
| 训练集非法对照（E05 实测） | **4566 / 5173 句（88.3%）** | 模型在 dev 上 86.1%，比训练集略低 → **与数据同分布，不是模型 bug**（CRF 那组训练集 4476 句 86.5%） |
| dev 与评测集 | dev_extract 930 句，两个阶段同一份 | 阶段 A/B 分数可横向比 |

## 6. 踩坑记录（按教学价值排序）

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **工作目录错位** | `python bert\train.py` 报 `can't open file`，相对路径全错 | 每次先 `cd D:\NLPEIGHT\homework\week7\code`（PyCharm 终端默认在 `D:\NLPEIGHT\homework`） |
| 2 | **F1=0 被误判为 bug** | 三次全 0，中间验证也 0 | 用"同配置换数据量"隔离：69 步 → 486 步，0 → 0.3489；再查 `n_truncated` 与 logit margin |
| 3 | **`best_f1` 初始化** | 全 0 时 `0 > 0` 恒假，只保存第 1 轮 | 人工确认 `best_epoch`，别信保存次数 |
| 4 | **4 层加载方式** | 用 `BertConfig(num_hidden_layers=4)` 会随机初始化后 8 层 | `from_pretrained` 装满 12 层后 `encoder.layer[:4]` 切片，只在内存改 config |
| 5 | **`LOAD REPORT` 的 UNEXPECTED** | RoBERTa 加载时 8 个 `cls.*`、2 个 `LayerNorm.*` 标 UNEXPECTED | 是预训练 MLM 头 + 转换冗余 key，不是缺权重；用 199 个 `bert.*` key 对齐 + 首步 loss 5.0573 vs 5.1130 证明 |
| 6 | **跨机比成本** | 老师 BiLSTM 1520s、本机 300s | 两边 `peak_gpu_mb` 都为 0 → 纯硬件差异，成本只在本机内横比 |
| 7 | **CRF loss 不可比** | BiLSTM train_loss 230 → 49，BERT 1.7 → 0.45 | CRF 是序列 NLL 累加，只看趋势不看绝对值 |
| 8 | **训练曲线混模型** | 把 BiLSTM 塞进 loss 面板，其余 5 条被压成直线 | 曲线只放 BERT 系列；BiLSTM 单独作图 |
| 9 | **`--curve_runs` 参数 bug** | 一指定就 `too many values to unpack` | 老师原实现直接把字符串当二元组解包，已改成两两配对 |
| 10 | **图标题写死** | 非法 BIO 图标题写"→ 硬约束归零"，但未跑硬约束 | 改成按 `hard_key` 动态拼接 |
| 11 | **`Copy-Item -Recurse 源\* 目标`** | 目标不存在时报错 | 先 `New-Item -ItemType Directory -Force` |
| 12 | **modelscope 缓存落 C 盘** | 默认缓存不在 D 盘 | 必须带 `--cache_dir D:\modelscope_cache`；目录名格式 `组织--模型名` |
| 13 | **PowerShell 未激活 conda** | `import numpy` 失败（`No module named 'numpy'`） | 用环境内 python 绝对路径 `& "D:\software\Anaconda3\envs\py312\python.exe"` |
| 14 | **`run_name` 覆盖老师原件** | 用 `bilstm_crf` 跑会把老师 json/pt 覆盖 | 一律加 `_full_data` 后缀 |
| 15 | **非法 0 条被当成好结果** | E03/E04 是 0 条（真全 `O`），E02 却是 100 句 10.75% | 全 `O` 天然合法；判据看"有没有标对"，不看非法率高低（见 §4.8） |

## 7. 待补 / 已知局限

| # | 项 | 状态 |
|---|---|---|
| 1 | LLM zero/few-shot、LoRA SFT | **未做**（无 API key，CPU 跑不动本地大模型）。老师参照（n=200 子集，非本机）：API few-shot 论元 0.535、LoRA 0.697 |
| 2 | 多组实验未收敛 | E05/E07/E08/E09 全部 `best_epoch=3`，3 轮仍在涨；"谁更好"成立，"差距多大"会随预算变 |
| 3 | 硬约束只验证过 BERT-CRF 一路 | BERT-4L CRF（本机）与 12 层 CRF（老师）都验证了；FinBERT + CRF + 硬约束、BiLSTM + 硬约束未跑。规则由标签 schema 生成，换数据集必须重生成 |
| 4 | 单种子 | 全部 seed=42，±1~2 点不可当结论 |
| 5 | 硬约束的残余非法不是 0 | `bert4L_crf_256_full_data_hard.json`：0 训练、论元 0.3840→**0.5286（+14.46）**、非法 **6.13%**（57/930，转移 115 处）。残余来自字符级 span 与 token 边界不完全对齐——"归零"只发生在 token 级转移，不是端到端 0 非法 |
| 6 | 4 层实验的 max_len 与 BiLSTM 不同 | E09 用 128（照抄老师），BERT 系列 256 |
| 7 | 两段式流水线（E12）未做 | 主动放弃：方向已被老师实测锁定（论元 0.3690 < 联合 0.5630，检测器丢 27.4% 句子），且它不属于四个主变量层。理由全文见 `EXPERIMENTS.md §六` |

## 8. PPT 可用的数字与金句

### 8.1 主表（dev_extract 930 句，统一口径）

见 §4.3，直接用。若 PPT 只能放 4 行：

| 方法 | 论元 F1 | 参数 | 训练 | 延迟 |
|---|---|---|---|---|
| BiLSTM-CRF | **0.4609** | **1.31M** | **300s** | **3.92ms** |
| FinBERT-4L | 0.4390 | 45.7M | 4894s | 78.5ms |
| BERT-4L 256 | 0.3489 | 45.7M | 5022s | 79.5ms |
| RoBERTa-4L | 0.1927 | 45.7M | 4977s | 81.4ms |

### 8.2 可直接引用的金句（每条都对应上面实测）

1. **"69 步到 486 步，0 到 0.3489 —— F1=0 不是 bug，是数据量。"**（§4.4）
2. **"1.31M 参数、5 分钟、每句 4 毫秒，论元 F1 打赢 45.7M、80 分钟、每句 79 毫秒的 FinBERT 2.2 点。"**（§4.7）
3. **"换领域语料 +9.0，换架构 −15.6：小预算下该换语料，不该换架构。"**（§4.6）
4. **"12 层砍到 4 层掉 21.35 点（同一环境的两行做减法），是换领域语料赚钱的 2.4 倍。"**（§4.3）
5. **"领域预训练救得了 30 例（股东增持 0.51→0.90），救不了 11 例（被约谈三个模型全 0）。"**（§4.10）
6. **"0.0006 和 0.25：预训练侧与从零侧都复现了老师，之后所有差异才能归因给被改的那个变量。"**（§4.1、§4.2）
7. **"非法 BIO 0 条不是好事——E03/E04 的确是 0 条，因为模型全输出 `O`；E02 还有 34 个错触发词。"**（§4.8）
8. **"触发词精确率 90.7 只误报 69 处，FinBERT 召回 51.9 却多报 3481 处 —— 一个是风控模型，一个是挖掘模型。"**（§4.7）
9. **"解除质押和质押互混 11/13 个，类型 F1 0.63 的根源差在'解除'两个字。"**（§4.11）
10. **"CRF 只买到 3.5 分，硬约束买到 14.5 分 —— 合法性不是训练出来的，是规则规定的。"**（§4.12）

### 8.3 图表清单（`outputs/figures/`，全部 150dpi / 白底 / 中文字体）

> 与老师 `NOTES_discriminative.md §8.3` 同一份清单，内容已换成本机实验：
> 重做命令 `python eval\make_figures_myexp.py`

| 文件 | 内容 | 对应 PPT |
|---|---|---|
| `fig_method_compare.png` | 方法对比（trigger/argument/event 三级并排，**8 组**：规则 / BiLSTM / 4L256 / 4L128 / RoBERTa / FinBERT / 4L+CRF / 4L+CRF+硬约束） | P37 |
| `fig_illegal_bio.png` | 非法 BIO **9 柱**：金标地板 → 4 个 Linear → 4L+CRF（84.6%）→ BiLSTM-CRF（47.9%）→ 硬约束（6.1%） | P25 |
| `fig_per_role_f1.png` | 各角色论元 F1（BiLSTM-CRF，dev ≥20 次的角色里最难的 18 个） | P27 |
| `fig_training_curves.png` | 5 条 BERT 训练曲线（loss + 内层验证 span F1） | P24 |
| `fig_type_confusion.png` | 事件类型混淆矩阵（FinBERT-4L，行归一化=召回率） | P27 |
| `fig_latency_cost.png` | 精度 / 延迟 / 参数量三角权衡表 | P37/P28 |
| `fig_backbone_compare.png` | **额外**：只换骨干的三级 F1 + 成本-效果散点（报告专用） | — |

### 8.4 复现口径

```powershell
python eval\compare_methods.py        # 总表 → outputs\metrics\disc\summary.md
python eval\make_figures_myexp.py     # 6 张图
python eval\fig_backbone_compare.py   # 骨干对比图
```

所有数字来自 `outputs/metrics/disc/*.json` 与 `outputs/logs/*_train.json`，汇总表由脚本生成、不手抄。
