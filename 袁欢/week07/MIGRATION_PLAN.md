# code7 数据换装定版方案：DuEE-Fin → CCED（CSED.json）

> 任务口径（2026-11-24 用户确认）：**接受事件检测降级，砍掉论元链路**；bert_pipeline 砍检测段；
> 类型按 9 类归并表执行；95 行多触发词取首个位置。
> 执行环境：本机（py312 = E:\soft\anaconda\envs\py312）做构建与秒级验证；BERT/LLM 训练留远程 GPU。

---

## 0. 数据实测与四个已定口径

CSED.json（data/CCED-main/）实测：2,014 行 JSONL（33 行脏行解析失败）、字段
`id/sentence/tokens/trigger/trigger_positions/eventype/eventype_id`；只有类型+触发词、
**无论元标注**；trigger_positions 为字符级左闭右开（与 code7 契约天然兼容，无需 find 回填）；
95 行多触发词位置；无负样本、无篇章数据、无官方 train/dev 切分。

| 口径 | 决策 |
|---|---|
| 任务定位 | 事件检测（类型 + 触发词 span），**论元链路整段删除** |
| 两段式 | 砍检测段 → 改造为「类型分类 → 触发词专家」两段，不产 \*_detect.jsonl |
| 类型归并 | 9 类：数据泄露 360 / 网络钓鱼 271（+钓鱼攻击3+网络钓鱼攻击1+钓鱼邮件1）/ 勒索赎金 170（赎金169+勒索1）/ DDoS攻击 119（DDos攻击88+DDoS攻击31）/ 恶意软件 422 / 供应链攻击 33（+供应链13）/ 漏洞影响 171（+安全漏洞3）/ 漏洞发现 269（+漏洞2）/ 漏洞补丁 192；空串 2 丢弃 |
| 多触发词 | 95 行取首个位置（保量约 2,007 条） |

---

## 1. 数据构建：新建 data/scripts/build_cced.py（完整代码见执行交付）

- 读 CSED.json JSONL：脏行（JSON 解析失败 / eventype 空串）过滤并计数；
- 类型归并映射表（上表 9 类）；trigger_positions 取首个区间；
- span 自洽校验：`sentence[s:e] == trigger`，不自洽行丢弃计数；
- 自切 train/dev：stratify（按归并后类型）8:2，seed=42；
- 产出（写入 data/processed/，契约键名不变使下游零改动）：
  - `event_schema.json`：`{"数据泄露": [], ...}`（角色列表全空，无 __enums__）
  - `train_extract.jsonl` / `dev_extract.jsonl`：`{"id","sent_id","text","event":{type,trigger{text,start,end},args:[]}}`
  - `dev_docs.jsonl`：每句包成单句「文档」`{"id","title":"","text","events":[...]}`（保 service 可跑）
  - **不产** \*_detect.jsonl（两段式已无检测段）
- verify_processed.py 同步改造（角色数=0 合法、detect 缺失合法）；stats 落 data/stats/。

## 2. 公共层

| 文件 | 改动 |
|---|---|
| common/evaluate_events.py | **口径重定义**：删除 argument 指标；主指标 = trigger F1、type_macro F1、event_exact（类型+触发词全对，即事件检测标准口径）；per_type 保留，per_role 删除 |
| common/event_data.py | schema 驱动的标签体系保留（roles 为空时 ARG 标签自动为 0，joint 标签 = TRG:类型×9+O ≈ 19）；docstring 与统计打印更新；论元归一化/BIO 的 ARG 分支保留为通用逻辑 |
| common/eval_utils.py | 评测封装随指标列调整 |
| common/models.py / crf.py / model_paths.py | 零改动（标签数/路径动态；bert-base-chinese 仍适用） |

## 3. 各方法目录

| 方法 | 改动 |
|---|---|
| rule/run_rule.py | 触发词词典法保留（词典从 train 统计，机制不变）；:48-52 金融同义词表换安全域；**论元正则/公司词典段删除**（MONEY_RE/SHARE_RE/PERCENT_RE/MONEY_KEYS 等） |
| bilstm_crf/train.py | 零代码改动（标签动态），docstring 数字更新 |
| bert/train.py | 零代码改动；docstring（149 标签→约 19）更新 |
| bert_pipeline/train.py | **砍检测段**：(a) 类型 9 分类（[CLS] 头）→(b) 类型条件触发词专家（B-TRG/I-TRG，每类 2~3 标签）；oracle_type 评测保留；不再读 detect 文件 |
| llm_zero_shot/event_prompt.py | SYSTEM_PROMPT_TEMPLATE 重写（安全事件检测专家、无论元指令、「共 9 类」）；schema_type_lines 动态复用；**6 个 few-shot 金融示例全部重写**（从新 train_extract 挑混淆对，如漏洞影响 vs 漏洞发现）；parse_model_output 删 arguments 分支 |
| llm_lora/train_sft.py / evaluate_sft.py | 目标 JSON 无 arguments（随 event_prompt）；prompt 变短 → max_length 调小 |
| service/extract_service.py | 删正则兜底（FALLBACK_PATTERNS/MONEY 等角色语义 keys）/类型→核心角色表/「缺核心角色」needs_review 判据；保留切句→抽取→schema/span 校验→去重；篇章合并退化为逐句事件聚合（安全告警视角）；自测样例重写 |

## 4. 基础设施与产物

- tools/make_sample_data.py：去掉 detect 抽样；sample_stats 相应调整；
- data/_mock/：重造安全域 20 条（train/dev_extract 各 20、event_schema 9 类、dev_docs 2 篇；detect 文件删除）；
- eval/compare_methods.py：表格列去论元指标；
- 旧数据归档：data/processed → data/processed_duee_fin/；sample/_mock 备份；outputs/ 全部归档 outputs/duee_fin/；
- 五份 md（README/ARCHITECTURE/USAGE/RESUME/NOTES×2）+ data/DATA_REPORT + make_figures 标题：任务叙事与数字重写（后置，训练完成后做）。

## 5. 执行顺序与验收

```
0) 备份归档旧数据与 outputs
1) data/scripts/build_cced.py 生成新 processed（校验：9 类计数、span 自洽、train/dev 无交集）
2) common/evaluate_events.py 口径改造 + 单元验证（gold-vs-gold = 1.0）
3) rule / bert_pipeline / event_prompt / service 改造
4) 本机验证：rule 全量 dev、bert --mock 冒烟、统一评测跑通
5) 远程 GPU（后置）：bert/train.py、bilstm_crf、llm_lora
6) 文档重写（后置）
```

### 5.1 验证命令（全部方式）

约定：`--mock` 冒烟一律 `--device cpu`（别抢卡）；正式训练在远程 GPU 跑。
所有训练脚本的产物约定一致：`outputs/checkpoints/<run_name>.pt`、
`outputs/disc/pred_<run_name>.jsonl`、`outputs/metrics/disc/<run_name>.json`。

**0) 公共层（一次性，换装后先跑这三条）**

```bash
python data/scripts/build_cced.py                  # 重造 processed，校验 9 类计数 / span 自洽 / 8:2 切分
                                                   # ✅ 2026-11-24：kept=1764 train=1411/dev=353，幂等复现
python common/evaluate_events.py --gold data/processed/dev_extract.jsonl --pred data/processed/dev_extract.jsonl
                                                   # gold-vs-gold 必须 = 1.0（对齐正确性验证）
                                                   # ✅ 2026-11-24：trigger/event_exact/type_macro 全 = 1.0000
python common/model_paths.py                       # 自检：打印 BERT 权重解析路径（找不到即报全部已尝试路径）
                                                   # ✅ 2026-11-24：→ pretrain_models\bert-base-chinese
```

**1) L0 规则法（本机可跑全量 dev）**

```bash
python rule/run_rule.py --mock                     # 20 条手造数据冒烟
                                                   # ✅ 2026-11-24：流程通，exact 45%
python rule/run_rule.py                            # 全量 dev：产出 outputs/metrics/disc/rule.json + 失败四分类分析
                                                   # ✅ 2026-11-24：trigger F1=0.5100 / type_macro=0.7826
```

**2) L1 BiLSTM-CRF（本机 mock；正式远程 GPU）**

```bash
python bilstm_crf/train.py --mock --device cpu     # 冒烟（run_name 自动 = bilstm_crf_mock）
                                                   # ✅ 2026-11-24：全流程通（mock 指标无意义）
python bilstm_crf/train.py --decoder crf           # ✅ 正式：bilstm_crf
python bilstm_crf/train.py --decoder linear        # ✅ 对照：bilstm_linear
```

**3) L2 BERT 联合（正式远程 GPU；12 层 + bert-base-chinese）**

```bash
python bert/train.py --mock --device cpu           # 冒烟（4 层小模型，run_name 自动带 _mock）
                                                   # ✅ 2026-11-24：全流程通，ckpt=bert_crf_256_mock.pt
python bert/train.py                               # ✅ 主线：bert_crf_256
python bert/train.py --decoder linear              # ✅ 对照：bert_linear_256
python bert/train.py --decoder linear --max_len 128 --run_name bert_linear_128   # ✅ 消融：max_len
python bert/train.py --bert_layers 4 --decoder linear                            # ✅ 消融：前 4 层（bert4L_linear_256）
python bert/train.py --decoder crf --hard_constraint                             # ✅ 消融：CRF 硬约束（bert_crf_256_hard）
python bert/evaluate.py --run_name bert_crf_256    # ✅ 训练后重评 / 换 split 复评
                                                   # ✅ 2026-11-24 mock ckpt 实测通过（ckpt 须显式 --ckpt 传入）
python bert/evaluate_hard_constraint.py            # ✅ 硬约束消融对照表（须 --crf 硬约束 ckpt 存在）
                                                   # ✅ 2026-11-24 mock ckpt 实测通过（argument 残留已修）
```

**4) L2 两段式流水线（类型 9 分类 → 每类触发词专家）**

```bash
python bert_pipeline/train.py --mock --device cpu  # 冒烟：mock ckpt（4 层）全链路
                                                   # ✅ 2026-11-24：oracle 上界 type_macro=0.9630，机制验证通过
python bert_pipeline/train.py                      # ✅ 正式远程 GPU：产 pipeline.json + pipeline_oracle_type.json（类型上界）
python bert_pipeline/train.py --only_experts       # 类型分类器已训好，只补训专家（断点续跑）
                                                   # ✅ 2026-11-24 mock 实测通过
python bert_pipeline/train.py --skip_train         # 只用已有 ckpt 评测
                                                   # ✅ 2026-11-24 mock 实测通过（skip 不生效+UnboundLocalError 已修）
```

**5) LLM zero-shot / few-shot（本机 mock；API 需 key，本地推理需 Qwen3-0.6B）**

```bash
python llm_zero_shot/llm_extract.py --backend mock --mode zero   # 离线冒烟（假响应，验证解析与统计链路）
                                                   # ✅ 2026-11-24：trigger F1=0.9138（mock 只证明链路通）
python llm_zero_shot/llm_extract.py --backend mock --mode few    # few-shot 版冒烟
                                                   # ✅ 2026-11-24：链路通
set LLM_API_KEY=sk-xxx & python llm_zero_shot/llm_extract.py --backend api --mode zero
                                                   # ✅ Windows 设环境变量后跑 API（默认 n=200；换模型只改 LLM_BASE_URL/LLM_MODEL）
python llm_zero_shot/llm_extract.py --backend local --mode zero --device cuda   # ✅ 本地 Qwen3-0.6B 推理（远程 GPU）
```

**6) LLM LoRA SFT（远程 GPU；基座 Qwen3-0.6B）**

```bash
python llm_lora/train_sft.py --max_steps 5         # ✅ 冒烟：只跑 5 个优化器步，验证数据与显存
                                                   # 注：本机无 Qwen3-0.6B 权重（2026-11-24 实测），冒烟也须远程
python llm_lora/train_sft.py                       # ✅ 正式：LoRA 全量 train（约 1411 条 → prompt 目标 JSON）
python llm_lora/evaluate_sft.py                    # ✅ SFT 后评测 dev n=200（与 API 实验同批，产 sft_lora_n200）
python llm_lora/train_sft.py --full_ft             # ✅ 可选对照：全参微调
```

**7) service 系统组合（依赖 ckpt；mock 冒烟不占卡）**

```bash
python service/demo_extract.py --mock              # 4 层 mock ckpt 冒烟：切句→抽取→校验→needs_review 全链路
                                                   # ✅ 2026-11-24：全链路通（--mock 现自动选 bert_crf_256_mock.pt）
python service/demo_extract.py --backend joint     # ✅ 正式：joint CRF ckpt（bert_crf_256 训完后）
python service/demo_extract.py --backend pipeline  # 正式：两段式 ckpt（pipeline 训完后）
                                                   # ✅ 2026-11-24 mock ckpt 实测通过
```

**8) 汇总与抽样（每次正式实验跑完后）**

```bash
python eval/compare_methods.py --with_n200 --out outputs/metrics/disc/summary.md
                                                   # 汇总主表+消融表（缺失方法标「未跑」，绝不补估算值）
                                                   # ✅ 2026-11-24：主表 rule 行正常（mock 结果按设计不入表）
python tools/make_sample_data.py                   # 重造 data/sample（PPT 用抽样）
                                                   # ✅ 2026-11-24：seed=42 固定，合计 1150 条
python data/scripts/verify_processed.py            # 数据契约校验（注：待按新契约改造——角色数=0 合法、detect 缺失合法）
```

验收标准：构建输出 9 类计数与归并表一致；trigger span 校验 0 错；gold-vs-gold F1 = 1.0；
rule 能跑出非平凡指标；bert --mock 全流程不崩；compare_methods 表格口径为新指标。
