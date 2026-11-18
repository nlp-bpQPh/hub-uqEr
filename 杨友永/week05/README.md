# week05 —— 最小 SFT 工程（全参微调）

把 **Qwen2.5-0.5B（base）** 用 3000 条指令数据微调成问答模型。

不用 LoRA、不用 Trainer；训练是一个普通循环。代码统一切到**类型注解 + Pydantic 模型**的写法。

## 目录

```
week5_llm/
├── config.py       # 路径与超参（frozen dataclass，全量类型注解）
├── device.py       # 设备与精度选择（CUDA→bf16 / MPS→fp16 / CPU→fp32）
├── schemas.py      # Pydantic 模型：样本、训练结果、请求、响应
├── data.py         # jsonl → Pydantic → chat template → 带掩码的张量
├── train.py        # 全参 SFT + 按验证 loss 存最优
├── chat.py         # ChatEngine：ask(ChatRequest) -> ChatResponse
├── main.py         # CLI：train / chat / demo
├── Qwen2.5-0.5B/  sft_train/  outputs/full/
└── requirements.txt
```

## 代码风格

**① 一切都有类型**

```python
def build_loader(path: Path, tok: Any, max_len: int, batch_size: int,
                 shuffle: bool) -> tuple[SFTDataset, DataLoader]:
```

**② 数据结构用 Pydantic 声明**

```python
class SFTExample(BaseModel):
    instruction: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(64, ge=1, le=512)
```

字段即校验：读 jsonl 时坏样本直接报错，而不是训到一半才发现。

**③ 配置是不可变 dataclass**

```python
@dataclass(frozen=True)
class TrainConfig:
    max_len: int = 256
    batch_size: int = 4
    ...
```

**④ 核心接口只有两个**

```python
engine = ChatEngine.load()
response = engine.ask(ChatRequest(question="..."))
print(response.answer, response.latency_ms)
```

## 用法

```bash
pip install -r requirements.txt

python main.py train                 # 全量训练（约 5 分钟）
python main.py train --limit 200     # 快速验证链路
python main.py chat                  # 交互问答
python main.py demo                  # 固定问题试答
```

## 训练配置

| 项 | 值 |
|---|---|
| 可训练参数 | 494 M（100%） |
| batch / 梯度累积 | 4 / 4 |
| 学习率 | 2e-5 |
| 梯度检查点 | 开 |
| max_len | 256 |

## 训练结果（RTX 3070 Ti，307 秒）

| epoch | train loss | val loss | |
|---|---|---|---|
| 1 | 1.3260 | 1.0887 | |
| 2 | 0.8371 | **1.0706** | ← 保存 |
| 3 | 0.7135 | 1.0906 | 过拟合 |

## 效果

| 问题 | 底座模型 | 微调后 |
|---|---|---|
| 把「Tom 发现了一份工作」改成被动语态 | `Tom 发现了一份工作。` | `Tom被发现了一份工作。` |
| 整理待办清单 | `您好！请问您需要我提供什么帮助？` | `1. 买水果，2. 回邮件，3. 洗衣服` |

实测延迟：RTX 3070 Ti **348~412 ms**，Mac MPS 656~807 ms。

## 环境兼容（踩过的三个坑）

**① tokenizer.json 跨版本不兼容。**
在云主机（transformers 5.20 / tokenizers 0.23）保存的 tokenizer.json，用旧环境（tokenizers 0.19）加载会报
`data did not match any variant of untagged enum ModelWrapper`。

修复：**训练的 checkpoint 不再保存 tokenizer**，推理时固定从底座目录读——SFT 本来就不改 tokenizer。

**② torch 2.2 的 MPS 不支持 bf16。**
报 `TypeError: BFloat16 is not supported on MPS`。

修复：新增 `device.py`，精度跟着设备走：

| 设备 | dtype |
|---|---|
| CUDA | bfloat16 |
| MPS | float16 |
| CPU | float32 |

**③ torch 2.2 的 MPS 缺 `aten::isin`。**
`generate()` 会用到，报 `NotImplementedError`。

修复：在导入 torch 前设 `PYTORCH_ENABLE_MPS_FALLBACK=1`，让个别算子回退 CPU。

> 换用 torch ≥ 2.4 就不会遇到 ②③，代码里保留是为了兼容旧环境。

## 两个坑

**① chat template 的 `<|im_end|>` ≠ eos。**
只传 `eos_token_id` 模型不会停，会一直生成到上限。两个都要传。

**② 3 轮就过拟合。**
val loss 第 2 轮触底（1.0706），第 3 轮回升（1.0906）。训练脚本按验证 loss 保存最优。
