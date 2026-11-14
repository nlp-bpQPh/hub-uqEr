"""数据模型：训练样本、训练结果、问答请求与响应。

全部用 Pydantic 声明——字段即校验规则，类型注解即文档。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SFTExample(BaseModel):
    """一条 SFT 训练样本。"""
    instruction: str = Field(..., min_length=1, description="用户指令")
    output: str = Field(..., min_length=1, description="期望回答")


class EpochRecord(BaseModel):
    """单个 epoch 的训练记录。"""
    epoch: int
    train_loss: float
    val_loss: float


class TrainResult(BaseModel):
    """一次训练的完整结果，落盘为 train_result.json。"""
    mode: str = "full"
    train_n: int
    val_n: int
    dropped: int
    epochs: float
    lr: float
    batch_size: int
    grad_accum: int
    grad_ckpt: bool
    max_len: int
    trainable: int
    total: int
    history: list[EpochRecord]
    best_epoch: int
    best_val_loss: float
    train_s: float
    device: str


class ChatRequest(BaseModel):
    """问答请求。"""
    question: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="0 表示贪心解码")


class ChatResponse(BaseModel):
    """问答响应。"""
    answer: str
    latency_ms: float
    model: str
