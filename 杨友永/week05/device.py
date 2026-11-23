"""设备与精度选择。

不同 torch 版本对 MPS 的 bf16 支持不一致（2.2 不支持、2.4+ 支持），
所以精度不能写死，要跟着设备走。
"""
from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """优先 CUDA，其次 MPS，最后 CPU。"""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """CUDA 用 bf16，MPS 用 fp16，CPU 用 fp32。"""
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return device.type
