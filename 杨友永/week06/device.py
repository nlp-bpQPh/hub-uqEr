"""设备封装：目标设备 + 混合精度策略。"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Device:
    target: torch.device
    amp: bool = False
    amp_dtype: torch.dtype = torch.float32

    @classmethod
    def detect(cls, prefer: str | None = None) -> "Device":
        target = torch.device(prefer) if prefer else cls._auto()
        return cls(target, *cls._amp_policy(target))

    @staticmethod
    def _auto() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _amp_policy(target: torch.device) -> tuple[bool, torch.dtype]:
        if target.type == "cuda":
            return True, torch.bfloat16
        return False, torch.float32

    @property
    def type(self) -> str:
        return self.target.type

    def to(self, obj):
        return obj.to(self.target)

    def autocast(self):
        return torch.autocast(device_type=self.target.type,
                              dtype=self.amp_dtype, enabled=self.amp)

    def describe(self) -> str:
        if self.type == "cuda":
            name = torch.cuda.get_device_name(self.target)
            prec = f"{self.amp_dtype} autocast" if self.amp else "fp32"
            return f"cuda ({name}) · {prec}"
        return f"{self.type} · fp32"

    def __str__(self) -> str:
        return self.describe()
