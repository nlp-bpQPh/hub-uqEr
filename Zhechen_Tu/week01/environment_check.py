"""验证第一周课程环境，失败时返回非零退出状态。"""
import importlib
import platform
import struct
import sys
from importlib.metadata import version
from pathlib import Path

print("Week 01 | Development environment verification")
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()} ({struct.calcsize('P') * 8}-bit)")
assert sys.version_info[:2] == (3, 12)
assert struct.calcsize("P") * 8 == 64

packages = {
    "numpy": ("numpy", "1.26.4"),
    "pandas": ("pandas", "2.2.2"),
    "matplotlib": ("matplotlib", "3.9.2"),
    "torch": ("torch", "2.6.0"),
    "gensim": ("gensim", None),
    "scikit-learn": ("sklearn", "1.5.1"),
    "peft": ("peft", "0.15.0"),
    "transformers": ("transformers", "4.55.0"),
}
for package, (module_name, expected) in packages.items():
    importlib.import_module(module_name)
    actual = version(package)
    assert expected is None or actual == expected, (package, actual, expected)
    print(f"[PASS] {package:<14} {actual}")

import numpy as np
import pandas as pd
import torch
from gensim.corpora import Dictionary
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model

x = np.arange(5, dtype=float).reshape(-1, 1)
y = 2 * x.ravel() + 1
assert pd.DataFrame({"x": x.ravel(), "y": y}).shape == (5, 2)
assert np.isclose(LinearRegression().fit(x, y).predict([[5]])[0], 11)
assert torch.equal(torch.tensor([1, 2]) * 2, torch.tensor([2, 4]))
assert len(Dictionary([["hello", "ai"], ["hello", "coding"]])) == 3

# 使用随机初始化的小模型验证 Transformers 和 PEFT，不下载模型。
config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=32)
model = BertModel(config)
adapted = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                                         target_modules=["query", "value"]))
assert any(p.requires_grad for n, p in adapted.named_parameters() if "lora_" in n)
assert adapted(input_ids=torch.tensor([[1, 2, 3]])).last_hidden_state.shape == (1, 3, 16)

plt.plot(x.ravel(), y, "o-", label="y = 2x + 1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("environment_plot.png"), dpi=120)
plt.close()
print("[PASS] NumPy / Pandas / Matplotlib / Torch / Gensim / sklearn smoke checks")
print("[PASS] Tiny BERT + LoRA forward pass (no model download)")
print("ALL ENVIRONMENT CHECKS PASSED")
