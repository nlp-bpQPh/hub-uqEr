"""从 hf-mirror 下载替代预训练模型（只取 PyTorch 权重与分词器，跳过 tf/flax 大文件）。

用法：
  set HF_ENDPOINT=https://hf-mirror.com
  python code/download_models.py
"""

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

TARGETS = [
    "hfl/chinese-roberta-wwm-ext",
    "hfl/chinese-macbert-base",
]

ALLOW = [
    "*.json", "*.txt", "*.bin", "*.safetensors", "*.model",
]

for repo in TARGETS:
    name = repo.split("/")[-1]
    dst = MODELS / name
    print(f"[download] {repo} -> {dst}", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dst),
        allow_patterns=ALLOW,
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ckpt"],
        max_workers=4,
    )
    print(f"[ok] {name}", flush=True)

print("all done")
