"""排队执行 RoBERTa / MacBERT 的训练（等 BERT-base 跑完后自动开始）。

用 Python 而不是 PowerShell 编排，避免 Windows PowerShell 5.1 读取 UTF-8 脚本时
中文路径乱码的问题。
"""

import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
OUT = CODE.parent / "outputs"
PY = sys.executable

WAIT = OUT / "enc_bert-base_tnews.json"
print("[queue] waiting for bert-base/tnews ...", flush=True)
while not WAIT.exists():
    time.sleep(20)

print("[queue] bert-base done; start roberta / macbert", flush=True)
for model in ("roberta-wwm-ext", "macbert-base"):
    for ds in ("chnsenticorp", "thucnews", "tnews"):
        print(f"=== {model} / {ds} ===", flush=True)
        subprocess.run([PY, str(CODE / "train_encoder.py"), "--model", model, "--dataset", ds],
                       cwd=str(CODE), check=False)

print("[queue] encoders done; start probes", flush=True)
for model in ("bert-base", "roberta-wwm-ext", "macbert-base"):
    for ds in ("chnsenticorp", "thucnews", "tnews"):
        print(f"=== probe {model} / {ds} ===", flush=True)
        subprocess.run([PY, str(CODE / "run_probe.py"), "--model", model, "--dataset", ds],
                       cwd=str(CODE), check=False)

print("[queue] ALL DONE", flush=True)
