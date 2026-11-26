"""只跑报告需要的冻结特征对照：TNEWS 三个模型 + 两个新数据集的 BERT。"""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
PY = sys.executable

COMBOS = [
    ("bert-base", "thucnews"),
    ("bert-base", "tnews"),
    ("roberta-wwm-ext", "tnews"),
    ("macbert-base", "tnews"),
]

for model, ds in COMBOS:
    print(f"=== probe {model} / {ds} ===", flush=True)
    subprocess.run([PY, str(CODE / "run_probe.py"), "--model", model, "--dataset", ds],
                   cwd=str(CODE), check=False)

print("[probe queue] ALL DONE", flush=True)
