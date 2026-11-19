"""跑 9 组「冻结特征 + 逻辑回归」对照实验。"""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
PY = sys.executable

for model in ("bert-base", "roberta-wwm-ext", "macbert-base"):
    for ds in ("chnsenticorp", "thucnews", "tnews"):
        print(f"=== probe {model} / {ds} ===", flush=True)
        subprocess.run([PY, str(CODE / "run_probe.py"), "--model", model, "--dataset", ds],
                       cwd=str(CODE), check=False)

print("[probe queue] ALL DONE", flush=True)
