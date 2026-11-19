"""打印所有编码器结果，便于检查。"""

import glob
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
for p in sorted(glob.glob(str(OUT / "enc_*.json"))):
    d = json.load(open(p, encoding="utf-8"))
    r, rc = d["report"], d.get("report_clean", d["report"])
    print(f"{d['model_key']:18s} {d['dataset']:13s} "
          f"full={r['acc']:.4f}/{r['macro_f1']:.4f} clean={rc['acc']:.4f}/{rc['macro_f1']:.4f} "
          f"ece={r.get('ece')} best_ep={d['best_epoch']} t={d['train_seconds']}s")
