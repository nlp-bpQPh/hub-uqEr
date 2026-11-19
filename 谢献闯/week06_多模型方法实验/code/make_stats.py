"""导出三个数据集的数据概况（供报告引用）。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg import DATASET_SPECS, OUT_DIR
from data_lib import dataset_stats, load_data

OUT_DIR.mkdir(parents=True, exist_ok=True)
stats = {name: dataset_stats(load_data(name)) for name in DATASET_SPECS}
(OUT_DIR / "dataset_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
