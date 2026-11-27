from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "common"))
from xfund_data import load_xfund, write_normalized_splits

if __name__ == "__main__":
    stats = write_normalized_splits(ROOT / "data/source/zh.train.json",
                                    ROOT / "data/source/zh.val.json",
                                    ROOT / "data/processed")
    for split in ("train", "val"):
        docs = load_xfund(ROOT / f"data/source/zh.{split}.json")
        stats[f"{split}_blocks"] = sum(len(d.get("document", [])) for d in docs)
        stats[f"{split}_links"] = sum(len(b.get("linking", [])) for d in docs for b in d.get("document", []))
    (ROOT / "data/processed/stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
