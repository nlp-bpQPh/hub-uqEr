# -*- coding: utf-8 -*-
"""Print per-role / per-type / illegal summary from a metrics JSON."""
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else CODE / "outputs/metrics/disc/bilstm_crf.json"
r = json.load(open(path, encoding="utf-8"))
print("method:", r.get("method"), "n:", r.get("num_samples"))
print("overall: trig %.4f arg %.4f event %.4f macro %.4f" % (
    r["trigger"]["f1"], r["argument"]["f1"], r["event_exact"]["f1"], r.get("type_macro_f1", 0)))
print("\ntop roles by tp:")
for k, v in sorted(r["per_role"].items(), key=lambda kv: -kv[1]["tp"])[:12]:
    print("  %-10s P=%.3f R=%.3f F1=%.3f tp=%d" % (k, v["precision"], v["recall"], v["f1"], v["tp"]))
print("\nper type (top tp):")
for k, v in sorted(r["per_type"].items(), key=lambda kv: -kv[1]["tp"])[:10]:
    print("  %-22s F1=%.3f tp=%d" % (k, v["f1"], v["tp"]))
print("\nillegal:", (r.get("meta") or {}).get("illegal_dev"))
print("meta keys:", sorted((r.get("meta") or {}).keys()))
