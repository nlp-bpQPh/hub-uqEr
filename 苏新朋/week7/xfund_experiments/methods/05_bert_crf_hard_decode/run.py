from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "methods/04_bert_crf/outputs/best.pt"
if not CHECKPOINT.exists():
    raise SystemExit("Run methods/04_bert_crf/run.py first to create its checkpoint.")
subprocess.run([
    sys.executable, str(ROOT / "common/evaluate_checkpoint.py"),
    "--checkpoint", str(CHECKPOINT),
    "--hard-bio",
    "--max-chars", "160",
    "--max-len", "176",
    "--out", str(Path(__file__).resolve().parent / "outputs"),
], check=True)
