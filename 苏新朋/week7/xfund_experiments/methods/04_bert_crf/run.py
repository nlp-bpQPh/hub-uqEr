from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
subprocess.run([
    sys.executable, str(ROOT / "common/train_sequence.py"),
    "--method", "bert_crf",
    "--out", str(Path(__file__).resolve().parent / "outputs"),
    "--epochs", "2",
    "--batch-size", "8",
    "--max-chars", "160",
    "--max-len", "176",
    "--bert-layers", "4",
], check=True)
