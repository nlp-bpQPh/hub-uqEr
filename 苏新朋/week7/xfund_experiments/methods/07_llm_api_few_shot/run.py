from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
subprocess.run([
    sys.executable, str(ROOT / "common/run_llm_api.py"),
    "--mode", "few",
    "--examples", "2",
    "--out", str(Path(__file__).resolve().parent / "outputs"),
], check=True)
