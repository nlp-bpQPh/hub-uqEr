"""对单条文本运行两阶段事件抽取。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline_predict import load_pipeline, predict_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="两阶段事件抽取")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = load_pipeline(
        args.stage1_dir,
        args.stage2_dir,
        args.device,
    )
    text = args.text or input("请输入新闻文本：")
    print(
        json.dumps(
            predict_text(bundle, text),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

