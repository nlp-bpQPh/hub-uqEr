# -*- coding: utf-8 -*-
"""
把实验产物整理成提交目录（作业仓库里的一份自包含副本）。

用法：
  python w7_ext/package_submission.py --dest "E:/大模型学习/学习资料/作业/hub-uqEr/谢献闯/week07_信息抽取对比"
只复制真实存在的文件；缺的会在输出里明确列出（不生成空占位）。
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W7 = ROOT / "outputs" / "w7_compare"


def cp(src: Path, dst: Path, log):
    if not src.exists():
        log.append(f"[缺] {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    dest = Path(args.dest)
    log: list[str] = []
    n = 0

    # 1) 代码
    for f in sorted((ROOT / "w7_ext").glob("*.py")):
        n += cp(f, dest / "code" / "w7_ext" / f.name, log)
    for f in sorted((ROOT / "w7_ext").glob("*.ps1")):
        n += cp(f, dest / "code" / "w7_ext" / f.name, log)

    # 2) 汇总结果
    for name in ("compare_all.csv", "compare_all.md", "compare_all.json",
                 "dataset_stats.json", "error_analysis.md", "error_analysis.json",
                 "report.md"):
        n += cp(W7 / name, dest / "results" / name, log)
    if (W7 / "report.md").exists():
        shutil.copy2(W7 / "report.md", dest / "实验报告.md")
        n += 1

    # 3) 每个方法的指标 JSON
    for d in (ROOT / "outputs" / "metrics" / "disc", ROOT / "outputs" / "metrics" / "llm"):
        for f in sorted(d.glob("w7_*.json")):
            n += cp(f, dest / "results" / "metrics" / f.name, log)

    # 4) 图
    for f in sorted((W7 / "figs").glob("*.png")):
        n += cp(f, dest / "figures" / f.name, log)

    # 5) 数据集转换统计
    n += cp(ROOT / "data" / "duee1" / "convert_stats.json",
            dest / "results" / "convert_stats_duee1.json", log)

    # 6) API 服务端逐请求日志（"换 API"一节的证据）
    n += cp(ROOT / "outputs" / "runlogs" / "api_server_stats.jsonl",
            dest / "results" / "api_server_stats.jsonl", log)

    print(f"copied {n} files → {dest}")
    for line in log:
        print(line)


if __name__ == "__main__":
    main()
