"""从 ModelScope 镜像下载两个替代预训练模型。

hf-mirror 对同一 IP 频繁返回 429/502，改用 ModelScope：
  - dienstag/chinese-roberta-wwm-ext  （与 hfl/chinese-roberta-wwm-ext 的 config 逐字节一致）
  - hfl/chinese-macbert-base
下载按 Range 续传，遇到 429/5xx 退避重试，按总长校验完整性。
"""

import sys
import time
from pathlib import Path

import requests

MODELS = Path(r"E:\大模型学习\作业\第六周作业-分类方法对比\models")
FILES = ["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json", "pytorch_model.bin"]

REPOS = [
    ("chinese-roberta-wwm-ext", "dienstag/chinese-roberta-wwm-ext"),
    ("chinese-macbert-base", "hfl/chinese-macbert-base"),
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})


def download(url, dst, retries=100):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        pos = dst.stat().st_size if dst.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            r = SESSION.get(url, headers=headers, stream=True, timeout=(20, 120))
        except Exception as e:
            print(f"  [retry {attempt}] {type(e).__name__}", flush=True)
            time.sleep(4)
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  [{r.status_code}] wait ({attempt})", flush=True)
            time.sleep(5)
            continue
        if r.status_code == 416:
            print(f"  [416 done] {dst.name}", flush=True)
            return True
        if r.status_code not in (200, 206):
            print(f"  [http {r.status_code}] retry", flush=True)
            time.sleep(4)
            continue

        if r.status_code == 200:
            pos = 0
            mode = "wb"
            total = int(r.headers["Content-Length"]) if r.headers.get("Content-Length", "").isdigit() else None
        else:
            mode = "ab"
            cr = r.headers.get("Content-Range", "")
            total = int(cr.split("/")[-1]) if "/" in cr else None

        got = 0
        try:
            with open(dst, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        got += len(chunk)
        except Exception as e:
            print(f"  [stream break] {type(e).__name__} @ {pos + got}", flush=True)
            time.sleep(3)
            continue
        finally:
            r.close()

        size = dst.stat().st_size
        if total is None or size == total:
            print(f"  [ok] {dst.name} {size/1e6:.1f} MB", flush=True)
            return True
        print(f"  [partial] {dst.name} {size}/{total}", flush=True)
        time.sleep(2)
    print(f"  [FAIL] {dst}", flush=True)
    return False


def main():
    ok = True
    for name, repo in REPOS:
        print(f"[repo] {repo}", flush=True)
        for f in FILES:
            url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{f}"
            dst = MODELS / name / f
            print(f" - {f}", flush=True)
            ok = download(url, dst) and ok
    print("ALL DONE" if ok else "SOME FAILED", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
