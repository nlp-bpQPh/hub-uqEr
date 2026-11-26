# -*- coding: utf-8 -*-
"""
本地 OpenAI 兼容服务（自建 API 端点，用于「换 API」对照实验）

为什么需要它：
  本周作业要比较「同一个模型，走 API 调用 vs 本地直接推理」的差异，但本机没有云端
  API Key。项目的 LLM 链路（llm_zero_shot/llm_extract.py --backend api）本来就是
  「OpenAI 兼容协议 + 环境变量配置」，所以这里起一个真实监听 HTTP 的兼容端点：
    - 客户端代码路径完全不变（openai SDK → /v1/chat/completions）；
    - 能测出「协议封装 + 单条串行」相对「本地批量直推」的额外延迟/吞吐代价；
    - 换成 DeepSeek 等云端服务时只需改 LLM_BASE_URL / LLM_API_KEY，代码零改动。

接口（够 openai>=1.x 的 chat.completions.create 用）：
  POST /v1/chat/completions
  GET  /v1/models
  GET  /health

用法：
  python w7_ext/serve_openai.py --model_path <Qwen3-0.6B目录> --port 8011 --dtype bf16
  然后：
  set LLM_BASE_URL=http://127.0.0.1:8011/v1 & set LLM_API_KEY=dummy
  python llm_zero_shot/llm_extract.py --backend api --mode zero --model qwen3-0.6b --n 200 --tag api_qwen06_zero

注意：这是**单卡串行**服务（一个 GPU 锁），故意不做连续批处理 —— 这样测出来的
      延迟才是「真实线上单卡端点」的量级。
"""
import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOKENIZER = None
LOCK = threading.Lock()
CFG = {}
STATS_PATH = None


def _resolve_dtype(name: str, device: torch.device):
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model(model_path: str, dtype: str = "auto"):
    global MODEL, TOKENIZER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=_resolve_dtype(dtype, device), trust_remote_code=True)
    MODEL = MODEL.to(device).eval()
    CFG["device"] = str(device)
    CFG["model_path"] = model_path
    print(f"[serve] loaded {model_path} on {device}", flush=True)


def build_prompt(messages: list[dict]) -> str:
    """chat messages → 一个 prompt 字符串（关掉 thinking，抽取任务只要正文）。"""
    try:
        return TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # 老模板不接受 enable_thinking
        return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(messages: list[dict], max_tokens: int) -> dict:
    prompt = build_prompt(messages)
    enc = TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False).to(CFG["device"])
    t0 = time.time()
    with LOCK, torch.no_grad():
        out = MODEL.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    gen_s = time.time() - t0
    new_ids = out[0][enc["input_ids"].shape[1]:]
    text = TOKENIZER.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "completion_tokens": int(new_ids.shape[0]),
        "server_gen_s": round(gen_s, 3),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静音，避免刷屏
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._send(200, {"status": "ok", "model": CFG.get("model_path"),
                             "device": CFG.get("device")})
        elif self.path in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [
                {"id": CFG.get("model_name", "local-model"), "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": {"message": f"bad json: {exc}"}})
            return

        messages = req.get("messages") or []
        max_tokens = int(req.get("max_tokens") or 512)
        t0 = time.time()
        try:
            res = generate(messages, max_tokens)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model") or CFG.get("model_name", "local-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": res["text"]},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
            },
        }
        # 服务端逐请求日志（用于分离「网络/排队」与「生成」耗时）
        if STATS_PATH:
            with open(STATS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": round(time.time(), 3),
                    "http_total_s": round(time.time() - t0, 3),
                    "server_gen_s": res["server_gen_s"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "finish_reason": "stop",
                }, ensure_ascii=False) + "\n")
        self._send(200, resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default="local-qwen")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--stats", default="")
    args = ap.parse_args()

    global STATS_PATH
    if args.stats:
        STATS_PATH = str(Path(args.stats))
        Path(STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(STATS_PATH).write_text("", encoding="utf-8")

    load_model(args.model_path, args.dtype)
    CFG["model_name"] = args.model_name
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}/v1 "
          f"(model_name={args.model_name})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
