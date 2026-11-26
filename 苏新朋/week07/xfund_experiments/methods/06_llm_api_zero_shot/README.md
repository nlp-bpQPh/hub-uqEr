# API zero-shot

XFUND OCR 块的结构化分类 prompt。验证时每份文档一次调用，输出标签和键值链接；当前机器没有 API 凭证，因此没有发起外部调用。

配置 `LLM_MODEL`、`LLM_API_KEY`，如需 OpenAI 兼容服务再配置 `LLM_BASE_URL`，然后运行 `python run.py`。
