# API few-shot

与 zero-shot 使用相同提示词、验证集及评测代码，只从训练集固定抽取少量 OCR 文档作为示例。当前机器没有 API 凭证，因此没有发起外部调用。

配置 `LLM_MODEL`、`LLM_API_KEY`，如需 OpenAI 兼容服务再配置 `LLM_BASE_URL`，然后运行 `python run.py`。
