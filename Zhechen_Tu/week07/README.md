# 第七周作业 · 英文人名和地名抽取

在 CoNLL-2003 新闻与 WNUT-17 论坛文本上，比较训练集词典、spaCy、DistilBERT 和经 Ollama HTTP API 调用的 Gemma。所有方法输出实体的字符起点、终点和类别，三项同时匹配才计为正确。

- [实验报告](report.md)
- [对比图](results/comparison.png) · [指标表](results/scores.csv) · [按类别分解和误差位置](results/scores.json)
- [配对 bootstrap 区间](results/paired_bootstrap.json) · [API 样本编号](results/api_sample_ids.json) · [校验日志](results/verification.txt)

## 复核已提交的结果

本次使用 Python 3.13。在本目录运行以下命令。首次执行会下载固定版本的测试数据，随后可直接用仓库内的逐条预测重算指标，无需启动 Ollama 或重新推理。

```sh
python -m pip install -r requirements.txt
python verify.py
python evaluate.py
```

`results/predictions/` 每行记录一条样本的 ID 和零起点、右端开区间的 `[start, end, label]` 列表。输入文本由数据集原始 token 用一个 ASCII 空格连接。`results/sources.json` 记录三个源文件的固定 URL 与 SHA-256，`results/core_run.json` 和 `results/api_run.json` 保存模型版本、参数和计时。

## 重新运行抽取

本地三个方法以 CPU 执行，首次运行会下载固定版本的 DistilBERT 权重。第四种方法使用本机 `127.0.0.1:11434` 上的 Ollama 服务，可以用 `ollama pull gemma4:e2b` 下载模型。运行顺序如下。

```sh
python run_experiment.py --stage core --threads 4
python run_experiment.py --stage api
python evaluate.py
python verify.py --replay-local
```

API 阶段按每批四条保存本机缓存，重试时可复用同一模型摘要、提示、参数及数据对应的已完成批次。若需重新请求，可移走 `.cache/api-*` 目录后运行。源码数据、预训练权重和原始 HTTP 回包留在本机缓存，提交物保留来源、设置和预测位置。

用 `python run_experiment.py --stage wordpiece` 可复跑同一 DistilBERT 检查点使用 `simple` 子词聚合的对照实验。
