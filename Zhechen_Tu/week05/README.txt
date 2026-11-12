第五周作业：基于语言模型的监督微调（SFT）
Zhechen Tu · 2026-11-12

实验问题
第四周的小模型原本只会接着 Shakespeare 文本往下写。如果给它“指令、句子、
标准回答”的示范，能否让它针对同一句话，按要求回答情绪、地点或时间？
本作业加载第四周 best_model.pt 做全参数微调；没有另加分类头或调用外部模型。
模型仍逐字符生成答案。架构、字符词表与基础权重兼容：2 层、4 头、128 维，
上下文 128 字符，430,080 个参数。基础权重来源固定到课程仓库提交
6af9b2a20fc846e9a499acffa21dbf51fe68339f，并在使用前检查 SHA-256。

数据与划分
dataset.py 合成英文单句。情绪、地点、时间各有三类，各类有三个线索词组。
例如：“I felt delighted near the schoolyard after lunch.” 三条指令的答案分别为
happy、school、afternoon。每个样本由 instruction、input、response 和审计用 task
组成。模型输入只有指令和句子，task 字段及生成器的词表映射不参与推理。
每个语义标签组合有 27 个措辞组合。固定种子 20261112，在每组内随机留出
21/3/3 个措辞组合给训练/验证/测试。对于同一个句子，其三条指令在同一集合。
共 729 个不同句子，训练 567 句（1,701 条），验证 81 句（243 条），测试 81 句
（243 条）。测试集各任务各 81 条。线索词和三个问句相同，测试的是未见组合。

训练方法
输入格式：Task: 指令 + 换行 + Text: 句子 + 换行 + Answer: 空格 + 答案 + 换行。
训练采用 causal LM 的右移目标。提示部分和右填充的标签设为 -100；
从提示末尾预测第一个答案字符，最后一个答案字符预测换行（结束符）。
训练与验证 NLL 按有效答案字符计权，越小越好。padding 位于右侧，不会被
有效位置的因果注意力看到。全参数 AdamW，初始学习率 0.0008，余弦下降至
0.00005，weight decay 0.01，batch 48，梯度裁剪 1，36 轮，MPS float32。
每轮评估验证集，仅以验证 NLL 选权重（第 30 轮最佳），再打开独立测试集。

结果
两个模型使用相同提示与贪心解码。不提供正确答案或候选列表，最多生成 14 个字符；
必须生成正确答案并以换行结束才算完全匹配。另用正确答案计算 response-only NLL。
                         基础模型             SFT
测试完全匹配              0/243               243/243
情绪 / 地点 / 时间        0 / 0 / 0           81 / 81 / 81（各 81 条）
测试答案 NLL              3.330576            0.000051
验证答案 NLL              3.328889            0.000068（最佳）
首个测试句分别执行三条指令的真实输出详见 results/comparison.txt。
微调前的模型生成类似 “the with the w” 的文本；微调后按要求生成标准标签。
原始测试预测逐条保存，可直接检查回答与停止条件。

模型是否能理解不同说法？
固定最佳权重，不再训练，在相同的 81 个留出句子上分别改变一种条件：
（1）只把时间放到句首、情绪放到句尾并改写句式：100/243（41.15%）。
（2）只把三种问句换成 How did I feel? / Where was I? / When was it?：
29/243（11.93%）。这两项是同批语义内容的敏感性诊断，不是新的独立测试集。
模型掌握了任务格式及见过的词语，但遇到改写时明显退化。因此固定模板的
100% 不能说明模型具备通用对话能力。也未评估微调后原有戏剧续写能力是否下降。

运行方式（本目录，Python 3.12）
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
直接使用随附权重：
  python predict.py --task mood --text "I felt delighted near the schoolyard after lunch."
  python predict.py --compare --task place --text "I felt delighted near the schoolyard after lunch."
复核提交结果（含基础模型及微调模型 CPU 重载、数据分组、标签和 padding 检查）：
  python verify.py
  python probe.py
复现训练（建议另存以保留随附权重），设备可选 cpu、mps、cuda：
  python dataset.py
  python sft.py --device cpu --output-dir rerun_results
原训练使用 --device mps；不同硬件和软件版本可能产生小幅数值差异。
默认训练输出 results；report.py 从 results 中的真实记录重新生成报告及曲线。
基础模型优先读取同级 week04/results/best_model.pt，独立使用作业包时如无基础
权重，训练/比较/验证脚本从固定版本 GitHub 地址下载并校验；仅做 SFT 推理无需联网。

文件索引
model.py / sft.py：模型、权重加载、SFT、验证选模、测试。
dataset.py / data/：确定性生成器及三个已生成的 JSONL 集合。
predict.py / verify.py / probe.py：实际推理、验证、模板敏感性实验。
results/：SFT 权重、训练日志、每轮记录、学习曲线、逐条预测、核验结果。
report.html：由真实记录生成的本地可读结果报告。
本作业由 Codex 协助实现并运行，结果均来自保存的模型和实测输出。
