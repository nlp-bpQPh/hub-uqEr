第四周作业：训练基于 Transformer 的语言模型
姓名：Zhechen Tu
日期：2026-11-04

一、任务与完成情况
从零训练字符级、自回归、decoder-only Transformer。输入一段英文前缀，
模型每次预测下一个字符，把新字符拼回上下文，持续生成文本。
本作业实际完成数据处理、训练、验证选择、独立测试、模型保存/加载和生成。
未调用预训练语言模型；全部 430,080 个参数从随机初始化开始学习。

二、数据
使用 char-rnn 仓库的 Tiny Shakespeare 文本，共 1,115,394 个字符。
原作者 William Shakespeare；数据整理/分发来源 Andrej Karpathy。
来源：https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
原文与来源说明见 data/input.txt、data/SOURCE.txt；SHA-256 见数据清单。

按原文连续位置划分，采用左闭右开区间：
训练 [0,1003854)：1,003,854 个字符（约 90%）。
验证 [1003854,1059624)：55,770 个字符（约 5%）。
测试 [1059624,1115394)：55,770 个字符（约 5%）。
先切分再采样，任何训练窗口都不会跨入验证或测试范围。
仅由训练段建立词表：65 个实际字符 + 1 个保留 <UNK>，总计 66 项。
本次验证、测试均无未知字符。未按所有窗口随机打乱后再切分。
这是同一文学语料的不同连续片段，不保证不同片段没有相似台词，
也不等价于不同作者、不同领域上的泛化评估。

三、结构与原理
字符 ID -> Token Embedding + 可学习位置 Embedding -> 2 个 Transformer Block
-> 最终 LayerNorm -> Linear(128,66) -> 下一个字符的 logits。
每个 Block：Pre-LayerNorm、多头自注意力、残差连接，再接
Pre-LayerNorm、前馈网络、残差连接。
隐藏维度 128；注意力头数 4（每头 32）；前馈维度 512；GELU；dropout 0.1。
上下文最长 128 个字符。没有 encoder，也不需要交叉注意力。
注意力调用 scaled_dot_product_attention(..., is_causal=True)：
第 t 个位置只能读取 0..t 位置，禁止读取未来字符。
输入 x=[c0,c1,...,c127]，目标 y=[c1,c2,...,c128]。
损失为所有 batch/时间位置上的交叉熵；预测过程不是输入原文的直接复制。

四、训练
Python 3.12、PyTorch 2.6.0；本次使用 Apple Silicon MPS，float32。
种子 20261104；batch_size=24；1,500 个优化步；共处理 4,608,000 个
目标字符（有放回随机窗口采样，包含重复曝光，不能视作独立样本数）。
AdamW：betas=(0.9,0.95)，weight_decay=0.01；梯度裁剪 max_norm=1.0。
前 100 步线性 warmup 到 0.001，随后余弦衰减至 0.0001。
每 100 步评估完整验证集，以最低验证 NLL 保存 best_model.pt。
只在权重选择结束后评估测试集，本次最佳步数为 1,500。
保存了权重、配置、词表、种子和数据校验值；未保存优化器状态，
因此支持重新训练和推理，不提供逐步无缝断点续训。
固定种子提高复现性，但 MPS/CPU、不同硬件及版本不保证逐位相同。

五、评估方法与实测结果
交叉熵/NLL 单位为 nats/字符。困惑度 PPL=exp(NLL)，越低越好。
它是字符级困惑度，不能直接与词级/BPE 模型的困惑度比较。
验证和测试均顺序遍历，以 128 字符为块，每块重置上下文；
最后不足 128 的块也计入。每个集合的第 1 个字符只作上下文，
之后 55,769 个目标各计分一次。评估时关闭 dropout。

训练前验证：NLL=4.176382，PPL=65.129797。
最佳模型验证：NLL=1.794100，PPL=6.014059。
独立测试：NLL=1.855954，PPL=6.397798，下一字符准确率=44.7973%。
对比基线（只用训练段计数，unigram/bigram 采用加一平滑）：
均匀分布：测试 NLL=4.189655，PPL=66.000000。
单字符频率 unigram：测试 NLL=3.361968，PPL=28.845898。
二元字符 bigram：测试 NLL=2.489599，PPL=12.056436。
Transformer 优于三个基线，说明利用了较长的前文信息。
训练曲线中的 train_nll 是最近 100 步训练态损失均值，验证损失
是当前权重在完整验证段的评估态结果，两者统计条件不同。

六、生成效果与局限
固定三个前缀：ROMEO:、KING RICHARD II:、First Citizen:，均附换行。
每次生成 400 个字符，temperature=0.8，top_k=20，seed=20261104。
完整原始输出见 results/generation_examples.txt / json，没有挑选或润色。
模型已经生成类似人物标签、换行和英文词形，但仍有拼写错误、
语法不通、重复表达和缺乏连贯含义的问题，例如输出包含“Murdere”等词。
这只是小数据、短上下文、小参数的教学模型，不能作为高质量写作或问答系统。
更长训练、更大数据和更长上下文可作为后续实验，本次未根据测试结果调参。
不支持中文提示；空提示、词表外字符会明确报错。生成不会输出 <UNK>。

七、复现操作（在本目录执行）
创建环境：
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
requirements_lock.txt 记录本次完整环境（含前几周已安装的额外包）。

直接加载已提交权重进行生成，不必重新训练：
  python generate.py --prompt "ROMEO:" --length 400
  python generate.py --prompt "First Citizen:" --temperature 0.8 --top-k 20
重新生成本次三组固定演示：
  python generate.py --examples
核验数据、因果掩码、模型重载和生成：
  python verify.py

重新训练：
  python train.py --device cpu
Apple Silicon 可用：
  python train.py --device mps
有 CUDA 的机器可用：
  python train.py --device cuda
保留已提交结果，另存新训练：
  python train.py --device cpu --output-dir rerun_results
默认重新训练会覆盖 results；重新训练后的生成样例需要再次运行 generate.py。
原数据已打包；如需恢复，可执行 python download_data.py（会核对固定 SHA-256）。

八、已实际运行的检查
数据源哈希、仅训练段建立词表、划分区间、输入与目标错位对应正确。
修改输入第 65 个字符及之后的内容，前 64 个位置的 logits 保持不变：
该检查验证因果掩码确实阻止未来信息泄漏。
CPU 重载权重后重新评估，NLL 与 MPS 原结果差异小于 0.0001，
测试目标计数正确，PPL 与 exp(NLL) 一致，优于 bigram 基线。
同种子生成结果相同，长度正确，空提示及词表外字符拒绝通过。
全部检查通过，原始输出见 results/verification_log.txt。

九、文件
model.py：自注意力、前馈网络、残差、位置编码、Transformer 模型。
train.py：训练、数据划分、验证、基线、测试、权重保存及训练曲线。
generate.py：模型加载与温度/top-k 自回归采样。
verify.py：核心正确性和结果复核；download_data.py：可选恢复数据。
data/：原始数据及来源说明。
results/：最佳权重、数据清单、指标、训练日志/CSV、图表、生成样例、复核记录。
screenshots/01_training.png：VS Code 中展示实际训练日志的真实界面截图。

参考：
https://github.com/karpathy/char-rnn （仅使用其示例数据）
https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html

本作业由 Codex 协助实现、实际运行、核验并整理。所有指标和样例均为实测。
