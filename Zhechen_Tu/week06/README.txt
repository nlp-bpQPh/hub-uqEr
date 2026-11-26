第六周作业：更换数据集，比较多种文本分类方法
Zhechen Tu · 2026-11-18

一、任务选择
本次选择题目中的“更换数据集”，从前几周的语种识别、戏剧文本、合成指令任务
改为真实新闻主题分类。使用 AG News 的 World、Sports、Business、Sci/Tech 四类。
比较五种方法的分类效果、随机初始化稳定性、资源开销和错误类型。
本次五种方法均不使用预训练权重，两个神经网络的词向量从随机初始化学习。
它不是“全量微调/LoRA/提示学习”的预训练模型比较；所完成的是更换数据集后的
传统分类器与神经分类器对照。

二、数据来源及实验范围
数据原始收集者 Antonio Gulli；四分类版本由 Xiang Zhang 等整理。
论文：Xiang Zhang, Junbo Zhao, Yann LeCun (2015), Character-level Convolutional
Networks for Text Classification, NeurIPS 2015。
数据镜像：https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
数据卡：https://huggingface.co/datasets/fancyzhx/ag_news
原始说明将数据用于研究及非商业活动；数据卡的许可证字段标为 unknown，
本作业不将新闻原文重新声明为 MIT 或其他开源许可。保留数据出处供教学复现。

原始 train=120,000，test=7,600。输入为标题与描述以空格拼接，标签减一为 0..3。
规范化包括 HTML 实体解码、Unicode NFKC、小写、空白合并。
规范化文本完全相同视为重复，保留原始出现的第一条。
训练内部移除 151 条重复；训练与测试重合的 10 条从训练候选池中移除；
测试内部无重复，因此完整保留官方的 7,600 条测试记录。
从剩余 119,839 条训练候选中，按类固定种子 20261118 打乱抽样：
训练 12,000 条（每类 3,000），验证 2,000 条（每类 500）。
其余训练候选不使用，控制本机实验规模。测试每类 1,900 条。
训练、验证、测试之间没有规范化后的完全重复文本。
未检测语义近重复或相同新闻事件；该限制可能导致对泛化能力的乐观估计。
数据 JSONL 包含原始文件行 ID；manifest 保存来源文件 SHA-256 和各划分 ID 哈希。

三、公平比较与方法设置
五种方法共用训练/验证/测试记录及原始标签；所有词表、IDF 只在训练集建立。
超参数在运行前固定，没有利用测试结果调整设置。全部模型训练/选权重结束后，
统一计算测试指标。训练集均衡，多数类固定猜测基线准确率为 25%。

传统方法共用同一个 TF-IDF：word 1~2 gram，max_features=40,000，min_df=2，
sublinear_tf=True，L2 归一化；输入全文；scikit-learn 默认分词规则。
1. MultinomialNB，alpha=0.5：以特征条件独立假设估计类别。
2. LogisticRegression，C=4，lbfgs，max_iter=500：学习多类线性决策边界。
3. LinearSVC，C=1，dual=auto，max_iter=5000：学习线性最大间隔决策边界。
传统模型每种运行一次，固定划分上的实现没有需要重复采样的初始化实验；
不把标准差标为零，不声称它们在不同数据划分下没有不确定性。

神经网络共用仅由训练集构建的最多 15,000 词词表（含 PAD/UNK）、词序列长度 100，
小写英文/数字正则分词；不足长度右填充，超长截断。
4. MeanPool：64 维词向量，忽略 padding 的平均池化，64 维 ReLU 隐藏层、四分类输出。
5. TextCNN：64 维词向量，卷积窗口 3/4/5，各 64 个卷积核，ReLU、最大池化，
拼接为 192 维后分类；池化排除纯 padding 窗口。
两个网络均使用 dropout=0.3，AdamW(lr=0.002,weight_decay=0.01)，batch=128，
梯度裁剪 1.0，训练 10 轮。种子 17/29/43 各运行一次；每次按验证 Macro-F1
选最佳轮次，出现相同分数保留最早轮次。本次六次运行均选到第 10 轮。
神经方法报告三个种子的均值和样本标准差（ddof=1）；未做种子集成。

特征处理是方法的一部分，未强行统一：TF-IDF 使用全文、默认词正则；神经网络
使用另外的词正则和前 100 词。测试集 21/7,600 条超过 100 词，因此比较的是
完整分类方案，不能把差异完全归因于分类头或网络结构。

四、实际结果（同一个测试集，百分比）
方法                    Accuracy        Macro-F1       F1 标准差（百分点）
TF-IDF + NB              88.83            88.79          单次，不估计
TF-IDF + LR              89.67            89.64          单次，不估计
TF-IDF + Linear SVM      89.89            89.86          单次，不估计
MeanPool（三种子均值）    87.72            87.70          0.465
TextCNN（三种子均值）     87.31            87.28          0.379
逐次结果、逐类 precision/recall/F1 以及混淆矩阵均随作业提交。

五、耗时与资源
传统拟合在 CPU 进行（线程限制 4），神经训练在 Apple MPS 进行。
模型拟合用时：NB 约 0.004 秒，LR 约 0.638 秒，SVM 约 0.096 秒；
MeanPool 三种子平均约 5.85 秒，TextCNN 平均约 8.26 秒。
这些数值不含公共特征构建；神经训练用时包括逐轮验证与保存待选权重的内存拷贝，
传统拟合用时不含验证预测。后端也不同，因此不能直接当作硬件无关的训练速度排名。
特征构建时间另存 results/settings.json。

为比较部署推理，统一在本机 CPU、4 线程、相同 7,600 条输入上计时：
包含规范化与特征转换，不含模型文件加载；预热后运行三遍取中位数。
NB 0.205 秒；LR 0.190 秒；SVM 0.174 秒；MeanPool 0.134 秒；TextCNN 4.415 秒。
这是批处理吞吐测试，不是单条在线请求延迟；不同硬件、线程数、batch 会改变结果。
神经方法计时使用验证分数最高的种子 17；不是根据测试分数挑选。
模型文件大小见 runs.csv；传统方法还需共享 tfidf.joblib，神经方法还需 vocab.json。

六、结论与不确定性
在该 12,000 条训练子集、固定参数和 10 轮预算下，TF-IDF 线性模型效果较好。
新闻中的主题关键词适合稀疏特征；随机词向量的神经网络需要学习表示，
本次训练预算和超参数未保证达到其最佳表现，因此不代表神经网络普遍更差。
SVM 比 LR 的准确率仅高 0.224 个百分点（17/7,600 个净正确数）。
针对同批样本做 10,000 次配对 bootstrap，差值 95% 区间为
[-0.079, 0.539] 个百分点；双侧精确 McNemar p=0.1898。
这是事后探索性比较，未作多模型比较校正；区间包含零，不能据此认定 SVM
显著优于 LR。不同新闻来源、时间段及训练抽样可能改变排序。
本实验没有比较预训练 BERT 或 LoRA，不对这些方法作效果结论。

七、复现与推理（Python 3.12，在本目录执行）
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

已有数据和全部九次运行的权重，无需下载或重新训练即可推理：
  python predict.py --method LinearSVM --text "The team won the championship after a close final."
  python predict.py --method TextCNN --text "A new processor improves computer performance."
可选方法：NaiveBayes、LogisticRegression、LinearSVM、MeanPool、TextCNN。
神经方法默认按验证分数选代表种子；未支持未知类别检测，非新闻输入仍会被强制分类。

复核九套模型的预测、指标、划分隔离、选轮依据及 CPU 时间：
  python verify.py
重新生成对照表、图表、误判记录与统计诊断：
  python report.py
重新训练会覆盖 results 中同名模型和结果，先复制目录保留旧实验：
  python experiment.py --device mps
无 MPS 时用 --device cpu；也支持 --device cuda。
如需恢复原始下载并重新创建相同子集：
  python data.py --cache /path/to/ag_news_cache
已有 manifest 时会校验下载文件哈希；默认下载至本目录 cache，不属于提交文件。
固定种子并不保证不同计算后端和版本逐位一致；CPU 重载核验本次九组所有预测一致。

八、文件索引
README.txt：本实验报告。
data.py / data/：下载、清洗、划分、原始行 ID 与来源清单、实际子集。
experiment.py：五方法训练、验证选轮、统一测试。
predict.py / verify.py / report.py：推理、可重复验证、报告生成。
results/models/：共享特征器/词表、3 个传统分类器、6 个神经检查点。
results/runs.csv / summary.json：逐运行与方法汇总指标。
results/history.csv / training_log.txt：神经训练曲线原始数值和实测日志。
results/predictions.csv：全部九次运行、每条测试记录的预测。
results/class_reports.json / errors.jsonl：逐类结果、代表模型全部错误样例。
results/cpu_inference.csv / paired_comparison.json：统一 CPU 耗时和统计诊断。
results/comparison.png / confusion_matrices.png：实测对照图与混淆矩阵。
results/verification_log.txt：实际执行的模型重载与数据检查输出。
report.html：由以上数值生成的可读本地报告。

参考：Yoon Kim (2014), Convolutional Neural Networks for Sentence Classification,
https://arxiv.org/abs/1408.5882 。本作业为随机词向量的简化 TextCNN，不是论文完整复现。
https://scikit-learn.org/1.5/modules/feature_extraction.html

由 Codex 协助实现、运行和整理。数据、权重、指标、误判样例均可检查和复算。

误差观察：SVM 将 170 条商业新闻判为科技、150 条科技新闻判为商业；
二者合计 320 条，占其 768 条错误的 41.7%。验证选出的 TextCNN seed=17
相应为 175 条和 222 条，共 397 条。这与企业产品、科技公司财报等主题交叉一致，
但这里是对混淆统计的解释，不代表对全部新闻标签进行了人工复核。
体育是两者较容易区分的类别，SVM 体育召回为 1853/1900，TextCNN 为 1829/1900。

结果表中传统方法的 seed=0 是单次运行标记，不表示重新设置随机种子；LinearSVC 的 random_state 固定为 17。
