第三周作业：12 类语种分类模型训练
姓名：Zhechen Tu
日期：2026-10-29

一、任务
输入一段文本，预测它属于 12 个语种中的哪一种。
语言标签：阿拉伯语 arb_Arab、德语 deu_Latn、英语 eng_Latn、法语 fra_Latn、
印地语 hin_Deva、意大利语 ita_Latn、日语 jpn_Jpan、韩语 kor_Hang、
葡萄牙语 por_Latn、俄语 rus_Cyrl、西班牙语 spa_Latn、简体中文 zho_Hans。
题目未附指定数据、语种名单或网络结构，本作业自行选取上述 12 类文本。

二、真实数据与划分
数据来自 Meta / NLLB Team 的 FLORES-200 公开数据集。
原始用途为机器翻译评估，本作业将公开 dev 与 devtest 中的 12 种译文
重新组织为“文本 -> 语种”监督分类任务；不使用机器翻译隐藏测试集。
这些结果是本作业自建划分的语言识别结果，不是官方 FLORES 翻译指标。
来源：https://github.com/facebookresearch/flores/tree/main/flores200
下载：https://tinyurl.com/flores200dataset
引用：NLLB Team (2022), No Language Left Behind: Scaling Human-Centered Machine Translation.

共读取 24,108 条文本。对 NFKC、casefold、空白规范化后重复或为空的
文本，移除其整个平行句组，共移除 2 组、24 条，保留 24,084 条。
以来源文章 URL 为组，固定随机种子 20261029，按文章数 70%/15%/15%
划分训练/验证/测试。同一篇文章及其所有译文只能出现在一个集合中。
训练集：393 篇文章，16,992 条，每类 1,416 条。
验证集： 84 篇文章， 3,564 条，每类   297 条。
测试集： 85 篇文章， 3,528 条，每类   294 条。
三个集合之间无共同文章、平行句 ID 或规范化后相同文本。
数据文件保留原文、语种、句子 ID、原始文章 URL；manifest 保存来源与校验值。
数据许可证 CC BY-SA 4.0，全文见 data/LICENSE_CC_BY_SA.txt。
本目录中重组后的语料同样按 CC BY-SA 4.0 分发；改变为选取 12 类、去重、
按文章重新划分及转为 JSONL，原始文本未改写。

三、模型与训练
1. 输入规范化：Unicode NFKC、大小写折叠、连续空白合并。
2. 字符 1～3 gram TF-IDF，min_df=2，sublinear_tf=True，L2 归一化；
   最多 30,000 个特征。词表和 IDF 仅在训练文本上拟合。
   字符特征无需分词，适用于中文、日文及不同字母体系。
3. PyTorch nn.Linear(30000,12) 输出 logits，训练时 CrossEntropyLoss。
   即可学习的多项逻辑回归/单层 Softmax 网络，参数 360,012 个。
   不调用预训练语种识别器，不通过 Unicode 规则直接指定标签。
4. AdamW，初始学习率 0.01，weight_decay=0.01；CosineAnnealingLR
   在 30 轮后降至 0.0001；batch_size=256；CPU 2 线程。
5. 固定 Python、NumPy、PyTorch 种子，开启确定性算法。
6. 按最低验证交叉熵选择权重；本次为第 30 轮。测试集只在选择权重后
   转换并评估。推理时 Softmax 给出各类分数，取最大者作为输出。

四、实测结果
未训练模型验证准确率：9.1470%。均衡随机猜测基线：8.3333%。
第 1 轮：训练损失 0.67461，验证损失 0.13220。
第 30 轮：训练损失 0.00340，验证损失 0.00584。
最佳权重验证准确率：99.9719%（3,563 / 3,564）。
独立测试准确率：100.0000%（3,528 / 3,528）。
测试 Macro-F1：1.000000；各语种均为 294 / 294。
完整分类报告、混淆矩阵和每轮日志见 results。
训练准确率/损失为每批更新前的在线统计，验证指标为每轮结束的统计。

五、局限与补充实验
测试结果很高，是当前数据规模、领域、类别选择和完整句子条件下的结果，
不能说明任意文本都能正确识别。多个类别的文字体系不同，也降低了难度。
固定模型后，仅把同一测试句子截取前几个 Unicode 字符，进行诊断：
前 5 字符：80.3288%；前 10 字符：87.7834%；前 20 字符：95.7483%。
这些诊断使用原有测试样本，不是新的独立测试集，未用于调参或选择模型。
另有手写 12 语种示例，12/12 预测正确，仅用来演示推理。
混合语种、短词、同形词、拼音/罗马化、其它语种和不同领域仍可能误判。
模型是封闭 12 分类，不具有可靠的未知语种检测功能；Softmax 分数未经
校准，不能当作正确概率保证。纯数字/标点、空文本和完全无特征输入会报错。

六、运行（Python 3.12）
在本文件所在目录执行：
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements_lock.txt

已有数据和权重均随作业提交，不需要重新下载或训练即可预测：
  python classifier.py --predict "我正在学习编程，也喜欢阅读新书。"
  python classifier.py --predict "I enjoy learning new languages."

复核数据隔离、模型重载、短文本诊断与示例：
  python verify.py

重新训练（约数分钟，具体取决于机器；会覆盖 results）：
  python classifier.py
保留既有结果另行训练：
  python classifier.py --output-dir rerun_results
从原始 FLORES 压缩包重新整理数据（可选）：
  python prepare_data.py --archive /path/to/flores200.tar.gz
未指定归档且默认归档不存在时，脚本会从来源地址下载。

七、文件说明
prepare_data.py：真实数据读取、去重、按文章分组划分。
classifier.py：训练、选择权重、测试评估、绘图及命令行预测。
verify.py：隔离检查、模型重载核验、输入检查、短文本诊断和示例。
requirements*.txt：直接依赖与本次运行的完整版本锁定。
data/：三个数据集、来源/划分清单、许可全文。
results/model.pt：真实训练的模型权重。
results/vectorizer.joblib：仅用训练集拟合的字符 TF-IDF。
results/metrics.json：测试及逐类指标。
results/training_history.csv、training_log.txt：每轮真实训练记录。
results/training_curves.png、confusion_matrix.png：由真实指标绘制的图表。
results/test_errors.jsonl：错误预测清单；本次 0 条，所以文件为空。
results/short_text_diagnostics.json：短文本诊断结果。
results/prediction_examples.json：12 个手写示例及模型预测。
results/verification_log.txt：已实际执行的检查输出。

本作业由 Codex 协助编写代码、实际运行训练、核验并整理。
数据来源为公开语料，训练结果为本机实测，没有手工编造准确率。
report.html：可在浏览器中查看的结果报告，展示真实训练/复核日志及图表。
本次执行时 Mac 锁屏，未取得桌面运行截图；已提交原始日志与实际指标图表。
