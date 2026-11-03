# 12 语种识别模型对比实验报告

## 1. 实验目的

识别12语种，并在在语种识别任务上比较循环网络结构和两个训练模块的作用：

1. 将基线 RNN 替换成单向 LSTM；
2. 将基线 RNN 替换成双向 LSTM；
3. 在 RNN 基线上单独去掉 LayerNorm；
4. 在 RNN 基线上单独去掉 Dropout。

## 2. 实验设置

| 项目               | 设置                                                    |
| ------------------ | ------------------------------------------------------- |
| 数据               | 项目内 `data/langid/v12/`                               |
| 语种类别           | 12 类                                                   |
| 训练 / 验证 / 测试 | 9,600 / 1,200 / 1,800 条                                |
| Epoch              | 8                                                       |
| 随机种子           | 42，所有实验相同                                        |
| 批大小             | 64                                                      |
| 学习率             | 0.001，Adam                                             |
| 最大输入长度       | 40 个字符                                               |
| 指标               | Test Accuracy、Macro-F1、Dev Accuracy、参数量、训练耗时 |

所有模型复用相同的数据划分与字符表。RNN、LSTM 和 BiLSTM 之间改变循环层类型，两个消融实验只关闭指定模块。BiLSTM 使用 packed sequence，避免反向 LSTM 将右侧 padding 当作真实上下文。

## 3. 实测结果

完整数据见 `comparisons/v12/summary.csv`和 `comparisons/v12/summary.json`。

## 4. 复现实验

在项目根目录运行：

```python
# 默认训练 12 类 RNN
python train_langid_cls.py

# 训练单向 LSTM
python train_langid_cls.py --model-type lstm --save langid_lstm_v12.pt

# 训练双向 LSTM
python train_langid_cls.py --model-type bilstm --save langid_bilstm_v12.pt

# RNN 消融实验：只关闭 LayerNorm
python train_langid_cls.py --no-layernorm --save langid_rnn_no_layernorm_v12.pt

# RNN 消融实验：只关闭 Dropout
python train_langid_cls.py --no-dropout --save langid_rnn_no_dropout_v12.pt

# 如果目的只是跑五组对比，直接运行 compare_models.py 即可，不必再手动单独启动此脚本
python compare_models.py
```

用 Excel 查看总结果：打开 `comparisons/v12/summary.csv`。

## 5. 各模型的独立评测

```bash
python eval_langid.py --model comparisons/v12/models/rnn_baseline.pt --data-dir data/langid/v12 --split test
python eval_langid.py --model comparisons/v12/models/lstm.pt --data-dir data/langid/v12 --split test
python eval_langid.py --model comparisons/v12/models/bilstm.pt --data-dir data/langid/v12 --split test
python eval_langid.py --model comparisons/v12/models/rnn_no_layernorm.pt --data-dir data/langid/v12 --split test
python eval_langid.py --model comparisons/v12/models/rnn_no_dropout.pt --data-dir data/langid/v12 --split test
```

