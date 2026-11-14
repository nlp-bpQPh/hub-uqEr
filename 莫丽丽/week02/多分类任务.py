"""
基于 PyTorch 的多分类找规律任务
规律：x 是 5 维向量，按优先级规则映射到 5 个风险等级 {低, 偏低, 中, 高, 极高}
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==================== 全局常量配置 ====================
CLASS_NAMES = ["低风险", "偏低风险", "中风险", "高风险", "极高风险"]
NUM_CLASSES = 5
INPUT_SIZE = 5


# ==================== 1. 模型定义 ====================
class TorchModel(nn.Module):
    """
    多层感知机（MLP）分类模型
    结构：Input (5) -> Linear (32) -> ReLU -> Linear (NUM_CLASSES) -> CrossEntropy / Softmax
    """
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, num_classes)
        self.loss = nn.functional.cross_entropy  # 多分类交叉熵

    def forward(self, x, y=None):
        """
        前向传播函数
            :param x: 输入张量，形状为 (Batch_Size, INPUT_SIZE)
            :param y: 真实标签张量
            :return: 训练模式返回 Loss 标量，预测模式返回归一化后的概率分布
        """
        h = self.relu(self.linear1(x))  # 隐藏层特征提取：(B, 32)
        logits = self.linear2(h)  # 输出层未归一化对数概率：(B, 5)

        if y is not None:
            return self.loss(logits, y)  # 训练阶段：返回交叉熵损失
        else:
            return torch.softmax(logits, dim=1)  # 推理阶段：转换成概率分布 (B, 5)


# ==================== 2. 标签生成规则 ====================
def classify_risk(x):
    """
    基于业务逻辑的确定性规则分类器
        :param x: 长度为 5 的一维数组或列表
        :int: 返回对应的风险等级标签（0 到 4）
    """
    x0, x1, x2, x3, x4 = x[0], x[1], x[2], x[3], x[4]

    # 类4 极高
    if (x0 > 0.85 and x1 > 0.85) or (x2 > 0.90 and x0 > 0.80) or (x4 > 0.90 and x1 > 0.80):
        return 4
    # 类3 高
    if (x0 > 0.70 and x1 > 0.60) or (x2 > 0.80 and x3 < 0.25) or (x4 > 0.80 and x0 > 0.60):
        return 3
    # 类2 中
    if (x0 > 0.60) or (x1 > 0.60) or (x2 > 0.70) or (x3 < 0.20 and x4 > 0.50):
        return 2
    # 类1 偏低
    if (x0 > 0.40) or (x1 > 0.40) or (x2 > 0.45) or (x4 > 0.60):
        return 1
    # 类0 低
    return 0


# ==================== 3. 样本与数据集构建 ====================
def build_sample(target_class=None, max_tries=100000):
    """
    单样本构建函数（支持按类别拒绝采样）
        :param target_class: 指定的目标类别（0-4），若为 None 则随机生成
        :param max_tries: 最大采样尝试次数，防止死循环
        :return: 样本特征数组 x 与标签 y
    """
    for _ in range(max_tries):
        x = np.random.random(INPUT_SIZE)
        y = classify_risk(x)
        if target_class is None or y == target_class:
            return x, y
    raise RuntimeError(f"类别 {target_class} 采样失败，请增大 max_tries")


def build_dataset(total_sample_num):
    """
    构建均衡的数据集
        :param total_sample_num: 总样本数
        :return: 打乱后的特征张量 X 与标签张量 Y
    """
    per_class = total_sample_num // NUM_CLASSES
    X, Y = [], []

    # 确保各类别样本均衡
    for label in range(NUM_CLASSES):
        for _ in range(per_class):
            x, y = build_sample(target_class=label)
            X.append(x)
            Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    # 随机打乱数据集顺序
    idx = np.random.permutation(len(X))
    return torch.FloatTensor(X[idx]), torch.LongTensor(Y[idx])


# ==================== 4. 模型评估 ====================
def evaluate(model):
    """
    模型评估函数：计算测试集上的准确率、混淆矩阵及各类别召回率
        :param model: 训练中的 PyTorch 模型实例
        :return: acc (float) 准确率
    """
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    correct = 0

    with torch.no_grad():
        y_pred = model(x)
        y_pred_cls = torch.argmax(y_pred, dim=1)
        for p, t in zip(y_pred_cls, y):
            p, t = int(p), int(t)
            confusion[t][p] += 1
            if p == t:
                correct += 1

    total = len(y)
    acc = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    print("混淆矩阵（行=真实，列=预测）：")
    print(confusion)

    # 打印每个类别的召回率
    for c in range(NUM_CLASSES):
        recall = confusion[c][c] / confusion[c].sum() if confusion[c].sum() > 0 else 0.0
        print("  %s 召回率：%.4f" % (CLASS_NAMES[c], recall))
    return acc


# ==================== 5. 训练主流程 ====================
def main():
    epoch_num = 40
    batch_size = 32
    train_sample = 10000
    learning_rate = 0.005

    # 初始化模型与优化器
    model = TorchModel(INPUT_SIZE, NUM_CLASSES)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 准备训练数据集
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        # 每个 Epoch 开始前打乱数据
        perm = torch.randperm(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        watch_loss = []
        # Mini-batch 训练迭代
        for batch_index in range(len(train_x) // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]

            loss = model(x, y)  # 前向传播计算损失
            loss.backward()  # 反向传播计算梯度
            optim.step()  # 更新模型参数
            optim.zero_grad()  # 清空历史梯度

            watch_loss.append(loss.item())

        print("="*50+"\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型权重
    torch.save(model.state_dict(), "risk_model.bin")
    print(log)

    # ======================== 绘图 ========================
    # 全局字体与负号配置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(log) + 1)
    accs = [l[0] for l in log]
    losses = [l[1] for l in log]

    # 创建画布：上下两子图共享 X 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

    # 子图 1：Loss 曲线
    loss_color = '#d62728'
    ax1.plot(epochs, losses, color=loss_color, linewidth=1.8, marker='o', markersize=3, label='Training Loss')
    ax1.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=loss_color, labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True)
    ax1.set_title('Training Dynamics and Performance Curves', fontsize=13, fontweight='bold', pad=10)

    # 子图 2：Accuracy 曲线
    acc_color = '#1f77b4'
    ax2.plot(epochs, accs, color=acc_color, linewidth=1.8, marker='s', markersize=3, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=acc_color, labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', shadow=True)

    # 自动调整子图间距
    plt.tight_layout()

    # 保存图片
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("图表已优化并成功保存为 'training_curves.png'")

    plt.show()


# ======================6. 预测 ========================
def predict(model_path, input_vec):
    """
    加载已保存的模型权重对新样本进行推理预测
        :param model_path: 模型权重文件路径 (.bin)
        :param input_vec: 待预测的输入向量列表或数组
    """
    model = TorchModel(INPUT_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))

    for vec, res in zip(input_vec, result):
        prob = res.numpy()
        pred = int(np.argmax(prob))
        print("输入：%s -> 预测类别:【%s】, 各类概率: %s" % (
            np.round(vec, 3),
            CLASS_NAMES[pred],
            np.round(prob, 4),
        ))


if __name__ == "__main__":
    main()

    # 5 条手工测试向量，覆盖 5 个类别
    test_vec = [
        [0.10, 0.10, 0.10, 0.50, 0.10],  # 类0 低
        [0.45, 0.10, 0.10, 0.50, 0.10],  # 类1 偏低
        [0.65, 0.10, 0.10, 0.50, 0.10],  # 类2 中
        [0.75, 0.65, 0.40, 0.50, 0.10],  # 类3 高
        [0.90, 0.90, 0.10, 0.50, 0.70],  # 类4 极高
    ]
    predict("risk_model.bin", test_vec)
