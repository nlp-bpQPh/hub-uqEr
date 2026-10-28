"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务

规律：x是一个5维向量，如果哪个数最大，就是第几类，比如第1个数最大，就是第1类，第2个数最大，就是第2类，以此类推

"""

import os
import random
from typing import Tuple, List, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


class ModelConfig:
    """模型训练配置参数

    包含训练超参数（轮数、批大小、学习率等）、
    数据规模（输入维度、类别数、样本数）和模型保存路径。
    """

    epoch_num = 800  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.0005  # 学习率
    model_file_name = "model.bin"  # 模型文件名
    name_classes = 5  # 类别数量
    test_sample_num = 100  # 测试样本数量

    @staticmethod
    def model_path() -> str:
        """模型保存路径（脚本所在目录）"""
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ModelConfig.model_file_name
        )


class TestModel(nn.Module):
    """线性分类模型

    单层线性网络，输入为 input_size 维特征向量，
    输出 num_classes 个类别的 raw logits，
    使用交叉熵损失函数进行训练。
    """

    def __init__(self, input_size: int, num_classes: int = ModelConfig.name_classes):
        super(TestModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),  # 扩大到 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        self.loss = nn.functional.cross_entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)

    def cal_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算损失值"""
        return self.loss(self(x.float()), y)


class DataTool:
    """数据生成工具类

    负责构造训练样本和批量数据：
    - build_sample: 生成单个样本（5维随机向量 + 最大值类别标签）
    - generate_data: 生成批量训练数据（特征张量 + 标签张量）
    """

    @staticmethod
    def build_sample() -> Tuple[List[float], int]:
        """
        生成单个样本（5维随机向量 + 最大值类别标签）
        包含一定比例的接近值困难样本
        """

        feature = [random.uniform(0, 1) for _ in range(ModelConfig.name_classes)]

        # 30% 概率生成困难样本：让前两大值接近
        if random.random() < 0.3:
            max_idx = feature.index(max(feature))
            second_max_val = max(v for i, v in enumerate(feature) if i != max_idx)
            # 把最大值调整为只比次大值大一点点
            feature[max_idx] = second_max_val + random.uniform(0.001, 0.02)

        label = feature.index(max(feature))
        return feature, label

    @staticmethod
    def generate_data(total_sample_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成训练数据
        """
        features = []
        labels = []
        for _ in range(total_sample_num):
            feature, label = DataTool.build_sample()
            features.append(feature)
            labels.append(label)

        # 输入特征用 FloatTensor，标签用 LongTensor
        return torch.FloatTensor(features), torch.LongTensor(labels)


class TestTool:
    """模型评估与预测工具类

    - evaluate: 在测试集上评估模型准确率
    - predict: 加载训练好的模型，对输入向量进行预测
    - predict_with_exist_num: 使用预设测试向量进行预测
    """

    @staticmethod
    def evaluate(model: nn.Module) -> float:
        """
        计算模型在测试集上的准确率
        """

        model.eval()

        x, y = DataTool.generate_data(ModelConfig.test_sample_num)
        print(f"""本次预测集中共有{len(y)}个样本，
        前5个真实类别:
        {y[:5].tolist()}""")

        with torch.no_grad():
            pred_labels = model(x).argmax(dim=1)  # 取概率最大的类别索引
            correct = (pred_labels == y).sum().item()

        accuracy = correct / len(y)
        print(f"正确预测: {correct}/{len(y)}, 准确率: {accuracy:.4f}")
        return accuracy

    @staticmethod
    def predict(model_path: str, input_vec: torch.Tensor) -> None:
        """
        加载训练好的模型，对输入向量进行预测
        """

        model = TestModel(ModelConfig.input_size)
        model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
        model.eval()  # 测试模式

        with torch.no_grad():  # 不计算梯度
            x = torch.FloatTensor(input_vec)
            if x.dim() == 1:  # 单个样本，增加 batch 维度
                x = x.unsqueeze(0)
            logits = model(x)  # 模型预测，输出 logits
            probs = torch.softmax(logits, dim=1)  # 转概率
            pred_labels = logits.argmax(dim=1)  # 预测类别

        for vec, label, prob in zip(x, pred_labels, probs):
            print(
                f"输入: {vec.tolist()}, 预测类别: {int(label)}, 概率: {prob.tolist()}"
            )

    @staticmethod
    def predict_with_exist_num() -> None:
        """使用预设测试向量进行预测"""
        test_vec = [
            [0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
            [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
            [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
            [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
        ]
        TestTool.predict(ModelConfig.model_path(), test_vec)


class TrainTool:
    """训练过程工具类

    提供训练全流程的工具方法：
    - set_seed: 设置随机种子，保证结果可复现
    - shuffle_data: 打乱训练数据顺序
    - iter_batches: 按 batch 切分数据（生成器）
    - train_one_batch: 训练单个 batch（标准四步：清零→前向→反向→更新）
    - train_one_epoch: 训练一轮，返回平均 loss
    - train: 完整训练循环，返回训练日志
    - plot_training_log: 绘制训练过程的 acc 和 loss 曲线
    """

    @staticmethod
    def set_seed(seed: int = 0) -> None:
        """设置随机种子，保证结果可复现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def shuffle_data(
        train_x: torch.Tensor, train_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """打乱训练数据顺序

        Returns:
            tuple: (shuffled_x, shuffled_y)
        """
        perm = torch.randperm(len(train_x))
        return train_x[perm], train_y[perm]

    @staticmethod
    def iter_batches(
        train_x: torch.Tensor, train_y: torch.Tensor, batch_size: int
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """生成器：按 batch 切分数据，逐个产出 (x, y)

        Args:
            train_x: 特征 [N, ModelConfig.input_size]
            train_y: 标签 [N]
            batch_size: 批大小

        Yields:
            tuple: (batch_x, batch_y)
        """
        num_batches = len(train_x) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            yield train_x[start:end], train_y[start:end]

    @staticmethod
    def train_one_batch(
        model: TestModel, optim: Optimizer, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        """训练一个 batch，返回 loss 值

        标准流程：zero_grad → forward → backward → step
        """
        optim.zero_grad()
        loss = model.cal_loss(x, y)
        loss.backward()
        optim.step()
        return loss.item()

    @staticmethod
    def train_one_epoch(
        model: TestModel,
        optim: Optimizer,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        batch_size: int | None = None,
    ) -> float:
        """训练一轮，返回平均 loss

        Args:
            model: 模型
            optim: 优化器
            train_x: 训练特征 [N, ModelConfig.input_size]
            train_y: 训练标签 [N]
            batch_size: 批大小，默认取 ModelConfig.batch_size
        """
        model.train()
        actual_batch_size: int = batch_size or ModelConfig.batch_size

        # 1. 打乱数据
        train_x, train_y = TrainTool.shuffle_data(train_x, train_y)

        # 2. 遍历所有 batch 训练，用列表推导式收集 loss
        watch_loss = [
            TrainTool.train_one_batch(model, optim, x, y)
            for x, y in TrainTool.iter_batches(train_x, train_y, actual_batch_size)
        ]

        return float(np.mean(watch_loss))

    @staticmethod
    def plot_training_log(log: List[Tuple[float, float]]) -> None:
        """绘制训练过程的 acc 和 loss 曲线"""
        epochs = range(len(log))
        acc_list = [item[0] for item in log]
        loss_list = [item[1] for item in log]

        plt.plot(epochs, acc_list, label="acc")
        plt.plot(epochs, loss_list, label="loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Process")
        plt.legend()
        plt.show()

    @staticmethod
    def train(
        model: TestModel, train_x: torch.Tensor, train_y: torch.Tensor, optim: Optimizer
    ) -> List[Tuple[float, float]]:
        log = []
        for epoch in range(ModelConfig.epoch_num):
            avg_loss = TrainTool.train_one_epoch(model, optim, train_x, train_y)
            acc = TestTool.evaluate(model)
            log.append([acc, avg_loss])

            print(
                f"Epoch {epoch + 1}/{ModelConfig.epoch_num} - loss: {avg_loss:.6f}, acc: {acc:.4f}"
            )

        return log


def main():
    """主函数：设置种子 → 建模型 → 生成数据 → 训练 → 保存 → 画图"""

    # 1. 设置随机种子
    TrainTool.set_seed(0)

    # 2. 建立模型和优化器
    model = TestModel(ModelConfig.input_size)
    optim = torch.optim.Adam(model.parameters(), lr=ModelConfig.learning_rate)

    # 3. 创建训练集
    train_x, train_y = DataTool.generate_data(ModelConfig.train_sample)

    # 4. 训练过程
    log = TrainTool.train(model, train_x, train_y, optim)

    # 5. 保存模型
    torch.save(model.state_dict(), ModelConfig.model_path())
    print(f"模型已保存到: {ModelConfig.model_path()}")

    # 6. 画训练曲线
    TrainTool.plot_training_log(log)


if __name__ == "__main__":
    main()

    # 检测训练后的模型预测效果
    # TestTool.predict_with_exist_num()
