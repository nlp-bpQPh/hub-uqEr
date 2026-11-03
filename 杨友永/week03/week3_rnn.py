"""
多语言识别（单标签）
"""

import os
import pandas as pd
import torch
from typing import List, Dict, Tuple
from itertools import groupby, islice
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F


class DataloaderTool:  # 数据集加载工具
    # fmt: off
    CLASSES: List[str] = ["ar","en","es","hi","ja","pt","ru","zh","ko","th","vi","tr",]
    # fmt: on

    # 类别名 → ID 映射
    CLS2ID: Dict = {name: i for i, name in enumerate(CLASSES)}

    @staticmethod
    def load_data(data_path: str = None) -> List[Dict[str, str]]:

        train_df = pd.read_csv(data_path)
        data = [
            {"cls_index": DataloaderTool.CLS2ID[cls_name], "text": text}
            for cls_name, text in zip(train_df["label"], train_df["text"])
        ]
        return data

    @staticmethod
    def load_sample_data() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data/v12")

        train_data = DataloaderTool.load_data(os.path.join(data_dir, "train.csv"))
        test_data = DataloaderTool.load_data(os.path.join(data_dir, "test.csv"))
        return train_data, test_data

    @staticmethod
    def build_vocab(data: List[Dict[str, str]]) -> Dict[str, int]:
        vocab = {"<pad>": 0, "<unk>": 1}
        for item in data:
            for ch in item["text"]:
                vocab.setdefault(ch, len(vocab))
        return vocab


class NewsDataset(Dataset):
    """将 (文本, 标签) 列表转为 Dataset。

    Args:
        data: [{text: str, cls_index: int}, ...]
        stoi: 字表字典 {字: ID}

    内部状态:
        self.X: list[Tensor]，每个元素形状 [Ti]（长度不一）
        self.y: LongTensor，形状 [N]

    注:
        长度对齐由 collate_fn 的 pad_sequence 完成，Dataset 本身不做 padding。
    """

    def __init__(self, data: List[Dict[str, str]], stoi: Dict[str, int]):
        self.X = [
            torch.tensor([stoi.get(ch, stoi["<unk>"]) for ch in item["text"]])
            for item in data
        ]
        self.y = torch.tensor([item["cls_index"] for item in data])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        """
        collate_fn 的作用解释:

        步骤	        操作	                            结果
        zip(*batch)	   把 {text, cls_index} 拆成两个元组	 xs=(文本列表), ys=(标签列表)
        pad_sequence	把所有文本补到相同长度（补 0）	       [B, max_len]
        torch.stack	    把标签堆成一个张量	                 [B]
        """
        text, cls_index = zip(*batch)
        return pad_sequence(text, batch_first=True), torch.stack(cls_index)


class LSTMModel(nn.Module):
    def __init__(
        self, vocab_size, embed_size=128, hidden_size=256, num_classes=12, dropout=0.3
    ):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, batch_first=True, bidirectional=True
        )  # 双向
        self.linear = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.linear(x.max(dim=1).values)
        return x


class RNNModel(nn.Module):
    def __init__(
        self, vocab_size, embed_size=128, hidden_size=256, num_classes=12, dropout=0.3
    ):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = x.max(dim=1).values
        x = self.linear(x)

        return x


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_classes=12, dropout=0.3):
        super(CNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_size, 128, kernel_size=3, padding=1)
        self.linear = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # fmt: off
        # x: [B, T]，数据进来后一步步变换
        x = self.embedding(x).transpose(1, 2)       # shape: [B, embed_size, T]
        x = self.dropout(x)                         # shape: [B, embed_size, T]
        x = self.conv1(x)                           # shape: [B, 128, T]
        x = F.relu(x)                               # shape: [B, 128, T]
        x = x.max(dim=2).values                     # [B, hid]
        return self.linear(x)                       # [B, num_classes]
        # fmt: on


def main():
    # 加载数据
    train_data, test_data = DataloaderTool.load_sample_data()

    # 构建字表
    stoi = DataloaderTool.build_vocab(train_data)
    V = len(stoi)
    print(f"字表大小: {V}")

    # 3. DataLoader
    train_dl = DataLoader(
        NewsDataset(train_data, stoi),
        batch_size=128,
        shuffle=True,
        collate_fn=NewsDataset.collate_fn,
    )
    test_dl = DataLoader(
        NewsDataset(test_data, stoi), batch_size=256, collate_fn=NewsDataset.collate_fn
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备: {device}")
    num_classes = len(DataloaderTool.CLASSES)
    # 模型F1对比：CNN： 0.9917  LSTM：0.9905  RNN：0.9850
    model = CNNModel(V, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(12):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    # 验证
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # 准确率
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    # 手动算 macro F1
    f1_per_class = []
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_per_class.append(f1)

    f1_macro = sum(f1_per_class) / len(f1_per_class)

    print(f"Epoch {epoch}: acc={acc:.4f}  f1_macro={f1_macro:.4f}")


if __name__ == "__main__":
    main()
