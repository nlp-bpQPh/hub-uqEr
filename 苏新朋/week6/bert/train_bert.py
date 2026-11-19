"""CNews BERT 微调，按课程 CLS 池化与线性分类头结构适配。"""

import random
import time
from pathlib import Path


def train_predict(train, test, labels, model_dir, epochs=1, batch_size=16, max_length=128, output_dir="results"):
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "BERT 依赖导入失败，请检查 torch、transformers、torchvision 版本。"
        ) from exc

    class NewsDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            encoded = tokenizer(
                row["text"], truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )
            return (
                encoded["input_ids"][0], encoded["attention_mask"][0],
                label_to_id[row["label"]],
            )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    label_to_id = {label: i for i, label in enumerate(labels)}

    class BertClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, len(labels))

        def forward(self, input_ids, attention_mask):
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return self.classifier(self.dropout(output.last_hidden_state[:, 0]))

    train_loader = DataLoader(NewsDataset(train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(NewsDataset(test), batch_size=batch_size, shuffle=False)
    model = BertClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(ids, mask), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        print(f"BERT epoch {epoch + 1}/{epochs} loss={loss_sum / max(1, len(train_loader)):.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for ids, mask, _ in test_loader:
            logits = model(ids.to(device), mask.to(device))
            predictions.extend(labels[i] for i in logits.argmax(-1).cpu().tolist())

    elapsed = round(time.perf_counter() - start, 2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "bert_cnews_state.pt"
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "labels": labels,
        "model_dir": str(model_dir),
        "max_length": max_length,
    }, checkpoint)
    return predictions, elapsed, str(checkpoint)
