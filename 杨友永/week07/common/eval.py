"""通用评估：逐 token argmax 的标注模型（BERT-linear / LLM-LoRA）共用。"""
from __future__ import annotations

import torch

from common import metrics as M


@torch.no_grad()
def evaluate_logits(model, loader, device, id2label: dict[int, str],
                    take_token_type_ids: bool = True) -> dict:
    model.eval()
    gold_all, pred_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if take_token_type_ids:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

        logits = model(**kwargs)["logits"]
        preds = logits.argmax(dim=-1).cpu()
        labels = batch["labels"]
        for b in range(labels.size(0)):
            valid = labels[b] != -100
            gold_all.append([id2label[int(i)] for i in labels[b][valid]])
            pred_all.append([id2label[int(i)] for i in preds[b][valid]])
    return M.evaluate(gold_all, pred_all)
