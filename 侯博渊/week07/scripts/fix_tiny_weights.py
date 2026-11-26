# -*- coding: utf-8 -*-
"""
把 structbert-tiny 的预训练权重整理成标准 HF BertModel 目录。

背景（真实踩坑）：week6 的 structbert-tiny/pytorch_model.bin 是 BertForPreTraining 风格，
state_dict 的 key 带 `bert.` 前缀、还带 `cls.*` 头。直接
`BertModel.from_pretrained(dir)` 会认为所有 BERT 权重缺失 → **随机初始化**，
训练 loss 降得动，但 F1 一直是 0（本项目实测两个 epoch 后 spanF1 仍为 0）。

修法：读 state_dict、去掉 `bert.` 前缀、丢掉 cls.*，再 save_pretrained 成标准目录。
"""

from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

SRC = Path(r"E:\python\第六周 分类任务\week6-数据接入与分流\_models\structbert-tiny")
DST = Path(r"E:\python\第七周 抽取任务\hw_duee10\pretrain_models\structbert-tiny")


def main():
    cfg = BertConfig.from_pretrained(SRC)
    model = BertModel(cfg)
    sd = torch.load(SRC / "pytorch_model.bin", map_location="cpu", weights_only=True)
    stripped = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print("missing:", list(missing)[:5], "unexpected:", list(unexpected)[:5])
    assert not [m for m in missing if "cls" not in m and "pooler" not in m], missing
    model.save_pretrained(DST)
    tok = BertTokenizerFast.from_pretrained(SRC)
    tok.save_pretrained(DST)
    print("saved:", DST)
    print("params(M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    print("files:", [p.name for p in DST.iterdir()])


if __name__ == "__main__":
    main()
