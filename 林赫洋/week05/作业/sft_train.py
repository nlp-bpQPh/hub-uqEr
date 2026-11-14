"""SFT 训练：把「概率续写器」变成「问答助手」（第 5 周 PPT 第 25-32 页）。

做的事情只有三件（PPT 第 27 页：改数据、不改结构）：
    ① 把数据换成 (提问, 回答) 对，套上 <|user|> / <|assistant|> / <|end|> 模板
    ② 把提问部分的 label 设成 -100，只有回答参与 loss（loss mask）
    ③ 用很小的学习率、很少的 epoch 继续做 next-token 预测

模型结构一个层都没加：还是那个 decoder-only、还是因果掩码、还是下一个词预测。

用法：
    python sft_train.py                 # 默认配置，几分钟跑完
    python sft_train.py --epochs 8      # 多训几轮
    python sft_train.py --quick         # 只跑几十步，先检查流程通不通
"""

import os
import sys
import time
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def parse_args():
    p = argparse.ArgumentParser(description="SFT 指令微调")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="SFT 用小学习率，防止灾难性遗忘")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--mix_ratio", type=float, default=0.1,
                   help="每个样本有多大概率换成通用语料（对抗灾难性遗忘），0=不混")
    p.add_argument("--general_len", type=int, default=256,
                   help="混入通用语料的块长（token）；太长会把整个 batch 补齐变慢")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="只跑少量步数，用来验证流程")
    return p.parse_args()


def _token_texts(tok, ids):
    """把 token 序列变成「每个 token 新贡献的文字」。

    GPT-2 是字节级 BPE：一个汉字常常被切成 2~3 个 token，单独 decode 一个
    token 只能得到半个 UTF-8 字节（显示成乱码）。这里按字节累积，等一个
    汉字凑齐了才显示出来，没凑齐的位置记作 ·（半个字）。
    """
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    u2b = {v: k for k, v in bytes_to_unicode().items()}
    special = set(tok.all_special_ids)
    pieces, buf, shown = [], b"", 0
    for tid in ids:
        tid = int(tid)
        if tid in special:
            pieces.append(tok.convert_ids_to_tokens(tid))
            buf, shown = b"", 0
            continue
        s = tok.convert_ids_to_tokens(tid)
        try:
            buf += bytes(u2b[c] for c in s)
        except KeyError:
            pieces.append(s)
            continue
        text = buf.decode("utf-8", errors="ignore")
        pieces.append(text[shown:])
        shown = len(text)
    return pieces


def show_one_sample(tok, question, answer):
    """把一条样本摊开给人看：哪些 token 被 -100 盖住了（PPT 第 28 页那张图）。"""
    ids, labels = common.encode_sample(tok, question, answer, 512)
    print("\n" + "=" * 72)
    print("一条 SFT 样本长这样（label = -100 的位置不参与 loss）")
    print("=" * 72)
    print(f"原始文本: {common.USER_TAG}{question}{common.ASST_TAG}{answer}{common.END_TAG}")
    print("-" * 72)
    print(f"{'位置':<5}{'token':<20}{'label':<10}说明")
    print("-" * 72)
    # 只看前 28 个 token：提问段 + 回答开头，足够看清 -100 从哪结束
    pieces = _token_texts(tok, ids)
    for i, (tid, lab) in enumerate(zip(ids, labels)):
        if i >= 28:
            print(f"…… 后面还有 {len(ids) - 28} 个 token")
            break
        piece = (pieces[i] or "·").replace("\n", "\\n")
        if lab == common.IGNORE_INDEX:
            note = "提问段，忽略"
        else:
            note = "回答段，算 loss"
        print(f"{i:<5}{piece:<20}{lab:<10}{note}")
    n_sup = sum(1 for lab in labels if lab != common.IGNORE_INDEX)
    print("-" * 72)
    print(f"监督 token: {n_sup}/{len(labels)} = {n_sup / len(labels):.0%}"
          f"　（PPT：预训练 100% 有信号，SFT 只有回答段有信号）")


def compute_loss(model, input_ids, labels, attention_mask):
    """因果 LM 的 loss：第 t 个位置输出，用来预测第 t+1 个 token。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=common.IGNORE_INDEX,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for ids, labels, mask in loader:
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        loss = compute_loss(model, ids, labels, mask)
        total += loss.item()
        count += 1
    return total / max(1, count)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = common.pick_device(args.device)
    common.log(f"设备: {device}")

    # ① 数据：先确认存在，没有就先跑 build_sft_data.py
    if not os.path.isfile(common.SFT_DATA):
        raise FileNotFoundError(
            f"缺少指令数据 {common.SFT_DATA}\n先运行: python build_sft_data.py"
        )
    rows = common.read_jsonl(common.SFT_DATA)
    # 按 pair_id 整组切分：同一个问题的三种问法不会一个在训练一个在验证
    train_rows, val_rows = common.split_pairs(
        rows, args.val_ratio, args.seed, group_key=lambda r: r.get("pair_id", r["question"])
    )
    train_pairs = [(r["question"], r["answer"]) for r in train_rows]
    val_pairs = [(r["question"], r["answer"]) for r in val_rows]
    common.log(f"指令数据: {len(rows)} 条（{len({r.get('pair_id') for r in rows})} 个问题）"
               f" = 训练 {len(train_pairs)} + 验证 {len(val_pairs)}")

    # ② 模型：基座 + 3 个特殊 token（结构不变，只扩了词表）
    tok, model = common.load_base_model(device)
    for p in model.parameters():
        p.requires_grad = True
    n_param = sum(p.numel() for p in model.parameters())
    common.log(f"基座模型: {common.MODEL_ID}  参数 {n_param / 1e6:.0f}M  词表 {len(tok)}")

    show_one_sample(tok, train_pairs[0][0], train_pairs[0][1])

    # ③ 数据集
    train_ds = common.SFTDataset(train_pairs, tok, args.max_len)
    val_ds = common.SFTDataset(val_pairs, tok, args.max_len)

    general_ds = None
    if args.mix_ratio > 0 and os.path.isfile(common.GENERAL_CORPUS):
        with open(common.GENERAL_CORPUS, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        general_ds = common.GeneralDataset(text, tok, args.max_len, args.general_len)
        common.log(f"通用语料: {len(general_ds)} 块，按 {args.mix_ratio:.0%} 概率混入")

    pad_id = tok.pad_token_id

    def collate(batch):
        return common.collate(batch, pad_id)

    class Mixed(torch.utils.data.Dataset):
        """以一定概率把 SFT 样本换成通用语料块，防止模型只认识问答格式。"""

        def __len__(self):
            return len(train_ds)

        def __getitem__(self, i):
            if general_ds is not None and random.random() < args.mix_ratio:
                return general_ds[random.randrange(len(general_ds))]
            return train_ds[i]

    train_loader = DataLoader(Mixed(), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    # ④ 优化器 + 调度：小学习率、warmup、余弦退火、梯度裁剪
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    if args.quick:
        total_steps = min(total_steps, 30)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, common.lr_lambda_factory(total_steps, args.warmup_ratio)
    )

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    common.log(f"混合精度: {'bf16' if use_bf16 else '关闭(fp32)'}")

    os.makedirs(common.OUT_DIR, exist_ok=True)
    best_val = float("inf")
    step = 0
    t0 = time.time()

    common.log("\n" + "-" * 72)
    common.log(f"{'epoch':>6}{'train_loss':>12}{'val_loss':>12}{'lr':>12}{'用时':>10}")
    common.log("-" * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for ids, labels, mask in train_loader:
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = compute_loss(model, ids, labels, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()

            total += loss.item()
            count += 1
            step += 1
            if args.quick and step >= total_steps:
                break

        train_loss = total / max(1, count)
        val_loss = evaluate(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val = val_loss

        common.log(f"{epoch:>6}{train_loss:>12.4f}{val_loss:>12.4f}"
                   f"{cur_lr:>12.2e}{time.time() - t0:>9.0f}s")

        if args.quick and step >= total_steps:
            common.log("--quick 模式，提前结束")
            break

    common.log("-" * 72)
    common.log(f"训练完成，最后一轮验证损失 {val_loss:.4f}（最低 {best_val:.4f}）")
    if args.quick:
        common.log("--quick 模式只验证流程，不保存模型、不做前后对比"
                   "（免得用没训完的权重覆盖已有的模型）。")
        return

    # 保存「训练结束时的模型」而不是验证损失最低的那个：
    # 本作业只有 250 条数据，验证损失从第 1 轮就开始反弹（过拟合），
    # 按 val 挑会永远挑到几乎没训过的模型，反而演示不出来 SFT 的效果。
    os.makedirs(common.SFT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(common.SFT_MODEL_DIR)
    tok.save_pretrained(common.SFT_MODEL_DIR)

    common.log(f"模型已保存到 {common.SFT_MODEL_DIR}")
    if device.type == "cuda":
        common.log(f"显存峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ⑤ 训练完立刻对比：基座 vs SFT（PPT 第 26 页那张对照表）
    compare_and_report(tok, device, args)


def compare_and_report(tok, device, args):
    """同一个问题，分别问「只预训练过的基座」和「SFT 之后的模型」。"""
    prompts = [
        "SFT是什么？",
        "loss mask是什么？",
        "梯度下降是什么？",
        "你好",
        "BERT和GPT有什么区别？",   # 数据里没有，专门用来看「SFT 不会凭空涨知识」
    ]

    common.log("\n" + "=" * 72)
    common.log("同题对比：基座模型（续写器） vs SFT 模型（问答助手）")
    common.log("=" * 72)

    base_tok, base_model = common.load_base_model(device)
    sft_tok, sft_model = common.load_sft_model(device)

    lines = ["# SFT 前后对比\n", "同一个问题，分别问两个模型（temperature=0.7）：\n"]
    for q in prompts:
        b = common.generate(base_model, base_tok, q, device, max_new_tokens=48)
        s = common.generate(sft_model, sft_tok, q, device, max_new_tokens=48)
        common.log(f"\n问：{q}")
        common.log(f"  基座  ：{b[:60]!r}")
        common.log(f"  SFT后 ：{s[:60]!r}")
        lines.append(f"## 问：{q}\n\n- **基座**：{b or '（空）'}\n- **SFT 后**：{s or '（空）'}\n")

    report = os.path.join(common.OUT_DIR, "sft_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    common.log(f"\n对比结果已写入 {report}")
    common.log("想继续对话：python chat.py")


if __name__ == "__main__":
    main()
