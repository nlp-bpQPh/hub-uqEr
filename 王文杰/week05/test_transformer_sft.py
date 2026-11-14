"""
SFT 模型验证脚本：在 sft_val.jsonl 上测试"给定 instruction，模型能否生成正确 output"。

用法:
    python test_transformer_sft.py
    python test_transformer_sft.py --model_file best_Transformer_sft.pt --num 10 --max_new 60
"""

import argparse
import json
import torch
import torch.nn.functional as F

# 复用训练脚本里的 LM 定义，保证权重 key 严格一致
from language_transformer_sft_model import LM, SFT_SEP_INSTR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    """从 SFT checkpoint 还原模型（含词表与超参）"""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["args"]
    char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
    model = LM(
        vocab_size=len(char2idx),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=0.0,                       # 推理关闭 dropout，保证确定性
        nhead=cfg.get("nhead", 4),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"已加载 {path}  (词表 {len(char2idx)} 字, nhead={cfg.get('nhead', 4)})")
    return model, char2idx, idx2char


@torch.no_grad()
def generate_fixed(model, char2idx, idx2char, device, prompt, max_new=60, seq_len=64):
    """自回归逐字生成：喂 instruction+分隔符，让模型续写 output 部分"""
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return "（提示词里的字都不在字表里，无法生成）"
    ctx = torch.tensor([ids], device=device)
    out = []
    for _ in range(max_new):
        logits = model(ctx[:, -seq_len:])[0, -1]
        nxt = torch.argmax(logits).item()
        ch = idx2char[nxt]
        if ch == "\n":          # 遇到结束符就停
            break
        out.append(ch)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="best_Transformer_sft.pt")
    parser.add_argument("--val_file",   default="sft_val.jsonl")
    parser.add_argument("--num",        type=int, default=10, help="batch 模式抽查多少条")
    parser.add_argument("--max_new",    type=int, default=60, help="每条最多生成字数")
    parser.add_argument("--seq_len",    type=int, default=64)
    parser.add_argument("--batch",      action="store_true", help="批量跑验证集（而非交互）")
    args = parser.parse_args()

    model, char2idx, idx2char = load_model(args.model_file)
    sep = SFT_SEP_INSTR

    # ── 批量模式：跑验证集抽查（带期望对比）──
    if args.batch:
        samples = []
        with open(args.val_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append((obj.get("instruction", ""), obj.get("output", "")))

        print(f"\n验证集共 {len(samples)} 条，抽查 {min(args.num, len(samples))} 条\n")
        print("=" * 60)
        for i, (instr, gold) in enumerate(samples[:args.num], 1):
            prompt = instr + sep
            gen = generate_fixed(model, char2idx, idx2char, DEVICE, prompt,
                                 max_new=args.max_new, seq_len=args.seq_len)
            print(f"\n[{i}] 指令: {instr}")
            print(f"    期望: {gold}")
            print(f"    生成: {gen}")
            print("-" * 60)
        return

    # ── 交互模式：你输入问题，模型即时回答 ──
    print("=" * 52)
    print("  SFT 模型交互测试")
    print("  输入指令/问题回车 → 模型回答")
    print("  r = 重置上下文  q = 退出")
    print("=" * 52)

    while True:
        raw = input("\n你> ").strip()
        if not raw:
            continue
        if raw == "q":
            print("退出。")
            break
        if raw == "r":
            print("（已重置）")
            continue

        # 拼接 instruction + 分隔符，让模型续写 output
        prompt = raw + sep
        ans = generate_fixed(model, char2idx, idx2char, DEVICE, prompt,
                             max_new=args.max_new, seq_len=args.seq_len)
        print(f"模型> {ans}")


@torch.no_grad()
def generate_fixed(model, char2idx, idx2char, device, prompt, max_new=60, seq_len=64):
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return "（提示词里的字都不在字表里，无法生成）"
    ctx = torch.tensor([ids], device=device)
    out = []
    for _ in range(max_new):
        logits = model(ctx[:, -seq_len:])[0, -1]
        nxt = torch.argmax(logits).item()
        ch = idx2char[nxt]
        if ch == "\n":          # 遇到结束符就停
            break
        out.append(ch)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)
    return "".join(out)


if __name__ == "__main__":
    main()
