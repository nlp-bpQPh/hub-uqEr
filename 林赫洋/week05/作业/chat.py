"""和 SFT 之后的模型聊天（第 5 周 PPT 第 4 页：生成循环 + 第 37 页：多轮对话）。

用法：
    python chat.py                     # 进入对话（默认单轮）
    python chat.py --multi             # 多轮：把历史一起传给模型
    python chat.py --once "什么是SFT"  # 只问一句就退出
    python chat.py --base              # 和「只预训练过的基座」聊，看它只会接龙
    python chat.py --temperature 0.2   # 更保守；越高越发散
"""

import argparse
import os
import sys

import common


def parse_args():
    p = argparse.ArgumentParser(description="SFT 模型对话")
    p.add_argument("--once", type=str, default=None, help="只回答一个问题")
    p.add_argument("--base", action="store_true", help="加载基座模型（只预训练，不会答）")
    p.add_argument("--multi", action="store_true", help="多轮对话，历史一起传进去")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    device = common.pick_device(args.device)

    if args.base:
        common.log("[加载] 基座模型（只有预训练，是个续写器）")
        tok, model = common.load_base_model(device)
    else:
        common.log("[加载] SFT 模型")
        tok, model = common.load_sft_model(device)

    def answer(question, history):
        return common.generate(
            model, tok, question, device,
            history=history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    if args.once:
        common.log(f"你 > {args.once}")
        common.log(f"助手 > {answer(args.once, None)}")
        return

    print()
    print("=" * 60)
    print("输入问题回车；输入 exit 退出，输入 clear 清空历史")
    print(f"当前：{'基座（只会接龙）' if args.base else 'SFT 模型'} · "
          f"{'多轮' if args.multi else '单轮'} · temperature={args.temperature}")
    print("=" * 60)

    history = []
    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            break
        if question.lower() in ("clear", "清空"):
            history.clear()
            print("（历史已清空）")
            continue

        reply = answer(question, history if args.multi else None)
        print(f"助手 > {reply}")
        if args.multi:
            # 多轮的关键：把这一问一答追加进历史，下次请求一起传（模型本身没有记忆）
            history.append((question, reply))


if __name__ == "__main__":
    main()
