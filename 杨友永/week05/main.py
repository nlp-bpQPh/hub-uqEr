"""工程入口。

    python main.py train                 # 全参 SFT
    python main.py train --limit 200     # 快速验证链路
    python main.py chat                  # 交互问答
    python main.py demo                  # 用固定问题试答
"""
from __future__ import annotations

import argparse

from config import TRAIN
from schemas import ChatRequest

DEMO_QUESTIONS: list[str] = [
    "把这句话改成被动语态：我们明天开会。",
    "创建一个待办清单：买水果、回邮件、洗衣服。",
]


def run_train(limit: int | None, epochs: float, lr: float | None,
              device: str | None) -> None:
    from dataclasses import replace

    from train import train

    train(cfg=replace(TRAIN, epochs=epochs), limit=limit, lr=lr, device_str=device)


def run_chat() -> None:
    from chat import ChatEngine
    from device import describe

    engine = ChatEngine.load()
    print(f"模型 {engine.name} · 设备 {describe(engine.device)}")
    while True:
        question = input("\n问题> ").strip()
        if not question:
            break
        print(f"回答> {engine.ask(ChatRequest(question=question)).answer}")


def run_demo(device: str | None) -> None:
    from chat import ChatEngine
    from device import describe

    engine = ChatEngine.load(device=device)
    print(f"模型 {engine.name} · 设备 {describe(engine.device)}\n")
    for question in DEMO_QUESTIONS:
        response = engine.ask(ChatRequest(question=question))
        print(f"问题: {question}")
        print(f"回答: {response.answer}")
        print(f"      （{response.latency_ms:.0f} ms）\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="week5_llm 最小 SFT 工程")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="全参微调")
    p_train.add_argument("--limit", type=int, default=None)
    p_train.add_argument("--epochs", type=float, default=TRAIN.epochs)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--device", type=str, default=None)

    sub.add_parser("chat", help="交互问答")

    p_demo = sub.add_parser("demo", help="固定问题试答")
    p_demo.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args.limit, args.epochs, args.lr, args.device)
    elif args.command == "chat":
        run_chat()
    else:
        run_demo(args.device)


if __name__ == "__main__":
    main()
