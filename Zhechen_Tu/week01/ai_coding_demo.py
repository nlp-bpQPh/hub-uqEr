# 文件名：ai_coding_demo.py
import numpy as np
import torch


def main():
    # 固定随机种子，在 CPU 上训练。
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    # 用 numpy 构造训练数据，再转换为张量。
    x_np = np.array([0, 1, 2, 3, 4], dtype=np.float32).reshape(-1, 1)
    y_np = 2 * x_np + 1
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    # 线性模型包含一个权重和一个偏置。
    model = torch.nn.Linear(1, 1).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # 重复计算误差、反向传播并更新参数。
    model.train()
    for _ in range(2000):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    # 使用训练后的参数重新计算误差和预测。
    model.eval()
    with torch.no_grad():
        final_mse = loss_fn(model(x), y).item()
        prediction = model(
            torch.tensor([[5.0]], dtype=torch.float32, device=device)
        ).item()

    weight = model.weight.item()
    bias = model.bias.item()
    print(f"Learned weight: {weight:.6f}")
    print(f"Learned bias: {bias:.6f}")
    print(f"Final MSE: {final_mse:.10f}")
    print(f"Prediction for x=5: {prediction:.6f}")

    # 检查拟合误差和预测是否达到要求。
    assert final_mse < 0.001, f"MSE too large: {final_mse}"
    assert abs(prediction - 11.0) < 0.1, f"Prediction inaccurate: {prediction}"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()