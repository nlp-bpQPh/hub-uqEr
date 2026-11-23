# week05 作业：给语言模型做 SFT（指令微调）

用第 5 周 PPT 讲的三件事，把一个「只会接龙」的预训练模型，改成「会回答问题」的助手：

1. chat 模板：`<|user|>问题<|assistant|>回答<|end|>`
2. loss mask：提问部分的 label 全设 `-100`，只学回答段
3. 生成循环：一次一个 token，拼回输入继续算，遇到 `<|end|>` 停

基座模型用的是 `IDEA-CCNL/Wenzhong-GPT2-110M`（中文 GPT-2，约 110M 参数），
显卡是 RTX 4060 8G，全参微调一轮只要几秒，整个训练 1 分钟左右。

---

## 怎么跑（按顺序）

先确认用的是带 CUDA 的 python：

```powershell
D:\Miniconda3\envs\py312\python.exe -c "import torch; print(torch.cuda.is_available())"
```

然后在本文件夹里：

```powershell
# ① 生成指令数据（已经生成过，改数据时再跑）
python build_sft_data.py

# ② 训练（约 1 分钟，显存峰值约 5G）
python sft_train.py

# ③ 对话
python chat.py
```

第一次运行会自动下载基座模型（约 274MB）到 `model/wenzhong-gpt2-110m`。
下载走 `hf-mirror.com` 镜像（`common.py` 里用 requests 直连，不需要装 hf CLI）。

### 常用命令

```powershell
python sft_train.py --epochs 6 --lr 1e-4     # 改超参
python sft_train.py --quick                  # 只跑几十步，验证流程通不通
python sft_train.py --mix_ratio 0            # 不混通用语料，看灾难性遗忘有多严重

python chat.py --once "SFT是什么？"          # 只问一句
python chat.py --multi                       # 多轮对话（历史一起传）
python chat.py --base                        # 和没做过 SFT 的基座聊，看它只会接龙
```

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `build_sft_data.py` | 人工整理的 86 组课程问答 → 加两种问法 → 250 条 `data/sft_data.jsonl` |
| `common.py` | 公共件：路径、加载模型、chat 模板、loss mask、生成循环 |
| `sft_train.py` | 训练主程序：打印一条样本的 mask → 训练 → 训练完立刻对比「基座 vs SFT」 |
| `chat.py` | 训练完的模型对话脚本（单轮 / 多轮） |
| `data/sft_data.jsonl` | 指令数据，每行 `{"pair_id", "question", "answer"}` |
| `data/general_zh.txt` | 通用中文语料（从第 4 周语料截 100 万字），防灾难性遗忘用 |
| `model/wenzhong-gpt2-110m/` | 下载的基座模型 |
| `out/sft_model/` | 训练好的模型 |
| `out/sft_report.md` | 每次训练后自动生成的「SFT 前后对比」 |

---

## 训练时你会看到什么

```
一条 SFT 样本长这样（label = -100 的位置不参与 loss）
位置  token               label     说明
0    <|user|>            -100      提问段，忽略
...
17   <|assistant|>       -100      提问段，忽略
18   人                   21689     回答段，算 loss     ← 从这开始才算 loss
...
监督 token: 40/58 = 69%（PPT：预训练 100% 有信号，SFT 只有回答段有信号）
```

然后每轮的 loss：

```
 epoch  train_loss    val_loss          lr        用时
     1      1.6646      1.5272    4.81e-05        6s
   ...
    12      0.8264      2.5414    0.00e+00       53s
```

- `train_loss` 里混了 10% 通用语料（loss 天然很高），所以数值偏高，不代表 SFT 没学好
- `val_loss` 从第 1 轮就开始涨，这是 250 条数据必然的过拟合，正常现象，梳理里有解释

最后会自动跑一遍对比：

```
问：SFT是什么？
  基座  ：'SFT是一种特殊的技术，一种新的技术，它将像自然…'     ← 只会接龙
  SFT后 ：'SFT是用指令数据教模型学会回答问题。'                  ← 学会回答了
```

---

## 常见问题

**中文输出乱码**：终端编码问题，先执行 `$env:PYTHONIOENCODING="utf-8"`（PowerShell）
或 `set PYTHONIOENCODING=utf-8`（cmd）再运行。

**提示没有 CUDA**：说明用错 python 了，见上面第一步。

**显存不够**：把 `--batch_size` 改成 4，`--max_len` 改成 128。

**它答得不对**：这个作业的模型只见过 250 条数据，换一种问法就可能答错。
SFT 学的是「回答的形式」，不是「知识」——数据里没有的东西，它不会凭空知道。
