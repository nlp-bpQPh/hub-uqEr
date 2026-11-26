# 规则基线

课件 L0 思路的 XFUND 适配：关键词/格式正则识别 question 与 answer，页面上方区域作为 header，并用 OCR 坐标估计邻近键值区域。无需训练。

运行：

```powershell
python run.py
```

结果写入本目录 `outputs/metrics.json` 与 `outputs/predictions.jsonl`。这是启发式基线，不读取金标标签做推断。
