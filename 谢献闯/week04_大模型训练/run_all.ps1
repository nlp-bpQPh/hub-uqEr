# 一键跑完：数据准备 -> 训练 -> 评估 -> 演示
# 用法：powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"

# 优先用 py312（本机已装 torch/transformers），否则退回 PATH 里的 python
$py = "D:\tool\anaconda3\envs\py312\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

Write-Host "== 1/4 准备数据 ==" -ForegroundColor Cyan
& $py "src\prepare_data.py"

Write-Host "`n== 2/4 微调 BERT ==" -ForegroundColor Cyan
& $py "src\train.py"

Write-Host "`n== 3/4 测试集评估 ==" -ForegroundColor Cyan
& $py "src\evaluate.py"

Write-Host "`n== 4/4 演示 ==" -ForegroundColor Cyan
& $py "src\predict.py" --demo

Write-Host "`n全部完成。结果在 outputs\ 目录。" -ForegroundColor Green
