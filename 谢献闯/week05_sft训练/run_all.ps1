# 一键跑完：数据准备 -> SFT 训练 -> 评估 -> 对话示例
# 用法：powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"

$py = "D:\tool\anaconda3\envs\py312\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

Write-Host "== 1/3 准备数据（ChatML + Loss Mask）==" -ForegroundColor Cyan
& $py "src\prepare_data.py"

Write-Host "`n== 2/3 LoRA SFT 训练 ==" -ForegroundColor Cyan
& $py "src\train_sft.py"

Write-Host "`n== 3/3 评估（SFT 前后对比）==" -ForegroundColor Cyan
& $py "src\evaluate.py" --num_samples 40

Write-Host "`n全部完成。结果在 outputs\ 目录，对话用 python src\chat.py" -ForegroundColor Green
