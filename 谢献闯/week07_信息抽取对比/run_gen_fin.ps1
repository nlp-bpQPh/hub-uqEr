# -*- coding: utf-8 -*-
<#
DuEE-Fin 生成式链路批量实验（本地 Qwen3-0.6B / 1.7B，零样本 / 少样本）
用法：  powershell -File w7_ext/run_gen_fin.ps1
日志：  outputs/runlogs/gen_<tag>.log
说明：  每条命令都独立落盘（pred/raw/report/metrics），失败不阻塞后续；
        指标走项目统一评测 common/evaluate_events.py，和判别式方法同一口径。
#>
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\tool\anaconda3\envs\py312\python.exe"
$models = "E:\大模型学习\week7_models"
New-Item -ItemType Directory -Force -Path outputs\runlogs | Out-Null

function Run-One([string]$tag, [string]$modelPath, [string]$mode, [string]$n, [string]$extra) {
    Write-Host "=== $tag ==="
    $cmdArgs = @("llm_zero_shot/llm_extract.py", "--backend", "local", "--mode", $mode,
              "--n", $n, "--tag", $tag, "--model_path", $modelPath,
              "--max_new_tokens", "320", "--gen_batch", "8")
    if ($extra) { $cmdArgs += $extra.Split(" ") }
    & $py @cmdArgs *> "outputs\runlogs\gen_$tag.log"
    Write-Host "$tag EXIT=$LASTEXITCODE"
}

# 0) 冒烟：2 条，确认本地生成链路 + 解析 + 评测跑通（数字无意义）
Run-One "w7_smoke_q06" "$models\Qwen3-0.6B" "zero" "2" "--max_length 1280"

# 1) 零样本：0.6B
Run-One "w7_qwen06_zero" "$models\Qwen3-0.6B" "zero" "200" "--max_length 1280"

# 2) 零样本：1.7B（换模型规模）；bs 减半防 OOM
Run-One "w7_qwen17_zero" "$models\Qwen3-1.7B" "zero" "200" "--max_length 1280 --gen_batch 4"

# 3) 少样本：0.6B（6 个示例，prompt 更长 → max_length 放宽到 2048，避免示例被截断）
Run-One "w7_qwen06_few" "$models\Qwen3-0.6B" "few" "200" "--max_length 2048"

Write-Host "GEN FIN DONE"
