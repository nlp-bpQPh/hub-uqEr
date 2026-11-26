# -*- coding: utf-8 -*-
<#
LoRA SFT 实验（生成式微调）：
  1) Qwen3-0.6B：train 1 epoch / 3500 条 → 评测 n=200
  2) Qwen3-1.7B：同协议（bs 2 / grad_accum 8）→ 评测 n=200
复用项目 llm_lora/ 的脚本；checkpoint 分目录，避免互相覆盖。
用法： powershell -File w7_ext/run_sft_fin.ps1 [06|17|all]
#>
param([string]$which = "all")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\tool\anaconda3\envs\py312\python.exe"
$models = "E:\大模型学习\week7_models"
New-Item -ItemType Directory -Force -Path outputs\runlogs | Out-Null

function Run-Sft([string]$size, [string]$modelDir, [string]$ckptDir, [string]$bs, [string]$ga) {
    Write-Host "=== train $size ==="
    & $py llm_lora\train_sft.py --model_path $modelDir --output_dir $ckptDir `
        --epochs 1 --num_train 3500 --batch_size $bs --grad_accum $ga `
        --max_length 1280 --grad_ckpt --dtype bf16 --run_name "w7_qwen${size}_sft" `
        --log_every 50 *> "outputs\runlogs\sft${size}_train.log"
    Write-Host "train $size EXIT=$LASTEXITCODE"
    Write-Host "=== eval $size ==="
    & $py llm_lora\evaluate_sft.py --model_path $modelDir --ckpt_dir "$ckptDir\sft_adapter" `
        --n 200 --tag "w7_qwen${size}_sft" --max_new_tokens 320 --gen_batch 8 `
        *> "outputs\runlogs\sft${size}_eval.log"
    Write-Host "eval $size EXIT=$LASTEXITCODE"
}

if ($which -eq "06" -or $which -eq "all") {
    Run-Sft "06" "$models\Qwen3-0.6B" "outputs\llm\ckpt_q06" "3" "5"
}
if ($which -eq "17" -or $which -eq "all") {
    Run-Sft "17" "$models\Qwen3-1.7B" "outputs\llm\ckpt_q17" "2" "8"
}
Write-Host "SFT FIN DONE"
