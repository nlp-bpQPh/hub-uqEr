# -*- coding: utf-8 -*-
<#
换数据集：DuEE 1.0（65 类 / 216 槽位）上的生成式零样本 + LoRA SFT
- schema 换成 data/duee1/event_schema.json
- 输出目录 outputs/llm_duee1（不覆盖 DuEE-Fin 的固定 200 条抽样锚点）
- prompt 含 65 类 schema（约 1545 token），所以 max_length 提到 2560
用法： powershell -File w7_ext/run_gen_duee1.ps1 [zero|sft|all]
#>
param([string]$which = "all")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\tool\anaconda3\envs\py312\python.exe"
$models = "E:\大模型学习\week7_models"
$data = "data\duee1"
$schema = "data\duee1\event_schema.json"
New-Item -ItemType Directory -Force -Path outputs\runlogs | Out-Null

if ($which -eq "zero" -or $which -eq "all") {
    Write-Host "=== duee1 zero 0.6B ==="
    & $py llm_zero_shot\llm_extract.py --backend local --mode zero --n 200 `
        --data "$data\dev_extract.jsonl" --schema $schema --llm_dir outputs\llm_duee1 `
        --tag w7_qwen06_zero_duee1 --model_path "$models\Qwen3-0.6B" `
        --max_new_tokens 320 --max_length 2560 --gen_batch 4 `
        *> "outputs\runlogs\gen_w7_qwen06_zero_duee1.log"
    Write-Host "duee1 zero EXIT=$LASTEXITCODE"
}

if ($which -eq "sft" -or $which -eq "all") {
    Write-Host "=== duee1 sft 0.6B ==="
    & $py llm_lora\train_sft.py --model_path "$models\Qwen3-0.6B" `
        --data_dir $data --schema $schema --output_dir outputs\llm\ckpt_q06_duee1 `
        --epochs 1 --num_train 3500 --batch_size 3 --grad_accum 5 `
        --max_length 1280 --grad_ckpt --dtype bf16 --run_name w7_qwen06_sft_duee1 `
        --log_every 50 *> "outputs\runlogs\sft06_duee1_train.log"
    Write-Host "duee1 sft train EXIT=$LASTEXITCODE"
    & $py llm_lora\evaluate_sft.py --model_path "$models\Qwen3-0.6B" `
        --ckpt_dir outputs\llm\ckpt_q06_duee1\sft_adapter --data "$data\dev_extract.jsonl" `
        --schema $schema --llm_dir outputs\llm_duee1 --n 200 --tag w7_qwen06_sft_duee1 `
        --max_new_tokens 320 --max_length 2560 --gen_batch 4 `
        *> "outputs\runlogs\sft06_duee1_eval.log"
    Write-Host "duee1 sft eval EXIT=$LASTEXITCODE"
}
Write-Host "DUEE1 GEN DONE"
