# -*- coding: utf-8 -*-
<#
生成式实验第二段（串行排队，避免两进程抢同一张卡）：
  1) LoRA SFT 0.6B（训练 + 评测）        —— L4
  2) 自建 OpenAI 兼容 API（zero / few）   —— 换调用方式
  3) DuEE 1.0 零样本                     —— 换数据集
  4) DuEE 1.0 LoRA SFT（训练 + 评测）     —— 换数据集 + 微调
用法： powershell -File w7_ext/run_after_gen.ps1
#>
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ps = "powershell.exe"
New-Item -ItemType Directory -Force -Path outputs\runlogs | Out-Null

function Step([string]$name, [scriptblock]$block) {
    Write-Host "########## $name ##########"
    $t0 = Get-Date
    & $block
    $dt = (Get-Date) - $t0
    Write-Host "########## $name done in $([math]::Round($dt.TotalMinutes,1)) min ##########"
}

Step "SFT 0.6B (DuEE-Fin)" { & $ps -NoProfile -ExecutionPolicy Bypass -File w7_ext\run_sft_fin.ps1 06 }
Step "API (self-hosted OpenAI-compatible)" { & $ps -NoProfile -ExecutionPolicy Bypass -File w7_ext\run_api_fin.ps1 }
Step "DuEE1 zero-shot" { & $ps -NoProfile -ExecutionPolicy Bypass -File w7_ext\run_gen_duee1.ps1 zero }
Step "DuEE1 SFT" { & $ps -NoProfile -ExecutionPolicy Bypass -File w7_ext\run_gen_duee1.ps1 sft }
Write-Host "AFTER GEN ALL DONE"
