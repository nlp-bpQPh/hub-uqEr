# -*- coding: utf-8 -*-
<#
「换 API」实验：起本地 OpenAI 兼容端点（w7_ext/serve_openai.py），
用项目原有的 --backend api 代码路径跑 zero/few-shot，和本地直推同模型对比。

产物：
  outputs/llm/report_w7_api_qwen06_{zero,few}.json   客户端侧（延迟/token/JSON 合法率）
  outputs/runlogs/api_server_stats.jsonl             服务端侧（每请求 http 总耗时 vs 纯生成耗时）
用法： powershell -File w7_ext/run_api_fin.ps1
#>
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\tool\anaconda3\envs\py312\python.exe"
$models = "E:\大模型学习\week7_models"
$port = 8011
New-Item -ItemType Directory -Force -Path outputs\runlogs | Out-Null

Write-Host "=== start server ==="
$stats = "outputs\runlogs\api_server_stats.jsonl"
$srv = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -ArgumentList @(
    "w7_ext/serve_openai.py", "--model_path", "$models\Qwen3-0.6B",
    "--model_name", "qwen3-0.6b-local", "--port", "$port",
    "--dtype", "bf16", "--stats", $stats
) -RedirectStandardOutput "outputs\runlogs\api_server.log" -RedirectStandardError "outputs\runlogs\api_server.err.log"

# 等 /health（最多 180s：0.6B 加载 + 首次编译）
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch {}
}
if (-not $ok) {
    Write-Host "SERVER NOT READY —— 查看 outputs\runlogs\api_server.err.log"
    Get-Content "outputs\runlogs\api_server.err.log" -Tail 20
    exit 1
}
Write-Host "server ready"

$env:LLM_BASE_URL = "http://127.0.0.1:$port/v1"
$env:LLM_API_KEY = "local-dummy-key"
$env:LLM_MODEL = "qwen3-0.6b-local"

function Run-Api([string]$tag, [string]$mode, [string]$n) {
    Write-Host "=== $tag ==="
    & $py llm_zero_shot\llm_extract.py --backend api --mode $mode --n $n --tag $tag `
        --model "qwen3-0.6b-local" --max_tokens 320 *> "outputs\runlogs\api_$tag.log"
    Write-Host "$tag EXIT=$LASTEXITCODE"
}

# 冒烟 3 条：确认协议、解析、评测链路
Run-Api "w7_api_smoke" "zero" "3"
Run-Api "w7_api_qwen06_zero" "zero" "200"
Run-Api "w7_api_qwen06_few" "few" "200"

Write-Host "=== stop server ==="
Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
Write-Host "API FIN DONE"
