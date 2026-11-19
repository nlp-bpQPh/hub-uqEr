# 备份下载脚本：hf-mirror 偶发 429，用 curl 断点续传 + 重试逐个文件拉取
$ErrorActionPreference = "Stop"
$root = "E:\大模型学习\作业\第六周作业-分类方法对比\models"
$repos = @("hfl/chinese-roberta-wwm-ext", "hfl/chinese-macbert-base")
$files = @("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
           "special_tokens_map.json", "added_tokens.json", "pytorch_model.bin")

foreach ($repo in $repos) {
    $name = $repo.Split("/")[-1]
    $dir = Join-Path $root $name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($f in $files) {
        $out = Join-Path $dir $f
        $url = "https://hf-mirror.com/$repo/resolve/main/$f"
        Write-Host "[get] $name/$f"
        curl.exe -L -C - --retry 30 --retry-delay 5 --retry-all-errors `
            --connect-timeout 30 -o $out $url
        if (Test-Path $out) {
            Write-Host ("   -> {0:N2} MB" -f ((Get-Item $out).Length / 1MB))
        }
    }
}
Write-Host "all done"
