# 排队训练：等 BERT-base 三个数据集跑完，再依次跑 RoBERTa / MacBERT
$ErrorActionPreference = "Continue"
$code = "E:\大模型学习\作业\第六周作业-分类方法对比\code"
$out = "E:\大模型学习\作业\第六周作业-分类方法对比\outputs"
$py = "D:\tool\anaconda3\envs\py312\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $code

Write-Host "[queue] waiting for bert-base/tnews ..."
while (-not (Test-Path (Join-Path $out "enc_bert-base_tnews.json"))) {
    Start-Sleep -Seconds 20
}
Write-Host "[queue] bert-base done, start roberta/macbert"

foreach ($m in @("roberta-wwm-ext", "macbert-base")) {
    foreach ($ds in @("chnsenticorp", "thucnews", "tnews")) {
        Write-Host "=== $m / $ds ==="
        & $py train_encoder.py --model $m --dataset $ds
    }
}
Write-Host "[queue] ALL DONE"
