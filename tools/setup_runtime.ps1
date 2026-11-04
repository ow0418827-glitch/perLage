param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& $Python -m venv --system-site-packages .venv
if ($LASTEXITCODE -ne 0) { throw '创建运行环境失败' }
& '.\.venv\Scripts\python.exe' -m pip install -r src/dependencies/requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
if ($LASTEXITCODE -ne 0) {
    Write-Output '中国大陆镜像未提供所需版本，尝试官方软件源。'
    & '.\.venv\Scripts\python.exe' -m pip install -r src/dependencies/requirements.txt --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw '安装依赖失败，请检查网络或使用本机已有的兼容运行环境' }
}
