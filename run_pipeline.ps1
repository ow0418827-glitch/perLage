param(
    [ValidateSet('select','generate','evaluate','verify','report','all','doctor')][string]$Stage = 'all',
    [ValidateSet('lucid','PerLAGE','PerLAGE_no_pixel')][string]$Method = 'PerLAGE',
    [string]$Config = 'src/experiments/config/pipeline.example.json',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) { throw '请先运行 tools/setup_runtime.ps1 准备运行环境' }
& $runtimePython -X utf8 -B -m src.experiments.pipeline $Stage --method $Method --config $Config --device $Device
exit $LASTEXITCODE
