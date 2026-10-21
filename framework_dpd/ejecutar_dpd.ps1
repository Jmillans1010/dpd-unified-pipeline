param([string]$Config = (Join-Path $PSScriptRoot 'dpd_config.json'))
$ErrorActionPreference = 'Stop'
$dpdPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $dpdPython)) {
    throw 'Falta .venv. Crea el entorno e instala requirements.txt según README_DPD.md.'
}
& $dpdPython (Join-Path $PSScriptRoot 'run_dpd.py') --config $Config
exit $LASTEXITCODE
