param([string]$Config = (Join-Path $PSScriptRoot 'dpd_config.json'))
$ErrorActionPreference = 'Stop'
$dpdPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $dpdPython)) {
    throw 'Missing .venv. Create the environment and install requirements.txt as described in README_DPD.md.'
}
& $dpdPython (Join-Path $PSScriptRoot 'run_dpd.py') --config $Config
exit $LASTEXITCODE
