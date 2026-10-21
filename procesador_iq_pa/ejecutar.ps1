param([switch]$SinGraficas)
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Falta el entorno .venv. Desde la carpeta del proyecto ejecuta: python -m venv .venv; después .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
}
$env:PYTHONUTF8 = '1'
$runArguments = @((Join-Path $PSScriptRoot 'run.py'))
if ($SinGraficas) { $runArguments += '--no-plots' }
& $pythonExe @runArguments
exit $LASTEXITCODE
