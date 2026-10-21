param([Alias('SinGraficas')][switch]$NoPlots)
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'The .venv environment is missing. From the project directory, run: python -m venv .venv, followed by .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
}
$env:PYTHONUTF8 = '1'
$runArguments = @((Join-Path $PSScriptRoot 'run.py'))
if ($NoPlots) { $runArguments += '--no-plots' }
& $pythonExe @runArguments
exit $LASTEXITCODE
