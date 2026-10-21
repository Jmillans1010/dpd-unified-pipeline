param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
$ErrorActionPreference = 'Stop'
$pythonPath = $env:DPD_PYTHON
if (-not $pythonPath) { $pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) {
    # Reuse the existing processor environment on the original workstation.
    $existingPython = Join-Path $PSScriptRoot '../procesador_iq_pa/.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $existingPython) {
        $pythonPath = (Resolve-Path -LiteralPath $existingPython).Path
        Write-Host "Using the existing environment: $pythonPath"
    } else { throw 'Create .venv and install requirements.txt, or set DPD_PYTHON to a prepared environment.' }
}
& $pythonPath (Join-Path $PSScriptRoot 'run_pipeline.py') @PipelineArgs
exit $LASTEXITCODE
