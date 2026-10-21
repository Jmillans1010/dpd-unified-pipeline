param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
$ErrorActionPreference = 'Stop'
$pythonPath = $env:DPD_PYTHON
if (-not $pythonPath) { $pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) {
    # Reuse the existing processor environment on the original workstation.
    $existingPython = Join-Path $PSScriptRoot '../procesador_iq_pa/.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $existingPython) {
        $pythonPath = (Resolve-Path -LiteralPath $existingPython).Path
        Write-Host "Usando el entorno existente: $pythonPath"
    } else { throw 'Crea .venv e instala requirements.txt, o define DPD_PYTHON con un entorno preparado.' }
}
& $pythonPath (Join-Path $PSScriptRoot 'run_pipeline.py') @PipelineArgs
exit $LASTEXITCODE
