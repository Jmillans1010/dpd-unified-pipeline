param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
# Compatibility entry point; new instructions use run.ps1.
& (Join-Path $PSScriptRoot 'run.ps1') @PipelineArgs
exit $LASTEXITCODE
