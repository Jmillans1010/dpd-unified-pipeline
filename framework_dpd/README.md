# Standalone DPD framework

This directory preserves the original inverse-model identification framework. For the integrated two-dataset workflow and the four-category run layout, use the repository-root `run_pipeline.py` or `run.ps1` and read the [unified pipeline guide](../README.md).

The standalone entry point remains available from this directory:

```powershell
.\ejecutar_dpd.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests_dpd -v
```

It requires its own prepared Python environment and aligned input files under the configured paths. Historical local data are not included in Git. Its standalone train/validation/test policy is separate from the unified pipeline's training-only BIC and full-validation comparison.

[Complete standalone framework documentation](README_DPD.md).
