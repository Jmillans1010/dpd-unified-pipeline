# Verification record

## English output and directory-layout revision

Verified on 2026-10-21 using the existing Python environment:

- All 10 unified pipeline tests passed, including the four-category layout, preprocessing-only runs, current and legacy model-directory loading, legacy feedback exports, and the old baseline-name alias.
- All 59 selected original framework/processor tests passed. Two historical integrations were excluded because their private fixture directories are not included in this repository.
- The complete small-model workflow ran again on the local training and validation datasets, producing 15 figures, English summaries and reports, and six PWL export directories in the new layout.
- For every model, coefficients, waveform NMSE, symbol NMSE, and the bytes of both PWL files were identical to the previous local-data run. This revision changes presentation and artifact organization rather than the numerical modeling behavior.
- The final symbol constellation was visually inspected. Source text for plots, summaries, messages, comments, and documentation was checked for remaining Spanish prose. Original source snapshots and legacy model files remain unchanged when reused.

The layout for new runs is `diagnostics/`, `model_results/`, `model_comparisons/`, and `outputs/`, with `manifest.json` at the run root. Existing runs and original source snapshots are preserved. The English launcher is `run.ps1`; `ejecutar.ps1` remains a backward-compatible alias.

## Previous version: 2026-10-21

Environment: Python 3.14, NumPy 2.5.3, and Matplotlib 3.11.2 on Windows.

- Eight automated tests passed, covering the synthetic end-to-end flow with five model families, training-only BIC, exploration past a worse candidate, common symbol indices, the 3 V limit and PWL export, invalid inputs, model reloading without coefficient changes, and synthetic feedback evaluated against the desired signal.
- The full pipeline ran on the local `training_data` and `Validation_1` datasets: reading, diagnostics, alignment, training, comparison, figures, and six PWL exports comprising five models and a no-DPD reference.
- Both datasets produced 8766 aligned samples with an integer delay of 3 samples. The source parameters and CSV time grid confirmed 16 samples per symbol.
- The local-data run used small search limits: MP/EMP P5 M2, GMP P3 M1 Q1, Pruned Volterra P3 M2 D1, and LUT with 8 to 16 nodes. It compared 8759 waveform samples and 547 common symbols. The final figures from that run were inspected and contained symbol instants only.
- The expanded limits in the default JSON were checked to produce a valid common context of 16 past samples and 4 future samples. **The full expanded search was not run as part of this verification.**
- Spectre was not executed, and no PA improvement was measured using the generated PWL signals. Feedback processing was verified using a synthetic case with known gain and delay.

Local data and run results are excluded from Git. Tests use reproducible synthetic data. On the original machine, the launcher reuses the processor's installed Python environment. Creating an additional environment was interrupted by insufficient disk space, and only that incomplete installation was removed.
