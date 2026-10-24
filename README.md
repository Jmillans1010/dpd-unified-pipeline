# Unified digital predistortion pipeline

Read two independent datasets containing four I/Q CSV files and a Spectre `input` file, diagnose and align each dataset, train MP/EMP/GMP/Pruned Volterra/LUT models, select a model using validation data, and export PWL signals for Cadence. Final constellation plots contain **symbol instants only**, with I/Q axes in volts. Newly generated reports, plot labels, messages, and output descriptions are in English.

Python >=3.10. Install and run on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1
```

On Linux/macOS, use `.venv/bin/python run_pipeline.py`. Paths in a configuration JSON are resolved relative to that file, regardless of the current working directory. Open `dpd-unified-pipeline.code-workspace` to work in VS Code. The Windows launcher also accepts `DPD_PYTHON` pointing to a prepared Python environment. If `.venv` is unavailable, it can reuse the existing environment at `../procesador_iq_pa/.venv` and reports this choice when it runs. `ejecutar.ps1` remains available as a backward-compatible alias for `run.ps1`.

## Required datasets and configuration

```text
dataset/
  training/
    I_in_training.csv
    Q_in_training.csv
    I_out_training.csv
    Q_out_training.csv
    input
  validation/
    I_in_validation.csv
    Q_in_validation.csv
    I_out_validation.csv
    Q_out_validation.csv
    input
```

Each CSV contains two columns: time in seconds and voltage, with Cadence headers such as `/I_IN X,/I_IN Y` (`X` denotes time, not another signal channel). The explicit header `time_s,voltage_V` is also supported. Units are declared in `pipeline.json`; they are not inferred from signal amplitude. Irregular sampling is rejected, and signals are not silently resampled.

`input` is a required Spectre text file; it does not contain the waveform samples. The existing `InputSCSReader` extracts source and transient settings, preserves literals and include statements, and **does not execute** netlists or referenced files. The parser supports the subset documented by the original project, rather than the entire Spectre language. Multiple wsource/tran declarations or unresolved essential expressions require explicit disambiguation.

The confirmed source uses QAM16, a data rate **in bits/s**, and OFDM=No. The pipeline calculates Rs=Rb/log2(M) and measures Fs from the CSV time columns. For Rb=1G and Ts=250ps, this gives L=16 samples per symbol. It checks that Fs/Rs is an integer and that the CSV time grid agrees with strobeperiod. The source's `oversample` setting is not used as the CSV samples-per-symbol value. Automatic bit-rate conversion rejects OFDM.

The local copy of this project uses `training_data` and `Validation_1`. **Validation_0 is excluded.** Original files remain unchanged: each run keeps byte-for-byte snapshots of all five files and their hashes. Source snapshots retain their original contents and language. The user confirmed a shared configuration with different seeds; exact seed values remain unknown unless supplied in `seed_provenance` or found in the netlist parameters. Different seeds alone do not establish independence for PN9 sequences.

`dataset/` and `runs/` are local and excluded from Git. Circuit netlists, PDK details, third-party paths, and private results are not published. A new clone requires the five files for each dataset. The following demo works without private data.

## Reproducible demo

```powershell
.\.venv\Scripts\python.exe make_demo.py dataset/demo
.\.venv\Scripts\python.exe run_pipeline.py --config dataset/demo/demo.json
```

This generates two different synthetic sequences with minimal metadata. **They are neither PA measurements nor a Spectre simulation.** The destination directory must not already exist.

## Processing and model selection

1. Validate files, headers, finite values, declared units, and time grids.
2. Diagnose raw signals using waveforms, FFT amplitude spectra, peaks, RMS, and PAPR.
3. Align input and output using complex correlation, optionally including fractional delay. Trimming and DC removal are configurable and recorded. The four original modules and their `aligned_waveform.csv` + `metadata.json` interface are preserved.
4. Determine symbol phase and initial EVM using the measured input with fitted gain/phase; this is not an ideal-QAM comparison. Store complete waveforms and symbol samples separately.
5. Calculate gain G exclusively from valid training samples. Fit the inverse relationship `F(y/G) ≈ x`.
6. **Select each model family's structure using BIC on training data only.** The search receives only the training partition and never reads validation data. Least-squares fitting, basis construction, pruning, and LUT interpolation reuse the original trainers.
7. Evaluate the five selected models on **all usable validation samples**, with common support and guard intervals. Validation is not split, and no additional final test partition is created.
8. Select the winner by the lowest validation `nmse_waveform_db`. Break ties by coefficient count and then model name. This dataset is used for selection; it is not a subsequent independent test.
9. Plot constellations using the same symbol indices and shared phase, with symbol NMSE in plot titles. Also export full-waveform NMSE, relative RMS errors, and hardware estimates. Errors and extreme values at symbol instants are retained. Provide both a full-range view and a zoomed view with out-of-view counts.
10. Generate predistortion `F(x)`, apply the voltage limit, and export signals. Include a no-DPD reference using the same interpolation.

Symbol phase is determined from the original validation record before comparing reconstructions. It is not adjusted separately for each model. Symbol selection affects only symbol plots and metrics: training and inference with memory retain all valid waveform samples.

## Explore larger model structures

The default maximum structures are MP/EMP P15 M12, GMP P15 M12 Q4, Volterra P9 M6 D2, and LUT with up to 1024 nodes. M is the maximum delay, including delays 0..M; polynomial orders are odd. The basis limit is 4096 terms.

`search.plateau_extra_depth=2` explores up to two additional levels after a level without improvement; `beam_width=3` retains three valid paths per level. The best BIC found is always retained, so exploring larger structures does not require selecting a larger model. Set the extra depth to `0` to disable this extension. The search remains bounded and local; it does not guarantee a global optimum.

The budgets are 150 iterations, 180 candidates per model family, and an estimated working memory of 1024 MiB, calculated as six complex N×K matrices. This is a preventive estimate, not a hard RAM limit. Candidates rejected because of resources, rank, or conditioning are recorded. An invalid search boundary is reported as incomplete, rather than presented as a global minimum.

Changing maximum memory depths changes the common context and the number of scored samples. Check the manifest's indices and guard intervals before comparing different runs.

## Run directory layout

Each execution creates a separate `runs/run_<UTC>/` directory. Its five main folders separate diagnostics, fitted models, model comparisons, export files, and references for simulation results:

```text
runs/
  latest_run.json
  run_<UTC>/
    manifest.json
    diagnostics/
      training/
      validation/
    model_results/
      MP.json
      EMP.json
      GMP.json
      PrunedVolterra.json
      LUT.json
    model_comparisons/
      metrics.json
      report.md
      plots/
      ... CSV comparison files
    outputs/
      MP/
      EMP/
      GMP/
      PrunedVolterra/
      LUT/
      no_dpd/
    results/
      EMP/
        I_in.csv
        Q_in.csv
        reference_metadata.json
        README.md
      ... one folder per exported model and no_dpd
```

`diagnostics/` holds source snapshots, alignment results, initial EVM, statistics, and diagnostic figures for each dataset. `model_results/` holds model coefficients and optimization traces. `model_comparisons/` holds comparison metrics, CSV files, the report, and final plots. `outputs/` holds the exported PWL signals and their metadata. The selected processing stage determines which files are populated; `export.models=winner` exports only the winning model and the no-DPD reference.

`latest_run.json` is updated only after successful completion. An error leaves a manifest with status `failed`. Existing runs are preserved, including their original directory names and language; the new layout and English text apply to newly generated results.

## References for simulation results

Full runs prepare `results/<model>/I_in.csv` and `Q_in.csv`, including `results/no_dpd/`. These are the **desired input before predistortion**, copied from `I_desired_V` and `Q_desired_V` in the corresponding export's `samples.csv`. Each has two columns, `time_s,voltage_V`, with the exact exported time grid, trimming, and desired-signal scale. No PA amplification, phase shift, or predistortion is applied to these references.

Use the PWL files in `outputs/<model>/` to drive Cadence. Later, place measured I_OUT and Q_OUT in the matching `results/<model>/` folder as `I_out.csv` and `Q_out.csv`, retaining their original timestamps. Output files are not fabricated or pre-created. Reference metadata records provenance and hashes, and preparing references again refuses to overwrite an existing model result directory.

The result folder is a collection area, not a new automatic feedback input format. The existing `--feedback-dir` command still takes the five-file measured Cadence dataset described below. Compare outputs with the desired reference after alignment and compensation by the saved complex training gain; do not treat matching row numbers as proof of alignment.

## PWL export and the 3 V limit

With `export.overflow_policy=zero_pair`, after applying `voltage_scale`, any sample with abs(I)>3 V **or** abs(Q)>3 V is replaced with **I=Q=0**. Values exactly equal to ±3 V are retained. This is not clipping. Values before modification, exported values, the affected-sample mask, and affected indices are saved; nonfinite values are rejected. The `reject` policy instead blocks an export that exceeds the limit.

`outputs/<model>/I.txt` and `Q.txt` contain two ASCII columns: seconds and volts. Each model directory also contains `samples.csv` with the desired signal, original predistortion, exported signal, and symbol indices. `metadata.json` records hashes, statistics, time origin, scale, context, and affected samples. `export.models` accepts `all` or `winner`.

Supported boundary policies are `zero_pad` (N samples, with boundaries identified) and `trim` (valid context only). GMP can use future samples: lookahead is recorded, and waveform generation is offline. Interpolation can be `linear` or `zoh` with `0 < transition_s < Ts`. ZOH uses a finite ramp without repeated timestamps. The exported time starts at 0, and the original time origin is retained in metadata. The segment is not automatically repeated.

Connect the PWL signals to the Cadence modulator's I/Q inputs, replacing the original source. Do not connect them directly to the RF PA or place ideal voltage sources in parallel. The simulator's RF time step and baseband sampling interval are different quantities. See `framework_dpd/cadence_bridge/README.md` for the original adapter's details. Its standalone exporter rejects out-of-range signals; the unified pipeline applies the configured zero-pair policy before calling that exporter.

## Run a single stage or reload models

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --stage preprocess
.\.venv\Scripts\python.exe run_pipeline.py --models-dir runs/run_TIMESTAMP --stage compare
.\.venv\Scripts\python.exe run_pipeline.py --models-dir runs/run_TIMESTAMP
.\.venv\Scripts\python.exe run_pipeline.py --no-plots
```

`--models-dir` accepts a unified run directory, its `model_results/` directory, the legacy `models/` directory, or an original-framework run containing a manifest and five model JSON files. It reuses the original G and copies models without changing their coefficients. Without this option, `compare` performs training and comparison but does not export PWL signals. There is no hidden test evaluation.

## Cadence feedback

After simulation, export four CSV files containing input/output I/Q and their `input` file into a new directory, using the validation filenames:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --feedback-dir path/to/feedback --reference-run runs/run_TIMESTAMP --feedback-model MP
```

The reference is loaded from `outputs/MP/samples.csv`: it is the **desired** signal, not the predistorted signal measured at I_IN/Q_IN. The output is aligned using the fixed training gain G and the reference's symbol indices. The pipeline produces NMSE, EVM, AM/AM, AM/PM, and a symbol constellation. Settling-time trimming is explicitly controlled in the configuration. Compare against feedback using `--feedback-model no_dpd` under equivalent conditions. Legacy runs using `exports/` and the reference name `sin_dpd` remain readable. This command neither launches Spectre nor automatically retrains models.

Feedback runs use the same five-folder layout. `diagnostics/` contains feedback source snapshots and aligned signals; `model_comparisons/` contains feedback metrics and figures. Folders for stages that do not run may remain empty.

## Tests and scope

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover all five models end to end, clock inference, common symbol indices, training-only BIC, exploration past a worse candidate, voltage limits and PWL, model reloading without coefficient changes, unchanged source files, invalid inputs, and synthetic feedback with known gain. Original tests are retained within both source projects; some require their historical local datasets. See `VERIFICATION.md` for the validation record and any checks still pending for this revision.

PA validation through a Spectre simulation remains pending. ACLR is not reported without a measurement and defined frequency bands. Hardware figures are estimates, without synthesis or quantization. The full pipeline requires modern Python; only the original adapter's portable evaluator retains its Python 3.6-compatible design.
