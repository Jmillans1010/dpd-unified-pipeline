# PA I/Q processor

This reusable processor reads four Cadence CSV files and a Spectre netlist, aligns input/output using complex cross-correlation, and calculates EVM relative to the measured input with fitted constant gain and phase. Full waveform samples remain available for MP, GMP, and Volterra modeling.

The unified pipeline calls this package automatically. Its root configuration and run layout take precedence over the standalone examples below. Original measurement files are never modified, and private datasets are not included in the public repository.

## Structure

```text
procesador_iq_pa/
├── run.py                       # Standalone entry point
├── ejecutar.ps1                 # Windows launcher (legacy filename)
├── config.json                  # Relative paths and parameters
├── requirements.txt
├── iq_processor/
│   ├── models.py                # Configuration, contracts, and results
│   ├── readers.py               # CSV, netlist, and metadata resolution
│   ├── processing.py            # Validation, correlation, selection, and EVM
│   └── outputs.py               # CSV, JSON, and plots
├── tests/test_processor.py      # Synthetic tests and an optional real-data test
└── data/                        # Local only; not included in the repository
    ├── input/                   # The five original input files
    └── output/                  # One directory per standalone run
```

The standalone configuration expects `I_in_training.csv`, `Q_in_training.csv`, `I_out_training.csv`, `Q_out_training.csv`, and the extensionless `input` netlist. The netlist does not need an `.scs` extension.

## Standalone execution

For a portable installation with Python 3.10 or later, run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\ejecutar.ps1
.\ejecutar.ps1 -NoPlots
```

The launcher uses `.venv/Scripts/python.exe` exclusively. The former `-SinGraficas` switch remains an alias for compatibility. On macOS/Linux, the virtual-environment interpreter is `.venv/bin/python`.

With the dependencies installed in the selected interpreter, these commands also work:

```text
python run.py
python run.py --no-plots
python run.py --config another_config.json
```

`run.py` can be invoked from any directory. The default configuration is located beside the script; data paths are resolved **relative to the configuration file**, not the terminal's current directory. Local copies do not require a network connection.

Select the intended interpreter in your editor. Binary extensions must match its Python version. An error referring to `_multiarray_umath.cp312-win_amd64.pyd` under Python 3.14 indicates mixed binary versions, rather than a numerical algorithm error. Do not add a historical `.runtime_deps` directory to `PYTHONPATH`. Recreate virtual environments when moving the project or changing Python versions; do not copy compiled packages between interpreters.

## Workflow and methods

1. **CSVReader** reads two comma-separated columns with a header: time in seconds and the signal value. It constructs `x = I_in + j Q_in` and `y = I_out + j Q_out`. I/Q channels with incompatible times are rejected.
2. **InputSCSReader** extracts values by instance and analysis, joins continued lines, and respects comments and quoted strings. It preserves literals and include statements without executing them. It does not interpret the complete Spectre language or evaluate arbitrary expressions. Multiple `wsource` sources or `tran` analyses cause an explicit ambiguity error.
3. **MetadataResolver** combines the netlist, previous JSON associated with the same CSV files, and manual configuration while detecting conflicts. `previous_metadata` requires matching SHA-256 hashes for all four CSV files. Previous EVM values and delays are not reused.
4. **SignalValidator** checks finite values, time order, grid uniformity, and channel compatibility. It records observed sampling and tolerances. It never silently resamples.
5. **SignalPreprocessor** applies only enabled initial cropping and DC removal. Both are disabled by default.
6. **SampleAligner** calculates zero-padded linear FFT correlation, equivalent to `R[l] = sum(conj(x[n])*y[n+l])`, and maximizes `abs(R)` within the configured range. Positive `D` pairs `x[n]` with `y[n+D]`. No circular shift is used.
7. **FractionalDelayRefiner**, disabled by default, fits a parabola around the correlation-magnitude peak and applies Hann-windowed sinc interpolation. Samples without full interpolation support are discarded. This is a local approximation, not an EVM-based delay search.
8. **SymbolTimingSelector** evaluates `p=0,...,L-1` with the same number of complete symbols. Each phase uses `X[k]=x_aligned[p+kL]` and `Y[k]=y_aligned[p+kL]`. It selects minimum EVM; ties within tolerance select the lowest phase. `symbol_phase` fixes a manual phase while retaining diagnostics for every phase.
9. **EVMCalculator** estimates `alpha = sum(conj(X)*Y)/sum(abs(X)**2)` and calculates `100*sqrt(sum(abs(Y-alpha*X)**2)/sum(abs(alpha*X)**2))`. The reference is the fitted measured input, not an ideal QAM constellation. Zero-energy or negligible-energy references are rejected.
10. **StatisticsCalculator**, **MetadataProcessor**, **CSVWriter**, **JSONWriter**, and **Grapher** generate statistics, provenance, and files. Full waveforms retain their gain and phase; compensation is not applied to the complete CSV.

The data objects contain typed NumPy arrays rather than DataFrames, so pandas is not required. Tabular outputs use CSV. All generated plots, summaries, messages, and diagnostic descriptions are in English; preserved source text remains unchanged.

## Standalone parameters

These values describe this package's historical standalone `config.json`. The unified pipeline has its own configuration, including the verified source bit-rate interpretation.

| Option | Standalone value | Meaning |
|---|---:|---|
| `samples_per_symbol` | 16 | Explicit hypothesis for this experiment; `null` resolves it from other sources. |
| `samples_per_symbol_verified` | false | Warns that the physical interpretation of L has not been verified. |
| `source_data_rate_is_bit_rate` | false | Does not automatically interpret `MSource_DataRate` as bits/s. |
| `max_delay_samples` | 128 | Configurable search from −128 to +128 samples. |
| `minimum_overlap_fraction` | 0.8 | Requires at least 80% of the shorter record's length. |
| `fractional_refinement` | false | Uses integer alignment by default. |
| `additional_initial_crop_s` | 0 | Does not remove another 8 ns from CSV files already starting at 8 ns. |
| `remove_dc` | false | Does not change signal means automatically. |
| `time_rtol` | 1e-5 | Tolerance relative to the sample interval, not absolute time. |
| `time_atol_s` | 1e-18 | Additional absolute tolerance in seconds. |
| `minimum_symbols` | 16 | Minimum required for phase selection. |
| `symbol_phase` | null | Automatic search; an integer selects a manual phase. |

The original verified CSV files have a median interval of 250 ps. Small step variations up to approximately 1.81 fs occur on all four channels. A tolerance of about 2.501 fs accepts these variations while preserving original times and all rows. The exported physical delay is the median time difference between aligned pairs; the nominal delay inferred from sampling is also retained.

`oversample=4` is retained as a source parameter and **is not** treated as the CSV samples-per-symbol ratio. Sixteen samples per symbol are consistent with 1 Gbit/s, 16-QAM, and 250 ps when DataRate means bit rate. A manual `symbol_rate_hz` may be provided in `manual_metadata.signal_config`; conflicts with L are rejected. Declaring the relationship verified requires an explicit configuration setting.

## Standalone output files

Each standalone execution creates `data/output/run_<UTC timestamp>/`. `latest_run.json` points to the latest completed export; earlier runs remain intact. The unified entry point manages its separate run structure described in the repository root README.

| File | Contents |
|---|---|
| `aligned_waveform.csv` | All aligned pairs, source indices, times, and I/Q values. |
| `selected_symbols.csv` | Symbols, reference `alpha*X`, and complex error. |
| `correlation_diagnostics.csv` | Correlation, normalized diagnostics, and overlap for each delay. |
| `symbol_phase_candidates.csv` | EVM, alpha, and status for each phase. |
| `metadata.json` | The four metadata sections described below. |
| `summary.txt` | Human-readable results and warnings. |
| `01_cross_correlation.png` | Delay search. |
| `02_aligned_waveforms.png` | First 512 aligned samples without amplitude normalization. |
| `03_evm_vs_phase.png` | EVM for every phase. |
| `04_constellations.png` | Measured input and output versus the fitted linear reference; symbol samples only. |
| `05_symbol_error.png` | Error magnitude per symbol. |

The JSON sections are `simulation_config`, `signal_config`, `processing`, and `statistics`. Source paths are relative to the configuration. Metadata also records SHA-256 hashes, effective options, versions, field provenance, candidate scores, and stage status. Paths within netlist literals document the simulation; the processor never opens them.

The standalone processor can align data without a netlist when configuration provides L. If L is unavailable, it exports the waveform and correlation and marks EVM as skipped. The unified pipeline separately requires an `input` netlist for each dataset. Essential data errors return exit code 1. An interrupted export contains `FAILED.txt` and does not replace `latest_run.json`.

## Interpreting results

Phase and alpha are fitted and evaluated on the same record: this is **fitted EVM, not independent validation**. The processor does not select phase by distance to ideal QAM or apply the old script's 70/30 split. It uses the measured input and input/output error. Amplitude-dependent distortion and memory remain in the residual.

The result characterizes the entire path between the exported I/Q nodes. If the path contains filters or modulators, additional characterization is needed to attribute the result exclusively to the PA.

The first historical verified run, without fractional refinement, obtained:

- 8,769 samples per channel and 8,766 aligned pairs.
- A delay of 3 samples, approximately 0.75 ns.
- Phase 6 relative to the aligned start, with 547 symbols.
- Approximately 22.29665% RMS EVM, using the then-unverified L=16.

These historical values depend on that configuration. Cropping, phase, reference, DC removal, and refinement can change the metrics.

## Tests

With the requirements installed:

```text
python -m unittest discover -s tests -v
```

Without activating the environment, use `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Do not add the old `.runtime_deps` directory to `PYTHONPATH`.

Tests cover FFT correlation against direct summation, positive and negative delays, rotation, different time origins, analytical EVM, zero energy, retained nonlinearity, invalid times, known symbol phase, fractional refinement, netlist parsing, inconsistent configuration, and stale metadata. The real-data integration test additionally reconstructs EVM from exported CSV files; it requires the local dataset configured in `config.json`, which is not included in the public repository.
