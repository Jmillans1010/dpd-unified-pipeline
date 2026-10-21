# DPD inverse-model identification framework

This document describes the retained **standalone** framework. The integrated entry point at the repository root uses a different run layout and two datasets without a final test split; see the [unified pipeline guide](../README.md). Running `run_dpd.py` uses the standalone policies below.

The framework implements MP, EMP, GMP, Pruned Volterra, and memoryless LUT models through `TrainingPipeline`, `CSVReader`, `DataProcessor`, `Trainer`, `SearchPolicy`, `HyperparameterOptimizer`, `HardwareComplexityEstimator`, and `JSONWriter`. Data and result contracts are dataclasses in `dpd/contracts.py`.

## Running the standalone entry point

From `framework_dpd`:

```powershell
.\ejecutar_dpd.ps1
# Equivalent command with an already prepared environment:
.\.venv\Scripts\python.exe run_dpd.py --config dpd_config.json
```

The launcher also works from other directories. Input and output paths are resolved relative to the configuration file. The standalone framework requires Python >=3.10 and NumPy. To prepare its own environment, run `python -m venv .venv` and `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

The processor in `../procesador_iq_pa/` retains its entry point, modules, and tests. The interface between the two components is `aligned_waveform.csv` plus `metadata.json`. The original standalone workflow used snapshots under `data/input/`, with provenance in `snapshot_manifest.json`; these historical datasets are not included in Git. Configure input paths explicitly for another experiment. The unified entry point now transfers these artifacts automatically.

## Standalone data partitions

The historical `validation_holdout` configuration uses:

- `data/input/training/aligned_waveform.csv`: training, apart from context and guards.
- The first chronological half of `data/input/Validation/aligned_waveform.csv`: diagnostic validation in BIC mode, or hyperparameter selection in the legacy reference-fraction mode.
- The second half of that CSV: reserved test data.
- The old `data/output/test` directory is excluded because its waveform duplicated training; exclusion does not delete it.

`validation_fraction=0.5` specifies the split. Explicit `guard_samples` separate the partitions and trim their boundaries. A null value derives the guard from the maximum allowed context. With the supplied expanded limits, the common context is 16 past and 4 future samples, and the automatic guard is 16. Samples without context are excluded. There is no zero padding, circular shifting, or mixing of states across partitions.

Records come from separate simulations whose timestamps restart. Ordering is preserved within each record without assuming a global chronology. PN9 periodicity can correlate records; splitting them does not establish statistical independence.

`CSVReader` reads the full waveform, not `selected_symbols.csv`, and preserves its existing alignment. `DataProcessor.align_signals()` supports integer correlation alignment for new records; it is not rerun on these already aligned inputs. `split_chronologically()` is available for a single record.

Gain uses only retained training samples:

```text
G = vdot(x_train, y_train) / vdot(x_train, x_train)
z = y / G
model: z -> x_hat; target: x
```

The processor validates time ordering and uniformity, finite values, compatible sampling, modulation, and carrier metadata. Original metadata and hashes are stored in `manifest.json`; paths in metadata are not executed. Matching metadata does not prove that two simulations used exactly the same PA.

For three independently supplied partitions, use `split_mode: "provided_partitions"` and add `inputs.test`. Entirely duplicated partitions are rejected, but this check does not detect every partial overlap. The explicit `duplicate_policy: "split_training"` option can reserve the last 25% of training when test duplicates it; this is not the supplied split policy.

## Mathematics and conventions

`M` is the maximum delay, so delay indices cover `0..M`. Only odd polynomial orders are included through `P`. Every model stores its exact ordered basis alongside its coefficients.

| Model | Basis |
|---|---|
| MP | `z[n-m] * abs(z[n-m])**(p-1)` |
| EMP | `z[n] * abs(z[n-m])**(p-1)` |
| GMP aligned | MP basis |
| GMP lagging | `z[n-m] * abs(z[n-m-q])**(p-1)` |
| GMP leading | `z[n-m] * abs(z[n-m+q])**(p-1)` |
| Volterra | Product of `(p+1)/2` direct `z` factors and `(p-1)/2` conjugate factors |
| LUT | `z[n] * complex_gain(abs(z[n]))` |

EMP keeps only one linear term, `z[n]`; repeating it for every delay would produce identical columns. Its term count is `1 + (M+1)*(P-1)/2`. GMP cross branches start at `P=3, M=1, Q=1`, with `m=0..M` and `q=1..Q`. Leading terms can require future samples. Context is respected, and a causal implementation would need the latency recorded in the hardware estimate. The aligned/lagging/leading organization follows [Morgan et al., 2006](https://doi.org/10.1109/TSP.2006.879264).

Volterra enumerates combinations with replacement within each factor group, avoiding permutations of identical products. `MaxDelaySpread` keeps terms satisfying `max(delays)-min(delays) <= D`. Generation and `PruningRule` are separate. Retained terms are serialized so a reloaded model reproduces custom pruning.

Polynomial fitting uses `numpy.linalg.lstsq` after balancing column norms. Coefficients are exported in the original basis, along with rank, singular values of the original Phi, and condition numbers of both original and balanced matrices. Rejection uses rank and the balanced condition number: different monomial units can inflate the original condition number without implying dependence. The implementation never forms or inverts the normal-equation matrix. `rcond` and `condition_limit` are configurable.

LUT centers use uniform amplitude, uniform power, or quantiles. Nearest interpolation chooses the closest center. Linear interpolation blends adjacent centers and saturates at the endpoints. Nodes are estimated only from training and reused for validation and test. Nearest fitting is exact least squares over disjoint cells; linear fitting includes interpolation weights in the least-squares regressors. Empty nearest cells are interpolated from trained cells, and effective rank is recorded. Rank-deficient linear LUTs are rejected. Zero-valued samples are handled without division by the sample.

## Training BIC search

`optimization.stopping_rule: "bic"` uses the criterion from [On the Optimum Number of Coefficients of Sparse Digital Predistorters: A Bayesian Approach](https://doi.org/10.1109/LMWC.2020.3027878), equations (6), (9), and (13). Here N is the number of scored complex training samples, and n is the number of independently fitted complex coefficients:

```text
penalty_db_per_coefficient = 10*log10(2*N)/N
J = NMSE_train_dB + n*penalty_db_per_coefficient
accept a candidate when J_candidate < J_current
equivalently: training_NMSE_improvement_dB > delta_n*penalty_db_per_coefficient
```

This is BIC in NMSE units, equivalent to the original BIC up to a positive scale and a shared constant. Validation is not used to calculate it. The RSS formulation assumes zero-mean residuals. Temporal correlation and PN9 periodicity affect its statistical interpretation; effective sample size is not automatically estimated.

At each step, neighbors are fitted and the smallest J is chosen; hardware estimates break exact ties. This mode does not train maximum-size reference models or use `target_fraction`, `min_improvement`, `plateau_patience`, or `tie_tolerance`. Polynomial steps that leave the basis unchanged, such as increasing M for a linear EMP, are crossed to reach genuine basis expansions. This is structural hill climbing, not DOMP. The root pipeline additionally implements bounded exploration beyond a non-improving candidate.

The penalty counts fitted coefficient rank rather than hardware cost. For valid polynomial fits, rank equals the number of terms. Applying the criterion to LUTs is conditional on their training-estimated grid: it penalizes fitted gain rank, not node estimation. It does not reproduce the paper's GMP experiment.

Standalone stopping reasons:

- `bic_minimum_reached`: no valid neighbor improves J.
- `search_limit_reached`: no allowed neighbors remain.
- `max_iterations`: the iteration budget was reached.
- `numerical_instability`: no stable neighbors are available.
- `numerical_frontier_incomplete`: stable neighbors do not improve J, but others could not be evaluated stably.

The numerical failures are not presented as a proven BIC plateau. Stopping is local to the explored terms and bounds; it neither proves a global optimum nor geometrically detects an elbow.

Results include `bic_score_db`, coefficient count, and the penalty per coefficient. Traces retain NMSE, coefficient-count and J changes, including rejected candidates. `progress_fraction` and `target_fraction` are null in this mode. Validation is reported without selecting the structure, and the selected standalone model is evaluated on test once, without refitting on validation.

## Legacy reference-fraction mode

Available only with `optimization.stopping_rule: "reference_fraction"`.

Each architecture has its own baseline and large reference configurations within the configured bounds. These are not global-optimum guarantees. An ill-conditioned reference is recorded but excluded from the target; if none is valid, the baseline is retained with `numerical_instability`.

```text
error = sum(abs(x - x_hat)**2) / sum(abs(x)**2)
NMSE_dB = 10*log10(error)
progress = (error_baseline - error_candidate) / (error_baseline - error_reference)
```

Progress is interpolated in linear error, not dB. If the reference fails to improve the baseline, the result is `performance_plateau` with explained zero progress. Progress can exceed one when a candidate outperforms the estimated reference.

Each neighbor increases a configuration dimension; LUT may also change binning at equal cost. Valid neighbors are fitted and cached. Selection uses progress improvement per complexity increment. Equal-complexity alternatives prioritize lower validation error. Within `tie_tolerance * error_baseline`, ties are resolved by real-equivalent multipliers, memory, and real-equivalent adders, in that order. Differences and decisions are recorded, and visited configurations are not revisited.

`min_improvement` is error reduction divided by baseline error; `plateau_patience` permits a limited number of neutral steps. If exploration worsens performance, the best visited state is returned rather than automatically choosing a large reference. The Pareto frontier covers evaluated valid candidates only, using validation error and structural complexity.

Stopping reasons are `target_reached`, `performance_plateau`, `max_iterations`, `no_valid_neighbors`, and `numerical_instability`. Only the selected model is evaluated on test, once per architecture, without validation refitting.

## Standalone configuration limits

| Architecture | Reference limits |
|---|---|
| MP / EMP | P <= 15, M <= 12 |
| GMP | P <= 15, M <= 12, Q <= 4 |
| Pruned Volterra | P <= 9, M <= 6, D <= 2 |
| LUT | 8 to 1024 nodes, geometric doubling |

These are resource bounds, not demonstrated capacity limits. They were increased from the earlier MP P7/M3 and smaller bounds for the other families. `max_basis_terms` checks maximum polynomial bases when preparing context; it is not a general RAM limit. Volterra grows combinatorially. The unified pipeline has its own configuration in the repository-root `pipeline.json`, including additional candidate and memory budgets.

## Hardware estimates

Results include parameter counts, complex/real multipliers and adders, delays, LUTs, entries, bits, cycles per sample, and an estimated clock-rate multiplier. Real operator counts exclude complex operators and must not be added without conversion. A complex multiplier is represented as four real multiplications plus two real additions. The 18-bit word size is an estimation assumption; training uses floating point without quantization simulation.

- `DirectHardwareEstimator` traverses actual basis terms, shares envelope powers by delay, counts Volterra products and accumulation, and includes coefficient and delay memory. LUT accounting includes gains, nodes, inverse widths for linear interpolation, searching, and square root operations.
- `LUTAssistedHardwareEstimator` replaces higher envelope-power products with parameterized tables. Its quantization error is not simulated.
- `MultiplierSharingEstimator` sizes shared banks and intermediate registers, providing a conservative serial cycle bound assuming single-cycle operators. It is neither an optimal scheduler nor an Fmax estimate.

The latter two are conceptual alternatives rather than formulas from a specific Guan-Zhu architecture reused for unrelated bases. Select `hardware.mode` as `lut_assisted` or `multiplier_sharing`, with `power_lut_entries` or `sharing_factor` respectively.

Estimates exclude interfaces, control, y/G normalization, and actual FPGA synthesis. Required clock rate is a cycles-per-sample factor, not proof of timing feasibility. Structural complexity is kept separate from hardware resources.

## Standalone results and reloading

Each run creates `data/dpd_results/run_<UTC>/` without overwriting earlier runs:

- `manifest.json`: configuration, G, versions, partitions, guards, metadata, and hashes.
- `MP.json`, `EMP.json`, `GMP.json`, `PrunedVolterra.json`, `LUT.json`: models, coefficients or LUT values, metrics, and optimization traces.
- `summary.json` and `summary.md`: compact comparison.
- `data/dpd_results/latest_run.json`: latest completed run location.

Candidates are stored once and referenced by canonical configuration in the baseline, references, and iterations. Fit diagnostics are at `candidate.model.diagnostics`. Rejected candidates include reasons. An invalid baseline produces no model or test result and is marked `numerical_instability`.

JSON is strict (`allow_nan=False`): complex numbers, arrays, and nonfinite values use tagged objects. An exact fit's negative-infinite NMSE is encoded as `{"__nonfinite__": "-inf"}`. Pickle is not used.

```python
from pathlib import Path
from dpd.pipeline import TrainingPipeline
from dpd.serialization import JSONWriter

pipeline = TrainingPipeline("dpd_config.json")
data = pipeline.prepare_data()
latest = JSONWriter.read("data/dpd_results/latest_run.json")
model = JSONWriter.load_model(Path(latest["directory"]) / "MP.json")
x_hat = model.predict(data.test)
# x_hat corresponds to data.test.indices. G is saved in manifest.json.
```

These are inverse-identification/postdistortion metrics, not EVM or ACPR/ACLR of a physically linearized PA.

## Extension and tests

Another architecture implements `Trainer` and `SearchPolicy` and registers in `TRAINERS` and `POLICIES`. It can supply a custom hardware estimator. `HyperparameterOptimizer` does not require modification. Custom Volterra pruning should share the same `PruningRule` between trainer and search policy.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests_dpd -v
```

Tests cover regressors, pruning, LUT indexing/interpolation, least squares, NMSE, linear progress, neighbors, costs, stopping, numerical rejection, training-only gain, guards, exclusion of test data from selection, hardware, and JSON reload. Historical integration tests require aligned local CSV fixtures excluded from Git. Processor tests are retained in `../procesador_iq_pa/tests/`. See the root `VERIFICATION.md` for the current verification record; old standalone fixture availability must not be confused with a failure of the unified two-dataset flow.
