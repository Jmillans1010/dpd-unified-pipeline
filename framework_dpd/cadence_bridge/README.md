# DPD to Cadence adapter

This adapter evaluates saved models and exports stimuli without modifying model coefficients, optimizers, or preprocessing. Python generates the stimulus; Spectre is not launched. The [unified pipeline](../../README.md) adds the four-category run layout, automatic preprocessing, and a 3 V zero-pair policy. The standalone adapter described here rejects out-of-range stimuli instead of zeroing them.

## Existing components

- `dpd/contracts.py`: `RawIQData` contains x, y, and time. `DataPartition` contains x, z=y/G, time, and scored indices; its target is x[indices].
- `dpd/serialization.py`: `JSONWriter.load_model` reconstructs a `TrainedModel` and its complex coefficients. Arrays use `__ndarray__` tags and complex numbers use `__complex__: [real, imag]`.
- `TrainedModel.predict(DataPartition)` dispatches through `TRAINERS`. MP, EMP, and GMP evaluate envelope basis terms. Volterra uses direct and conjugate products. LUT models use nodes, complex gains, and nearest or linear interpolation.
- `dpd/trainers.py` requires valid context without implicit padding or trimming. Delays come from serialized basis terms, not just the M label.
- `../procesador_iq_pa/` imports and aligns waveforms, writes `aligned_waveform.csv`, and retains metadata. It can process new Cadence output on a modern Python installation; it has not been ported to Python 3.6.

The original adapter check used 8766 complex samples from `framework_dpd/data/input/Validation/aligned_waveform.csv`, with Ts approximately 250 ps and Fs approximately 4 GHz. The input started at 8 ns. That origin was preserved in metadata while the exported stimulus explicitly restarted at 0 s. Its final linear PWL point was at 2.19125 us. Samples preceding the input record are not reconstructed, and replaying a segment does not guarantee the PA's original initial state. Historical raw data are not included in Git.

Input samples are volts, as confirmed by the user and the I_IN/Q_IN netlist interface. They are not peak/RMS-normalized or mapped to [-1,1]. Normalization by G was used during inverse identification. Predistortion applies `x_dpd=F(x)` without dividing x by G. Ideally `PA(F(x))` approaches `G*x`; this must be tested with another PA simulation, not inferred from postdistortion metrics.

## Interfaces and commands

- `core.py`: `DPDModel.apply`, I/Q loading, signal statistics, and the shared exporter.
- `__main__.py`: CLI, metadata, and diagnostic figures.
- `feedback.py`: NMSE, relative RMS error, AM/AM, AM/PM, and EVM with explicit symbol indices.
- `test_bridge.py`: equivalence checks, PWL checks, and validation tests.

From `framework_dpd`, with existing aligned data and saved models:

```python
from cadence_bridge import DPDModel, load_iq, export_to_cadence

x, Fs, source_t0 = load_iq('data/input/Validation/aligned_waveform.csv')
model = DPDModel('data/dpd_results/run_TIMESTAMP/MP.json')
x_dpd = model.apply(x)
V_SCALE = 1.0  # V/V: the input already uses volts.
metadata = export_to_cadence(
    x_dpd, Fs, 'new_export/I_DPD.txt', 'new_export/Q_DPD.txt',
    voltage_scale=V_SCALE, max_abs_voltage=3.0)
```

CLI example, always using a new output directory:

```text
python -m cadence_bridge --model data/dpd_results/run_TIMESTAMP/MP.json --input data/input/Validation/aligned_waveform.csv --output data/cadence_exports/example_MP --voltage-scale 1 --max-abs-voltage 3
python -m unittest cadence_bridge.test_bridge -v
```

Replace MP with EMP, GMP, PrunedVolterra, or LUT as needed. `--diagnose-only` produces samples, plots, and statistics without PWL files. `--backend portable` forces the portable evaluator. `--no-plots` avoids Matplotlib. Diagnostic image names are `time.png` and `spectrum_constellation.png`. This standalone diagnostic constellation includes waveform samples; the unified pipeline's final comparison plots use symbol instants only.

The voltage limit is an absolute value per I/Q channel, not peak-to-peak voltage. It is checked after `V_SCALE`. The standalone adapter does not clip, rescale automatically, omit extreme samples, or zero them. Nonfinite signals, irregular clocks, existing output files, and voltages above the configured limit are rejected. Statistics are computed before export. A linear PWL cannot exceed its endpoint voltage range within an interval.

## Memory, lookahead, and timing

- `zero_pad`, the default, pads the context required by actual basis delays and returns N samples at t[n]=n/Fs without shifting sample positions. Padding-dependent boundaries are recorded. Digital padding does not guarantee analog PA settling.
- `trim` removes past/future boundaries without valid context, records `first_original_index` and `last_original_index`, and explicitly resets time to zero. Diagnostic references use the same retained indices.
- GMP leading terms need future samples. They are available offline; streaming would require buffering and latency equal to the maximum lookahead. `offline_lookahead_s` and latency are recorded. This adapter does not simulate a real-time causal processor.

## Interpolation

Linear PWL interpolates between x[n] and x[n+1]. It is an explicit waveform-reconstruction choice, not proof that the original wsource has the same waveform between samples. Compare DPD with a no-DPD case using the same exporter, clock, interface, and interpolation method.

ZOH approximates a sample-and-hold DAC, ideally with its reconstruction filter when that matches the intended hardware. Request `--interpolation zoh --transition-s 1e-12`. The 1 ps value is an example, not a DAC specification. The required range is `0 < transition_s < Ts`. For x0, x1, x2, the exported points are:

```text
0          x0
Ts         x0
Ts+eps     x1
2Ts        x1
2Ts+eps    x2
3Ts        x2
```

Timestamps are strictly increasing. The previous value is held until the clock edge, and the transition finishes eps later. This is a finite ramp, not an ideal discontinuity or duplicate timestamps; the simulator must resolve those transitions. Linear export ends at (N-1)Ts; ZOH includes the final interval up to NTs. The final voltage is then held without automatic repetition or a return to zero.

## Power and spectra

Statistics include I/Q/envelope peaks, complex RMS `sqrt(mean(I**2+Q**2))`, `mean_square_V2`, and `PAPR=10*log10(max(abs(x)**2)/mean(abs(x)**2))`. Squared volts are not labeled watts. With an explicit `--resistance-ohm R`, the adapter also reports ideal RF power `mean(abs(x)**2)/(2*R)`. This assumes `vRF=I*cos-Q*sin` without gain, loss, or mismatch; it is not a measurement of power delivered to this testbench's PA.

Spectra use a complex Hann-windowed FFT with amplitude referenced to 1 V, not dBm, PSD, or ACLR. Figures do not modify samples.

## Connecting the available testbench

The original inspection used the local `../procesador_iq_pa/data/input/input` netlist for cell `TO2608_DPDTESTING/dpd_testbench`. This was a saved file, not a live Virtuoso session. Check that the file corresponds to the intended circuit revision before applying these connections.

```text
W0 (I_IN 0 Q_IN 0) wsource ...
I1 (I_IN Q_IN RF_IN net11) IQ_modulator ... flo=240e9
I2 (RF_IN net8 net7) balun ...
I0 (net8 net7 net4 net1 vcc!) PA_COMPONENT_EMCoreCorrect_FULLEM_tightlayinv
```

For the I/Q interface, replace or disable W0 rather than placing ideal voltage sources in parallel. Add two analogLib/vsource instances with `type=pwl` and Waveform Entry Method=File, or `vpwlf` if available. Connect negative terminals to 0 and positive terminals to I_IN and Q_IN respectively. Use absolute server paths in the file fields, and copy both files to the server before simulation. Retain the existing modulator, balun, PA, demodulator, and filters. Do not connect baseband I/Q directly to the PA's RF input nodes.

```spectre
V_I_DPD (I_IN 0) vsource type=pwl file="/server/path/I_DPD.txt"
V_Q_DPD (Q_IN 0) vsource type=pwl file="/server/path/Q_DPD.txt"
```

PWL text files are referenced by the source, not added as model/include files. Any added CDF scaling or offset must be reflected in metadata. Check measured I_IN and Q_IN against the exported files: source and modulator impedances can affect reproduction and power, even when voltage appears to match.

The inspected netlist used `stop=2.2u`, `outputstart=8n`, `strobeperiod=250p`, and `maxstep=100f` at 240 GHz RF. Exported time starts at zero, so the original 8 ns offset is not part of the PWL file. To capture the full segment, use `outputstart=0` and a stop time matching metadata `end_time_s`. Apply any extra settling period explicitly and identically to the baseline, then account for it during alignment. Do not replace the RF integration step with the 250 ps baseband sampling interval: a 240 GHz carrier has a period of approximately 4.17 ps. Strobing controls saved output times; it does not by itself resolve the carrier.

For an RF-only PA interface, an I/Q modulator or behavioral source must generate `vRF(t)=I_DPD(t)*cos(2*pi*fc*t)-Q_DPD(t)*sin(2*pi*fc*t)`, with gain and Q sign consistent with the demodulator. This testbench retains I1 and its differential balun. A 4 GHz baseband grid cannot directly export a 240 GHz RF waveform; frequency conversion occurs within Cadence. The internal rfLib sign convention has not been established from its Verilog-A model, so verify it with an I/Q tone if replacing I1.

Previously consulted Cadence references:

- [Using a file for a vpwlf source](https://community.cadence.com/cadence_technology_forums/f/custom-ic-design/35602/using-a-file-for-vpwlf-source)
- [Syntax of a vpwlf file](https://community.cadence.com/cadence_technology_forums/f/custom-ic-design/11426/syntax-of-vpwlf-file)

## Returning Cadence results to Python

1. Save I_IN, Q_IN, I_OUT, Q_OUT, and their times. Also preserve RF_IN/RF_OUT if evaluating RF spectra. Simulate a no-DPD reference using the same PWL method.
2. Export the waveforms to a new dataset directory and process them without overwriting training. Preserve delay, alignment, and source hashes.
3. Compare aligned demodulated output with the original desired signal in `samples.csv`, not the predistorted input. Match time/indices and explicitly account for settling; equal lengths do not establish alignment.
4. `analyze_feedback(reference, measured, gain, symbol_indices=None)` computes NMSE, AM/AM, and AM/PM. Reuse the original G for an absolute comparison. Any additional gain compensation must be recorded separately.
5. EVM requires verified symbol instants and an appropriate reference. The legacy processor configuration had unverified samples-per-symbol metadata; the unified pipeline resolves this from confirmed source settings and the CSV clock. Do not silently substitute a symbol rate or phase.
6. Retraining is a separate operation on the new dataset. The adapter does not automatically change coefficients or demonstrate ILA convergence.

The unified `--feedback-dir` command automates the reference lookup and writes feedback diagnostics and comparisons under its four-folder layout.

## CentOS 7 / Python 3.6.8

The full framework uses dataclasses, modern type annotations, and numpy.typing and cannot be imported in Python 3.6. The adapter automatically uses its portable evaluator of serialized basis terms without redesigning or fitting models. On Python >=3.10 it can use native prediction.

Copy `cadence_bridge/`, the model JSON files, and the input CSV into a common server directory, then run `python3 -m cadence_bridge ... --backend portable`. Python 3.6 dependencies are `numpy==1.19.5` and, for plots, `matplotlib==3.3.4`. Modern machines can use their current NumPy/Matplotlib environment.

Historical local checks covered native/portable equivalence for all five model families and both boundary policies, linear/ZOH ASCII export, limits, feedback, and Python 3.6 syntax parsing. Actual execution in Python 3.6, CentOS, or Spectre remains unverified. Some original adapter tests require historical local data that are not published with the repository.
