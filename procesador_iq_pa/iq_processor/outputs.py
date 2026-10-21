"""Explicit data exports and plots using a non-interactive backend."""
from pathlib import Path
import csv
import json
import numpy as np


class JSONWriter:
    def write(self, metadata, path):
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def save_table(path, columns, names):
    matrix = np.column_stack(columns)
    np.savetxt(path, matrix, delimiter=",", header=",".join(names), comments="", fmt="%.17g")


class CSVWriter:
    def write(self, result, directory):
        a, s = result.alignment, result.selection
        save_table(directory/"aligned_waveform.csv",
                   [np.arange(len(a.x)), a.time, a.input_indices, a.output_positions,
                    a.time, a.output_times, a.x.real, a.x.imag, a.y.real, a.y.imag],
                   ["aligned_index", "time_reference_s", "input_original_index", "output_original_position",
                    "input_original_time_s", "output_original_time_s", "I_in", "Q_in", "I_out", "Q_out"])
        save_table(directory/"correlation_diagnostics.csv", a.lag_table.T,
                   ["lag_samples", "correlation_magnitude", "normalized_correlation", "overlap_samples"])
        paths = ["aligned_waveform.csv", "correlation_diagnostics.csv"]
        if s is not None:
            e = s.evm
            save_table(directory/"selected_symbols.csv",
                       [np.arange(len(s.x)), s.indices, a.time[s.indices], a.input_indices[s.indices],
                        a.output_positions[s.indices], s.x.real, s.x.imag, s.y.real, s.y.imag,
                        e.reference.real, e.reference.imag, e.error.real, e.error.imag, np.abs(e.error)],
                       ["symbol_index", "aligned_sample_index", "time_reference_s", "input_original_index",
                        "output_original_position", "I_in_symbol", "Q_in_symbol", "I_out_symbol", "Q_out_symbol",
                        "I_ref", "Q_ref", "error_I", "error_Q", "error_magnitude"])
            with (directory/"symbol_phase_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["phase", "symbol_count", "evm_percent", "alpha_real", "alpha_imag", "status"])
                writer.writeheader()
                writer.writerows(s.candidates)
            paths += ["selected_symbols.csv", "symbol_phase_candidates.csv"]
        return paths


class Grapher:
    def plot(self, result, directory):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        a, s = result.alignment, result.selection
        paths = []
        plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2,
                             "axes.spines.top": False, "axes.spines.right": False})

        def save(fig, name):
            fig.savefig(directory/name, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(name)

        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(a.lag_table[:, 0], a.lag_table[:, 1], color="#147d86")
        ax.axvline(a.integer_delay, color="#ae5f16", linestyle="--", label=f"Integer delay: {a.integer_delay}")
        ax.set(xlabel="Delay [samples]", ylabel="Correlation magnitude", title="Complex cross-correlation")
        ax.legend()
        save(fig, "01_cross_correlation.png")

        n = min(len(a.x), 512)
        t = (a.time[:n]-a.time[0])*1e9
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        for ax, component, label in ((axes[0], np.real, "I"), (axes[1], np.imag, "Q")):
            ax.plot(t, component(a.x[:n]), label="Input", color="#147d86")
            ax.plot(t, component(a.y[:n]), label="Output", color="#ae5f16", alpha=.8)
            ax.set_ylabel(f"{label} [CSV units]")
            ax.legend(loc="upper right")
        axes[0].set_title(f"Aligned waveforms: first {n} samples, without gain compensation")
        axes[1].set_xlabel("Time from aligned start [ns]")
        save(fig, "02_aligned_waveforms.png")
        if s is None:
            return paths
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        valid = [c for c in s.candidates if c["evm_percent"] is not None]
        ax.plot([c["phase"] for c in valid], [c["evm_percent"] for c in valid], "o-", color="#147d86")
        ax.scatter([s.phase], [s.evm.rms_percent], color="#ae5f16", s=65, zorder=3)
        ax.set(xlabel="Phase relative to aligned start [samples]", ylabel="RMS EVM [%]",
               title=f"Phase selection: p={s.phase}, EVM={s.evm.rms_percent:.5f}% (same record)")
        ax.set_xticks([c["phase"] for c in s.candidates])
        save(fig, "03_evm_vs_phase.png")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        axes[0].scatter(s.x.real, s.x.imag, s=10, alpha=.55, color="#147d86", label="Measured input")
        axes[0].set_title("One sample per symbol: input")
        axes[0].legend()
        axes[1].scatter(s.y.real, s.y.imag, s=10, alpha=.5, color="#ae5f16", label="Measured output")
        axes[1].scatter(s.evm.reference.real, s.evm.reference.imag, s=14, marker="+", color="#147d86", label="Reference αX")
        axes[1].set_title("Output versus fitted linear reference")
        axes[1].legend()
        for ax in axes:
            ax.set(xlabel="I [CSV units]", ylabel="Q [CSV units]")
            ax.set_aspect("equal", adjustable="box")
        save(fig, "04_constellations.png")
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(np.abs(s.evm.error), color="#147d86", linewidth=.8)
        ax.set(xlabel="Symbol index", ylabel="|Y − αX| [CSV units]", title="Error per symbol")
        save(fig, "05_symbol_error.png")
        return paths


def write_summary(result, path):
    a, s, m = result.alignment, result.selection, result.metadata
    lines = ["PA I/Q PROCESSOR", "",
             f"Aligned samples: {len(a.x)}",
             f"Index delay: {a.integer_delay+a.fractional_delay:.9g} samples",
             f"Time delay (including time origins): {a.delay_s:.9g} s",
             f"Observed sample rate: {m['processing']['sampling_observed']['sample_rate_hz']:.9g} Hz"]
    if s:
        lines += [f"Selected symbol phase: {s.phase}", f"Symbols: {len(s.x)}",
                  f"RMS EVM fitted on the same record: {s.evm.rms_percent:.9g} %",
                  f"RMS EVM [dB]: {s.evm.rms_db}", f"Alpha: {s.evm.alpha}"]
    else:
        lines.append("EVM skipped: "+m["processing"]["stage_status"]["symbols_evm"]["reason"])
    lines += ["", "WARNINGS:"] + m["processing"]["warnings"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
