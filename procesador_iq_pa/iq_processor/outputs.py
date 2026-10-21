"""Exportaciones sin cálculos ocultos y gráficas con backend no interactivo."""
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
        ax.axvline(a.integer_delay, color="#ae5f16", linestyle="--", label=f"Retardo entero: {a.integer_delay}")
        ax.set(xlabel="Retardo [muestras]", ylabel="Magnitud de correlación", title="Correlación cruzada compleja")
        ax.legend()
        save(fig, "01_cross_correlation.png")

        n = min(len(a.x), 512)
        t = (a.time[:n]-a.time[0])*1e9
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        for ax, component, label in ((axes[0], np.real, "I"), (axes[1], np.imag, "Q")):
            ax.plot(t, component(a.x[:n]), label="Entrada", color="#147d86")
            ax.plot(t, component(a.y[:n]), label="Salida", color="#ae5f16", alpha=.8)
            ax.set_ylabel(f"{label} [unidad CSV]")
            ax.legend(loc="upper right")
        axes[0].set_title(f"Formas de onda alineadas: primeras {n} muestras, sin compensar ganancia")
        axes[1].set_xlabel("Tiempo desde el inicio alineado [ns]")
        save(fig, "02_aligned_waveforms.png")
        if s is None:
            return paths
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        valid = [c for c in s.candidates if c["evm_percent"] is not None]
        ax.plot([c["phase"] for c in valid], [c["evm_percent"] for c in valid], "o-", color="#147d86")
        ax.scatter([s.phase], [s.evm.rms_percent], color="#ae5f16", s=65, zorder=3)
        ax.set(xlabel="Fase respecto del inicio alineado [muestras]", ylabel="EVM RMS [%]",
               title=f"Selección de fase: p={s.phase}, EVM={s.evm.rms_percent:.5f}% (mismo registro)")
        ax.set_xticks([c["phase"] for c in s.candidates])
        save(fig, "03_evm_vs_phase.png")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        axes[0].scatter(s.x.real, s.x.imag, s=10, alpha=.55, color="#147d86", label="Entrada medida")
        axes[0].set_title("Una muestra por símbolo: entrada")
        axes[0].legend()
        axes[1].scatter(s.y.real, s.y.imag, s=10, alpha=.5, color="#ae5f16", label="Salida medida")
        axes[1].scatter(s.evm.reference.real, s.evm.reference.imag, s=14, marker="+", color="#147d86", label="Referencia αX")
        axes[1].set_title("Salida frente a referencia lineal ajustada")
        axes[1].legend()
        for ax in axes:
            ax.set(xlabel="I [unidad CSV]", ylabel="Q [unidad CSV]")
            ax.set_aspect("equal", adjustable="box")
        save(fig, "04_constellations.png")
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(np.abs(s.evm.error), color="#147d86", linewidth=.8)
        ax.set(xlabel="Índice de símbolo", ylabel="|Y − αX| [unidad CSV]", title="Error por símbolo")
        save(fig, "05_symbol_error.png")
        return paths


def write_summary(result, path):
    a, s, m = result.alignment, result.selection, result.metadata
    lines = ["PROCESADOR I/Q PARA PA", "",
             f"Muestras alineadas: {len(a.x)}",
             f"Retardo por índices: {a.integer_delay+a.fractional_delay:.9g} muestras",
             f"Retardo temporal (incluye orígenes de tiempos): {a.delay_s:.9g} s",
             f"Frecuencia de muestreo observada: {m['processing']['sampling_observed']['sample_rate_hz']:.9g} Hz"]
    if s:
        lines += [f"Fase de símbolo seleccionada: {s.phase}", f"Símbolos: {len(s.x)}",
                  f"EVM RMS ajustado sobre el mismo registro: {s.evm.rms_percent:.9g} %",
                  f"EVM RMS [dB]: {s.evm.rms_db}", f"Alfa: {s.evm.alpha}"]
    else:
        lines.append("EVM omitido: "+m["processing"]["stage_status"]["symbols_evm"]["reason"])
    lines += ["", "ADVERTENCIAS:"] + m["processing"]["warnings"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
