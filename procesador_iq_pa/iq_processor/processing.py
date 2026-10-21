"""Algoritmos numéricos. Convención: y[n] = a*x[n-D] implica D positivo."""
from dataclasses import asdict
from datetime import datetime, timezone
import platform
import numpy as np
from . import __version__
from .models import (IQSignal, InputData, AlignmentResult, EVMResult,
                     SymbolSelectionResult, ProcessingResult, ProcessingError)


class SignalValidator:
    def validate(self, data, config):
        measurements = {}
        for name, signal in (("input", data.input), ("output", data.output)):
            if len(signal.time) != len(signal.values) or len(signal.time) < 3:
                raise ProcessingError(f"Longitud inválida en {name}.")
            if not np.isfinite(signal.time).all() or not np.isfinite(signal.values).all():
                raise ProcessingError(f"Valores no finitos en {name}.")
            differences = np.diff(signal.time)
            period = float(np.median(differences))
            error = float(np.max(np.abs(differences-period)))
            if np.any(differences <= 0) or error > config.time_atol_s + config.time_rtol*period:
                raise ProcessingError(f"Tiempos de {name} duplicados, decrecientes o no uniformes. No se remuestrea automáticamente.")
            measurements[name] = {"sample_period_s": period, "maximum_step_error_s": error,
                                  "first_time_s": float(signal.time[0]), "last_time_s": float(signal.time[-1]),
                                  "count": len(signal.time)}
        period = measurements["input"]["sample_period_s"]
        if not np.isclose(period, measurements["output"]["sample_period_s"], rtol=config.time_rtol, atol=config.time_atol_s):
            raise ProcessingError("Entrada y salida tienen frecuencias de muestreo incompatibles.")
        # La tolerancia acumulada también debe ser pequeña: no admitir deriva de reloj.
        n = min(len(data.input.time), len(data.output.time))
        drift = (data.output.time[:n]-data.output.time[0]) - (data.input.time[:n]-data.input.time[0])
        if np.max(np.abs(drift)) > config.time_atol_s + config.time_rtol*period:
            raise ProcessingError("Las rejillas temporales acumulan deriva; se requiere remuestreo explícito.")
        return {"sample_period_s": period, "sample_rate_hz": 1/period, "channels": measurements,
                "accepted_step_tolerance_s": config.time_atol_s + config.time_rtol*period,
                "maximum_interchannel_grid_drift_s": float(np.max(np.abs(drift)))}


class SignalPreprocessor:
    def prepare(self, data, config):
        prepared, report = [], {}
        for name, signal in (("input", data.input), ("output", data.output)):
            cutoff = signal.time[0] + config.additional_initial_crop_s
            keep = signal.time >= cutoff - config.time_atol_s
            if np.count_nonzero(keep) < 3:
                raise ProcessingError("El recorte inicial deja menos de tres muestras.")
            values = signal.values[keep].copy()
            mean = complex(np.mean(values)) if config.remove_dc else 0j
            values -= mean
            prepared.append(IQSignal(signal.time[keep], values, signal.original_indices[keep]))
            report[name] = {"cropped_samples": int(np.count_nonzero(~keep)),
                            "removed_dc_real": mean.real, "removed_dc_imag": mean.imag,
                            "first_retained_time_s": float(signal.time[keep][0])}
        return InputData(*prepared, data.sources), report


def complex_correlation(x, y):
    """Correlación LINEAL por FFT: R[l] = sum conj(x[n])*y[n+l]."""
    total = len(x) + len(y) - 1
    size = 1 << (total - 1).bit_length()
    result = np.fft.ifft(np.fft.fft(y, size) * np.fft.fft(np.conjugate(x[::-1]), size))[:total]
    return np.arange(-(len(x)-1), len(y)), result


def sinc_interpolate(values, positions, half_width):
    """Sinc con ventana Hann y normalización DC; exige soporte completo."""
    base = np.floor(positions).astype(int)
    indices = base[:, None] + np.arange(-half_width+1, half_width+1)
    if indices.min() < 0 or indices.max() >= len(values):
        raise ProcessingError("Interpolación solicitada sin soporte de borde completo.")
    distance = positions[:, None] - indices
    weights = np.sinc(distance) * np.where(np.abs(distance) < half_width,
                                         0.5*(1+np.cos(np.pi*distance/half_width)), 0)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(values[indices]*weights, axis=1)


class FractionalDelayRefiner:
    def refine(self, lags, correlation, delay, allowed_limit):
        index = int(np.searchsorted(lags, delay))
        if abs(delay) >= allowed_limit or index == 0 or index == len(lags)-1:
            return 0., "Refinamiento omitido: pico en el límite de búsqueda."
        a, b, c = np.abs(correlation[index-1:index+2])
        denominator = 2*(a-2*b+c)
        if abs(denominator) <= np.finfo(float).eps*max(b, np.finfo(float).tiny):
            return 0., "Refinamiento omitido: pico demasiado plano."
        fraction = float((a-c)/denominator)
        if not np.isfinite(fraction) or abs(fraction) > 0.5:
            return 0., "Refinamiento omitido: ajuste parabólico fuera de ±0.5 muestras."
        return fraction, None


class SampleAligner:
    def align(self, data, sampling, config):
        x, y = data.input.values, data.output.values
        if np.vdot(x, x).real <= np.finfo(float).tiny or np.vdot(y, y).real <= np.finfo(float).tiny:
            raise ProcessingError("No se puede correlacionar una señal de energía nula.")
        lags, correlation = complex_correlation(x, y)
        overlap = np.minimum(len(x), len(y)-lags) - np.maximum(0, -lags)
        minimum = max(3, int(np.ceil(config.minimum_overlap_fraction*min(len(x), len(y)))))
        allowed = (np.abs(lags) <= config.max_delay_samples) & (overlap >= minimum)
        if not np.any(allowed):
            raise ProcessingError("No hay retardos con suficiente solapamiento.")
        candidates = np.flatnonzero(allowed)
        peak = int(candidates[np.argmax(np.abs(correlation[candidates]))])
        delay = int(lags[peak])
        warnings = []
        if len(candidates) > 1 and peak in (candidates[0], candidates[-1]):
            warnings.append("Pico de correlación en un límite: revise max_delay_samples y el solapamiento.")
        table = []
        for idx in candidates:
            lag = int(lags[idx])
            begin, end = max(0, -lag), min(len(x), len(y)-lag)
            energy = np.sqrt(np.vdot(x[begin:end], x[begin:end]).real * np.vdot(y[begin+lag:end+lag], y[begin+lag:end+lag]).real)
            table.append((lag, abs(correlation[idx]), abs(correlation[idx])/energy if energy > 0 else 0., end-begin))
        table = np.asarray(table)
        peaks = [i for i in candidates if 0 < i < len(lags)-1
                 and abs(correlation[i]) >= abs(correlation[i-1]) and abs(correlation[i]) >= abs(correlation[i+1])]
        alternatives = sorted(peaks, key=lambda i: abs(correlation[i]), reverse=True)[:5]
        if len(alternatives) > 1 and abs(correlation[alternatives[1]]) >= 0.98*abs(correlation[peak]):
            warnings.append("Hay picos de correlación similares; el retardo podría ser ambiguo.")
        fraction = 0.
        if config.fractional_refinement:
            fraction, warning = FractionalDelayRefiner().refine(lags, correlation, delay, config.max_delay_samples)
            if warning:
                warnings.append(warning)
        if abs(fraction) < 1e-12:
            fraction = 0.
        shift = delay + fraction
        n = np.arange(len(x))
        positions = n + shift
        if fraction:
            bases = np.floor(positions).astype(int)
            valid = (bases >= config.interpolation_half_width-1) & (bases+config.interpolation_half_width < len(y))
        else:
            valid = (positions >= 0) & (positions < len(y))
        n, positions = n[valid], positions[valid]
        if len(n) < minimum:
            raise ProcessingError("El refinamiento/recorte deja menos del solapamiento mínimo configurado.")
        aligned_y = sinc_interpolate(y, positions, config.interpolation_half_width) if fraction else y[positions.astype(int)]
        period = sampling["sample_period_s"]
        output_times = np.interp(positions, np.arange(len(y)), data.output.time) if fraction else data.output.time[positions.astype(int)]
        physical_delay = float(np.median(output_times-data.input.time[n]))
        report = {"sign_convention": "y[n] = alpha*x[n-D] => D positive for equal time origins",
                  "search_limit_samples": config.max_delay_samples, "minimum_overlap_samples": minimum,
                  "criterion": "maximum_absolute_complex_cross_correlation",
                  "correlation_implementation": "linear_fft_zero_padded",
                  "normalized_correlation_at_integer_peak": float(table[table[:, 0] == delay, 2][0]),
                  "top_local_peaks": [{"lag": int(lags[i]), "magnitude": float(abs(correlation[i]))} for i in alternatives],
                  "warnings": warnings, "aligned_count": len(n),
                  "input_original_interval": [int(data.input.original_indices[n[0]]), int(data.input.original_indices[n[-1]])],
                  "output_original_interval": [float(data.output.original_indices[0]+positions[0]), float(data.output.original_indices[0]+positions[-1])],
                  "integer_delay_samples": delay, "fractional_delay_samples": fraction,
                  "interpolation": "hann_windowed_sinc" if fraction else "none",
                  "interpolation_half_width": config.interpolation_half_width if fraction else None,
                  "fractional_estimator": "parabolic_correlation_magnitude" if config.fractional_refinement else "disabled",
                  "delay_s_method": "median of paired original times; interpolated output time if fractional",
                  "nominal_delay_s": float(data.output.time[0]-data.input.time[0]+shift*period),
                  "paired_delay_maximum_deviation_s": float(np.max(np.abs(output_times-data.input.time[n]-physical_delay)))}
        return AlignmentResult(data.input.time[n], x[n], aligned_y, data.input.original_indices[n],
                               data.output.original_indices[0]+positions, output_times,
                               delay, fraction, physical_delay, report, table)


class EVMCalculator:
    def calculate(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 1 or y.shape != x.shape or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ProcessingError("EVM necesita dos vectores finitos de igual longitud, no vacíos.")
        energy = float(np.vdot(x, x).real)
        if energy <= np.finfo(float).tiny:
            raise ProcessingError("EVM indefinido: entrada de energía nula.")
        alpha = complex(np.vdot(x, y)/energy)
        reference = alpha*x
        reference_energy = float(np.vdot(reference, reference).real)
        output_energy = float(np.vdot(y, y).real)
        if reference_energy <= max(np.finfo(float).tiny, np.finfo(float).eps*output_energy):
            raise ProcessingError("EVM indefinido: referencia lineal de energía nula o numéricamente despreciable.")
        error = y-reference
        ratio = float(np.sqrt(np.vdot(error, error).real/reference_energy))
        return EVMResult(alpha, reference, error, 100*ratio, 20*float(np.log10(ratio)) if ratio > 0 else None)


class SymbolTimingSelector:
    def select(self, alignment, sps, config):
        count = len(alignment.x)//sps
        if count < config.minimum_symbols:
            raise ProcessingError(f"Solo hay {count} símbolos completos; se requieren {config.minimum_symbols}.")
        if config.symbol_phase is not None and config.symbol_phase >= sps:
            raise ProcessingError("symbol_phase debe estar entre 0 y L-1.")
        candidates, evaluated = [], {}
        for phase in range(sps):
            indices = phase + np.arange(count)*sps
            try:
                result = EVMCalculator().calculate(alignment.x[indices], alignment.y[indices])
                evaluated[phase] = (indices, result)
                candidates.append({"phase": phase, "symbol_count": count, "evm_percent": result.rms_percent,
                                   "alpha_real": result.alpha.real, "alpha_imag": result.alpha.imag, "status": "valid"})
            except ProcessingError as exc:
                candidates.append({"phase": phase, "symbol_count": count, "evm_percent": None, "status": str(exc)})
        if not evaluated:
            raise ProcessingError("Ninguna fase tiene una referencia válida para EVM.")
        if config.symbol_phase is not None:
            phase = config.symbol_phase
            if phase not in evaluated:
                raise ProcessingError("La fase manual no tiene un EVM válido.")
        else:
            minimum = min(r.rms_percent for _, r in evaluated.values())
            phase = min(p for p, (_, r) in evaluated.items() if r.rms_percent <= minimum+config.phase_tie_tolerance_pp)
        indices, result = evaluated[phase]
        return SymbolSelectionResult(phase, indices, alignment.x[indices], alignment.y[indices], result, candidates)


class StatisticsCalculator:
    @staticmethod
    def signal(values):
        power = np.abs(values)**2
        average = float(np.mean(power))
        mean = complex(np.mean(values))
        return {"sample_count": len(values), "mean_real": mean.real, "mean_imag": mean.imag,
                "complex_rms": float(np.sqrt(average)), "peak_magnitude": float(np.max(np.abs(values))),
                "papr_db": float(10*np.log10(np.max(power)/average)) if average > 0 else None,
                "i_min": float(values.real.min()), "i_max": float(values.real.max()),
                "q_min": float(values.imag.min()), "q_max": float(values.imag.max()),
                "amplitude_units": "CSV values (volts if exported as voltage); no conversion to watts"}

    def calculate(self, raw, alignment, selection):
        result = {"raw_input": self.signal(raw.input.values), "raw_output": self.signal(raw.output.values),
                  "aligned_input": self.signal(alignment.x), "aligned_output": self.signal(alignment.y),
                  "evm_rms_percent": selection.evm.rms_percent if selection else None,
                  "evm_rms_db": selection.evm.rms_db if selection else None,
                  "symbol_count": len(selection.x) if selection else None}
        try:
            waveform = EVMCalculator().calculate(alignment.x, alignment.y)
            result["waveform_error"] = {"nmse_fitted_reference_db": waveform.rms_db,
                                         "beta_real": waveform.alpha.real, "beta_imag": waveform.alpha.imag,
                                         "scope": "all aligned samples; separate fit from symbol alpha"}
        except ProcessingError as exc:
            result["waveform_error"] = {"status": str(exc)}
        return result


class MetadataProcessor:
    def build(self, raw, metadata, config, sampling, preprocessing, alignment, selection, stats, stages):
        warnings = metadata.warnings + alignment.diagnostics["warnings"]
        if selection and config.symbol_phase is None:
            warnings.append("La fase y alfa se ajustan y evalúan sobre los mismos símbolos; EVM de ajuste, no validación independiente.")
        return {"simulation_config": metadata.simulation_config,
                "signal_config": metadata.signal_config,
                "processing": {"schema_version": "0.1", "processor_version": __version__,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python_version": platform.python_version(), "numpy_version": np.__version__,
                    "sources": raw.sources, "field_provenance": metadata.provenance,
                    "configuration": asdict(config), "sampling_observed": sampling,
                    "preprocessing": preprocessing, "alignment": {**alignment.diagnostics, "delay_s": alignment.delay_s},
                    "symbol_selection": {"phase": selection.phase, "samples_per_symbol_used": metadata.signal_config["samples_per_symbol"],
                                         "candidate_scores": selection.candidates,
                                         "scope": "same-record fitted EVM; not independent test",
                                         "phase_origin": "first sample of aligned waveform"} if selection else None,
                    "evm_configuration": {"reference": "measured_input_times_fitted_alpha",
                                          "normalization": "sum_squared_error / sum_squared_fitted_reference",
                                          "alpha_real": selection.evm.alpha.real if selection else None,
                                          "alpha_imag": selection.evm.alpha.imag if selection else None,
                                          "db_status": "finite" if selection and selection.evm.rms_db is not None else "zero_error_or_not_available"},
                    "stage_status": stages, "warnings": warnings,
                    "measurement_scope": "entire path between exported I/Q nodes, including any filters/modulators"},
                "statistics": stats}


class DataProcessor:
    def process(self, raw, metadata, config, sampling, sps):
        prepared, preprocessing = SignalPreprocessor().prepare(raw, config)
        alignment = SampleAligner().align(prepared, sampling, config)
        stages = {"validation": {"status": "completed"}, "alignment": {"status": "completed"}}
        selection = None
        if sps is None:
            stages["symbols_evm"] = {"status": "skipped", "reason": "Faltan muestras por símbolo o una tasa de símbolos verificable."}
        else:
            try:
                selection = SymbolTimingSelector().select(alignment, sps, config)
                stages["symbols_evm"] = {"status": "completed"}
            except ProcessingError as exc:
                stages["symbols_evm"] = {"status": "skipped", "reason": str(exc)}
        statistics = StatisticsCalculator().calculate(raw, alignment, selection)
        document = MetadataProcessor().build(raw, metadata, config, sampling, preprocessing,
                                             alignment, selection, statistics, stages)
        return ProcessingResult(alignment, selection, document)
