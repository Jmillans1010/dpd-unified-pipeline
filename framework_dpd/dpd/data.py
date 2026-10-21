from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
from .contracts import AlignedInputData, DataPartition, RawIQData


class CSVReader:
    """Solo lectura del contrato exportado por iq_processor."""

    def read(self, directory: str | Path) -> RawIQData:
        directory = Path(directory)
        path = directory / "aligned_waveform.csv"
        table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
        required = {"time_reference_s", "I_in", "Q_in", "I_out", "Q_out"}
        if not required.issubset(table.dtype.names or ()):
            raise ValueError(f"Columnas ausentes en {path}: {required}")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
        return RawIQData(np.atleast_1d(table["time_reference_s"]),
                         np.atleast_1d(table["I_in"] + 1j * table["Q_in"]),
                         np.atleast_1d(table["I_out"] + 1j * table["Q_out"]),
                         metadata, str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())


class DataProcessor:
    def __init__(self, past: int, future: int, guard: int | None = None):
        self.past, self.future = past, future
        self.guard = max(past, future) if guard is None else guard
        if min(past, future, self.guard) < 0 or self.guard < max(past, future):
            raise ValueError("Guarda insuficiente para la memoria máxima del experimento.")

    @staticmethod
    def validate(raw: RawIQData) -> None:
        if not (len(raw.x) == len(raw.y) == len(raw.time)) or len(raw.x) < 3:
            raise ValueError("Longitudes IQ inválidas.")
        if any(not np.all(np.isfinite(v)) for v in (raw.x, raw.y, raw.time)):
            raise ValueError("Datos IQ no finitos.")
        steps = np.diff(raw.time)
        if np.any(steps <= 0) or not np.allclose(steps, np.median(steps), rtol=1e-5, atol=1e-18):
            raise ValueError("La rejilla temporal debe ser creciente y uniforme.")

    @staticmethod
    def estimate_gain(x_train: np.ndarray, y_train: np.ndarray) -> complex:
        energy = np.vdot(x_train, x_train).real
        if energy <= np.finfo(float).tiny:
            raise ValueError("Training sin energía.")
        gain = complex(np.vdot(x_train, y_train) / energy)
        if abs(gain) <= np.finfo(float).tiny or not np.isfinite(gain):
            raise ValueError("Ganancia compleja nula/no finita.")
        return gain

    @staticmethod
    def normalize_output(y: np.ndarray, gain: complex) -> np.ndarray:
        return y / gain

    def align_signals(self, raw: RawIQData, max_delay: int = 128) -> RawIQData:
        """Para nuevos registros sin alinear; los CSV entregados no pasan por aquí."""
        self.validate(raw)
        if max_delay < 0 or max_delay >= len(raw.x) // 2:
            raise ValueError("Límite de alineación inválido.")
        def slices(delay: int) -> tuple[slice, slice]:
            n = len(raw.x)
            return slice(max(0, -delay), min(n, n-delay)), slice(max(0, delay), min(n, n+delay))
        delays = range(-max_delay, max_delay + 1)
        delay = max(delays, key=lambda d: abs(np.vdot(raw.x[slices(d)[0]], raw.y[slices(d)[1]])))
        sx, sy = slices(delay)
        return RawIQData(raw.time[sx], raw.x[sx], raw.y[sy],
                         {**raw.metadata, "dpd_alignment_delay": delay}, raw.source, raw.sha256)

    @staticmethod
    def subset(raw: RawIQData, start: int, end: int) -> RawIQData:
        return RawIQData(raw.time[start:end], raw.x[start:end], raw.y[start:end],
                         raw.metadata, raw.source, raw.sha256)

    def split_chronologically(self, raw: RawIQData, fractions: tuple[float, float] = (.6, .2)) -> tuple[RawIQData, ...]:
        a, b = fractions
        if a <= 0 or b <= 0 or a+b >= 1:
            raise ValueError("Fracciones de división inválidas.")
        n = len(raw.x)
        i, j = int(n*a), int(n*(a+b))
        return (self.subset(raw, 0, i), self.subset(raw, i+self.guard, j),
                self.subset(raw, j+self.guard, n))

    def apply_guard_intervals(self, raw: RawIQData) -> RawIQData:
        # Conservador también para registros separados: nunca heredar estado.
        return self.subset(raw, self.guard, len(raw.x)-self.guard)

    def prepare(self, training: RawIQData, validation: RawIQData, test: RawIQData,
                duplicate_policy: str = "error", train_fraction: float = .75) -> AlignedInputData:
        raws = [training, validation, test]
        for raw in raws:
            self.validate(raw)
        periods = [float(np.median(np.diff(r.time))) for r in raws]
        if not np.allclose(periods, periods[0], rtol=1e-5, atol=1e-18):
            raise ValueError("Frecuencias de muestreo incompatibles entre particiones.")
        for key in ("carrier_frequency_hz", "modulation"):
            values = [r.metadata.get("signal_config", {}).get(key) for r in raws]
            if any(v != values[0] for v in values[1:]):
                raise ValueError(f"Metadatos incompatibles: {key}.")
        def identical(a: RawIQData, b: RawIQData) -> bool:
            return np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
        if identical(training, validation) or identical(validation, test):
            raise ValueError("Validation duplica otra partición; se necesitan datos independientes.")
        provenance = {"sources": [{"role": role, "path": r.source, "sha256": r.sha256, "metadata": r.metadata}
                                  for role, r in zip(("training", "validation", "test"), raws)], "sample_period_s": periods[0],
                      "guard_samples_each_edge": self.guard, "common_past": self.past,
                      "common_future": self.future, "alignment": "preserved_from_aligned_waveform",
                      "split_mode": "provided_partitions_in_original_sample_order",
                      "warnings": ["Los tiempos de simulaciones separadas no establecen cronología global.",
                                   "La independencia estadística de señales PN9 no se demuestra con hashes distintos."]}
        if identical(training, test):
            if duplicate_policy != "split_training":
                raise ValueError("training y test son idénticos. Proporcione test independiente o configure duplicate_policy=split_training.")
            if not 0 < train_fraction < 1:
                raise ValueError("train_fraction debe estar entre 0 y 1.")
            boundary = int(len(training.x) * train_fraction)
            test = self.subset(training, boundary + self.guard, len(training.x))
            training = self.subset(training, 0, boundary)
            provenance["split_mode"] = "duplicate_test_replaced_by_chronological_training_holdout"
            provenance["holdout_boundary"] = boundary
            provenance["warnings"].append("Test es un holdout del registro training, no una nueva simulación.")
        guarded = [self.apply_guard_intervals(r) for r in (training, validation, test)]
        if any(len(r.x) <= self.past + self.future + 2 for r in guarded):
            raise ValueError("Pocas muestras después de guardas y contexto.")
        gain = self.estimate_gain(guarded[0].x, guarded[0].y)
        parts = [DataPartition(r.x, self.normalize_output(r.y, gain), r.time,
                               np.arange(self.past, len(r.x)-self.future)) for r in guarded]
        provenance["samples"] = {name: {"retained": len(p.x), "scored": len(p.indices),
                                        "start_s": float(p.time[0]), "end_s": float(p.time[-1])}
                                 for name, p in zip(("training", "validation", "test"), parts)}
        return AlignedInputData(*parts, gain, provenance)
