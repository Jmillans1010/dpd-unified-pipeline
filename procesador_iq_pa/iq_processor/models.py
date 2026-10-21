"""Contratos de configuración y datos. Los vectores usan unidades SI."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import json
import math
import numpy as np


class ProcessingError(ValueError):
    """Error de datos/configuración que impide una etapa."""


@dataclass
class ProcessingConfig:
    input_files: dict = field(default_factory=dict)
    netlist: str | None = None
    previous_metadata: str | None = None
    output_directory: str = "data/output"
    manual_metadata: dict = field(default_factory=dict)
    samples_per_symbol: int | None = None
    samples_per_symbol_verified: bool = False
    source_data_rate_is_bit_rate: bool = False
    additional_initial_crop_s: float = 0.0
    remove_dc: bool = False
    time_atol_s: float = 1e-18
    time_rtol: float = 1e-5
    max_delay_samples: int = 128
    minimum_overlap_fraction: float = 0.8
    fractional_refinement: bool = False
    interpolation_half_width: int = 16
    minimum_symbols: int = 16
    symbol_phase: int | None = None
    phase_tie_tolerance_pp: float = 1e-10
    plots: bool = True

    @classmethod
    def read(cls, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ProcessingError("El archivo de configuración debe contener un objeto JSON.")
        allowed = {f.name for f in fields(cls)}
        if set(raw) - allowed:
            raise ProcessingError(f"Opciones desconocidas: {sorted(set(raw) - allowed)}")
        result = cls(**raw)
        result.validate()
        return result

    def validate(self):
        if not isinstance(self.input_files, dict) or set(self.input_files) != {"I_in", "Q_in", "I_out", "Q_out"}:
            raise ProcessingError("input_files debe definir I_in, Q_in, I_out y Q_out.")
        for name in ("samples_per_symbol", "max_delay_samples", "interpolation_half_width",
                     "minimum_symbols", "symbol_phase"):
            value = getattr(self, name)
            if value is None and name in ("samples_per_symbol", "symbol_phase"):
                continue
            minimum = 0 if name in ("symbol_phase", "max_delay_samples") else 1
            if type(value) is not int or value < minimum:
                raise ProcessingError(f"{name} debe ser un entero >= {minimum}.")
        if self.interpolation_half_width < 2:
            raise ProcessingError("interpolation_half_width debe ser >= 2.")
        for name in ("additional_initial_crop_s", "time_atol_s", "time_rtol", "phase_tie_tolerance_pp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ProcessingError(f"{name} debe ser finito y no negativo.")
        if (isinstance(self.minimum_overlap_fraction, bool)
                or not isinstance(self.minimum_overlap_fraction, (int, float))
                or not 0 < self.minimum_overlap_fraction <= 1):
            raise ProcessingError("minimum_overlap_fraction debe estar en (0, 1].")
        for name in ("remove_dc", "fractional_refinement", "plots", "samples_per_symbol_verified",
                     "source_data_rate_is_bit_rate"):
            if type(getattr(self, name)) is not bool:
                raise ProcessingError(f"{name} debe ser booleano.")
        if not isinstance(self.manual_metadata, dict):
            raise ProcessingError("manual_metadata debe ser un objeto JSON.")
        for value in [*self.input_files.values(), self.netlist, self.previous_metadata, self.output_directory]:
            if value is not None and (not isinstance(value, str) or not value or Path(value).is_absolute()):
                raise ProcessingError("Las rutas configuradas deben ser cadenas relativas al archivo de configuración.")


@dataclass
class IQSignal:
    time: np.ndarray
    values: np.ndarray
    original_indices: np.ndarray


@dataclass
class InputData:
    input: IQSignal
    output: IQSignal
    sources: list[dict] = field(default_factory=list)


@dataclass
class InputMetadata:
    simulation_config: dict = field(default_factory=dict)
    signal_config: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    input_indices: np.ndarray
    output_positions: np.ndarray
    output_times: np.ndarray
    integer_delay: int
    fractional_delay: float
    delay_s: float
    diagnostics: dict
    lag_table: np.ndarray


@dataclass
class EVMResult:
    alpha: complex
    reference: np.ndarray
    error: np.ndarray
    rms_percent: float
    rms_db: float | None


@dataclass
class SymbolSelectionResult:
    phase: int
    indices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    evm: EVMResult
    candidates: list[dict]


@dataclass
class ProcessingResult:
    alignment: AlignmentResult
    selection: SymbolSelectionResult | None
    metadata: dict
