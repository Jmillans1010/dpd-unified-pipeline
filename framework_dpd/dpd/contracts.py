from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

ModelConfig = dict[str, Any]
Array = NDArray[np.complex128]


@dataclass
class RawIQData:
    time: NDArray[np.float64]
    x: Array
    y: Array
    metadata: dict = field(default_factory=dict)
    source: str = ""
    sha256: str = ""


@dataclass
class DataPartition:
    x: Array
    z: Array
    time: NDArray[np.float64]
    indices: NDArray[np.int64]

    @property
    def target(self) -> Array:
        return self.x[self.indices]


@dataclass
class AlignedInputData:
    train: DataPartition
    validation: DataPartition
    test: DataPartition
    gain: complex
    provenance: dict = field(default_factory=dict)


@dataclass
class FitDiagnostics:
    rank: int
    singular_values: list[float]
    condition_number: float
    num_coefficients: int
    scaled_condition_number: float
    stable: bool
    reason: str = ""


@dataclass
class TrainedModel:
    model_name: str
    config: ModelConfig
    coefficients: Array
    diagnostics: FitDiagnostics
    lut_values: Array | None = None
    lut_nodes: NDArray[np.float64] | None = None
    lut_edges: NDArray[np.float64] | None = None
    basis_terms: list[dict] = field(default_factory=list)

    def predict(self, data: DataPartition) -> Array:
        # Registry is an integration point, never part of the optimizer.
        from .trainers import TRAINERS
        return TRAINERS[self.model_name]().predict(data, self)


@dataclass
class HardwareComplexity:
    num_parameters: int
    complex_multipliers: int
    real_multipliers: int
    complex_adders: int
    real_adders: int
    delay_elements: int
    num_luts: int
    lut_entries: int
    memory_bits: int
    cycles_per_sample: int
    clock_multiplier: float
    assumptions: list[str] = field(default_factory=list)
    auxiliary_operations: dict[str, int] = field(default_factory=dict)


@dataclass
class CandidateResult:
    candidate_id: str
    model_name: str
    config: ModelConfig
    model: TrainedModel | None
    nmse_train_db: float | None
    nmse_validation_db: float | None
    normalized_error_linear: float | None
    train_error_linear: float | None
    progress_fraction: float | None
    structural_complexity: float
    hardware: HardwareComplexity | None
    marginal_efficiency: float | None = None
    pareto_optimal: bool = False
    valid: bool = True
    failure: str = ""
    bic_score_db: float | None = None
    bic_num_coefficients: int | None = None


@dataclass
class OptimizationCriteria:
    stopping_rule: str = "bic"
    target_fraction: float = 0.99
    min_improvement: float = 1e-5
    max_iterations: int = 60
    plateau_patience: int = 2
    tie_tolerance: float = 1e-5
    complexity_mode: str = "structural"

    def __post_init__(self) -> None:
        if self.stopping_rule not in ("bic", "reference_fraction"):
            raise ValueError("stopping_rule debe ser bic o reference_fraction.")
        if not 0 < self.target_fraction <= 1:
            raise ValueError("target_fraction debe estar en (0, 1].")
        if self.min_improvement < 0 or self.tie_tolerance < 0:
            raise ValueError("Las tolerancias deben ser no negativas.")
        if self.max_iterations < 1 or self.plateau_patience < 1:
            raise ValueError("Iteraciones y paciencia deben ser positivas.")
        if self.complexity_mode != "structural":
            raise ValueError("Solo se admite structural; hardware se conserva separado.")


@dataclass
class OptimizationTrace:
    baseline: str
    reference_candidates: list[str]
    reference_nmse_db: float | None
    reference_error_linear: float | None
    target_fraction: float | None
    candidates: dict[str, CandidateResult] = field(default_factory=dict)
    iterations: list[dict] = field(default_factory=list)
    selected: str | None = None
    stop_reason: str = ""
    notes: list[str] = field(default_factory=list)
    stopping_rule: str = "reference_fraction"
    identification_samples: int | None = None
    bic_penalty_db_per_coefficient: float | None = None
    selected_bic_score_db: float | None = None
    boundary_reached: bool = False


@dataclass
class ModelResult:
    model_name: str
    best_config: ModelConfig | None
    model: TrainedModel | None
    nmse_train_db: float | None
    nmse_validation_db: float | None
    nmse_test_db: float | None
    test_error_linear: float | None
    progress_fraction: float | None
    structural_complexity: float | None
    hardware: HardwareComplexity | None
    optimization_trace: OptimizationTrace
