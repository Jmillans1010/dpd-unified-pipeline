from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .basis import BasisTerm, PruningRule, polynomial_terms, gmp_terms, volterra_terms
from .contracts import DataPartition, FitDiagnostics, ModelConfig, TrainedModel, Array


class Trainer(ABC):
    name: str

    def __init__(self, condition_limit: float = 1e10, rcond: float = 1e-12):
        if not np.isfinite(condition_limit) or condition_limit < 1 or not 0 < rcond < 1:
            raise ValueError("Límites numéricos inválidos.")
        self.condition_limit, self.rcond = condition_limit, rcond

    @abstractmethod
    def build_regressors(self, data: DataPartition, config: ModelConfig) -> np.ndarray: ...

    @abstractmethod
    def fit(self, train_data: DataPartition, config: ModelConfig) -> TrainedModel: ...

    @abstractmethod
    def predict(self, data: DataPartition, model: TrainedModel) -> Array: ...

    @abstractmethod
    def count_basis_terms(self, config: ModelConfig) -> int: ...


class PolynomialTrainer(Trainer):
    @abstractmethod
    def terms(self, config: ModelConfig) -> list[BasisTerm]: ...

    def count_basis_terms(self, config: ModelConfig) -> int:
        return len(self.terms(config))

    def build_regressors(self, data: DataPartition, config: ModelConfig) -> np.ndarray:
        return self._build_terms(data, self.terms(config))

    @staticmethod
    def _build_terms(data: DataPartition, terms: list[BasisTerm]) -> np.ndarray:
        delays = [d for t in terms for d in t.delays()]
        indices, z = data.indices, data.z
        if not len(indices) or indices[0]-max(delays) < 0 or indices[-1]-min(delays) >= len(z):
            raise ValueError("Contexto de memoria insuficiente; no se permite wrap-around.")
        columns = []
        for t in terms:
            if t.kind == "envelope":
                value = z[indices-t.signal_delay].copy()
                if t.envelope_power:
                    value *= np.abs(z[indices-t.envelope_delay]) ** t.envelope_power
            else:
                value = np.ones(len(indices), dtype=complex)
                for delay in t.direct_delays:
                    value *= z[indices-delay]
                for delay in t.conjugate_delays:
                    value *= np.conj(z[indices-delay])
            columns.append(value)
        return np.column_stack(columns)

    def fit(self, train_data: DataPartition, config: ModelConfig) -> TrainedModel:
        phi = self.build_regressors(train_data, config)
        if not np.all(np.isfinite(phi)):
            raise FloatingPointError("Regresores no finitos.")
        # Equilibrado de columnas y SVD; nunca se forman ecuaciones normales.
        scales = np.linalg.norm(phi, axis=0)
        safe_scales = np.where(scales > 0, scales, 1)
        coefficients, _, rank, scaled_s = np.linalg.lstsq(phi / safe_scales, train_data.target, rcond=self.rcond)
        raw_s = np.linalg.svd(phi, compute_uv=False)
        condition = float(raw_s[0] / raw_s[-1]) if raw_s[-1] > 0 else float("inf")
        scaled_condition = float(scaled_s[0] / scaled_s[-1]) if scaled_s[-1] > 0 else float("inf")
        n = phi.shape[1]
        stable = bool(rank == n and scaled_condition <= self.condition_limit and np.all(scales > 0))
        diagnostics = FitDiagnostics(int(rank), raw_s.tolist(), condition, n, scaled_condition, stable,
                                     "" if stable else "rank_deficient_or_scaled_condition_limit")
        return TrainedModel(self.name, config, coefficients / safe_scales, diagnostics,
                            basis_terms=[t.to_dict() for t in self.terms(config)])

    def predict(self, data: DataPartition, model: TrainedModel) -> Array:
        # Persist the retained basis, including custom pruning decisions.
        terms = [BasisTerm(**term) for term in model.basis_terms]
        return self._build_terms(data, terms) @ model.coefficients


class MPTrainer(PolynomialTrainer):
    name = "MP"
    def terms(self, config: ModelConfig) -> list[BasisTerm]:
        return polynomial_terms(config)


class EMPTrainer(PolynomialTrainer):
    name = "EMP"
    def terms(self, config: ModelConfig) -> list[BasisTerm]:
        return polynomial_terms(config, envelope=True)


class GMPTrainer(PolynomialTrainer):
    name = "GMP"
    def terms(self, config: ModelConfig) -> list[BasisTerm]:
        return gmp_terms(config)


class PrunedVolterraTrainer(PolynomialTrainer):
    name = "PrunedVolterra"
    def __init__(self, condition_limit: float = 1e10, rcond: float = 1e-12,
                 pruning_rule: PruningRule | None = None):
        super().__init__(condition_limit, rcond)
        self.pruning_rule = pruning_rule

    def terms(self, config: ModelConfig) -> list[BasisTerm]:
        return volterra_terms(config, self.pruning_rule)


class LUTTrainer(Trainer):
    name = "LUT"

    @staticmethod
    def grid(amplitudes: np.ndarray, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
        bins = config["bins"]
        maximum = max(float(np.max(amplitudes)), np.finfo(float).eps)
        if config["binning"] == "uniform_amplitude":
            edges = np.linspace(0, maximum, bins+1)
        elif config["binning"] == "uniform_power":
            edges = np.sqrt(np.linspace(0, maximum**2, bins+1))
        elif config["binning"] == "quantile":
            edges = np.quantile(amplitudes, np.linspace(0, 1, bins+1))
            edges[0] = 0
            # Repeated amplitudes: blend by a tiny uniform amount to keep nodes strict.
            edges = (1-1e-10)*edges + 1e-10*np.linspace(0, maximum, bins+1)
        else:
            raise ValueError("Binning LUT desconocido.")
        return edges, (edges[:-1] + edges[1:]) / 2

    @staticmethod
    def indexing(amplitudes: np.ndarray, nodes: np.ndarray, interpolation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if interpolation == "nearest":
            idx = np.searchsorted((nodes[:-1]+nodes[1:])/2, amplitudes)
            return idx, idx, np.zeros_like(amplitudes)
        if interpolation != "linear":
            raise ValueError("Interpolación LUT desconocida.")
        right = np.clip(np.searchsorted(nodes, amplitudes, side="right"), 1, len(nodes)-1)
        left = right-1
        weight = np.clip((amplitudes-nodes[left])/(nodes[right]-nodes[left]), 0, 1)
        return left, right, weight

    def _matrix(self, z: np.ndarray, nodes: np.ndarray, interpolation: str) -> np.ndarray:
        left, right, weight = self.indexing(np.abs(z), nodes, interpolation)
        phi = np.zeros((len(z), len(nodes)), dtype=complex)
        rows = np.arange(len(z))
        phi[rows, left] += z*(1-weight)
        phi[rows, right] += z*weight
        return phi

    def build_regressors(self, data: DataPartition, config: ModelConfig) -> np.ndarray:
        _, nodes = self.grid(np.abs(data.z[data.indices]), config)
        return self._matrix(data.z[data.indices], nodes, config["interpolation"])

    def count_basis_terms(self, config: ModelConfig) -> int:
        return config["bins"]

    def fit(self, train_data: DataPartition, config: ModelConfig) -> TrainedModel:
        z = train_data.z[train_data.indices]
        edges, nodes = self.grid(np.abs(z), config)
        left, right, weight = self.indexing(np.abs(z), nodes, config["interpolation"])
        # Disjoint nearest cells: exact scalar least squares, O(N + bins).
        if config["interpolation"] == "nearest":
            energy = np.bincount(left, weights=np.abs(z)**2, minlength=len(nodes))
            product = np.conj(z)*train_data.target
            rhs = (np.bincount(left, weights=product.real, minlength=len(nodes)) +
                   1j*np.bincount(left, weights=product.imag, minlength=len(nodes)))
            active = energy > max(float(energy.max())*self.rcond**2, np.finfo(float).tiny)
            values = np.zeros(len(nodes), dtype=complex)
            values[active] = rhs[active]/energy[active]
            s = np.sort(np.sqrt(energy))[::-1]
            rank = int(active.sum())
        else:
            phi = self._matrix(z, nodes, "linear")
            values, _, rank, s = np.linalg.lstsq(phi, train_data.target, rcond=self.rcond)
            active = np.linalg.norm(phi, axis=0) > 0
        # Empty cells have no data: deterministic training-only interpolation.
        if np.any(active):
            values[~active] = (np.interp(nodes[~active], nodes[active], values[active].real) +
                               1j*np.interp(nodes[~active], nodes[active], values[active].imag))
        positive = s[s > (s[0]*self.rcond if len(s) else 0)]
        effective_condition = float(positive[0]/positive[-1]) if len(positive) else float("inf")
        condition = float(s[0]/s[-1]) if len(s) and s[-1] > 0 else float("inf")
        # Empty nearest cells are handled explicitly; linear rank deficiency is rejected.
        stable = bool(rank > 0 and effective_condition <= self.condition_limit and
                      (config["interpolation"] == "nearest" or rank == len(nodes)))
        reason = "empty_cells_interpolated_from_training" if not np.all(active) else ""
        if not stable:
            reason = "rank_deficient_or_condition_limit"
        diagnostics = FitDiagnostics(int(rank), s.tolist(), condition, len(nodes), effective_condition, stable, reason)
        return TrainedModel(self.name, config, values, diagnostics, values, nodes, edges)

    def predict(self, data: DataPartition, model: TrainedModel) -> Array:
        z = data.z[data.indices]
        left, right, weight = self.indexing(np.abs(z), model.lut_nodes, model.config["interpolation"])
        return z*((1-weight)*model.lut_values[left] + weight*model.lut_values[right])


TRAINERS = {t.name: t for t in (MPTrainer, EMPTrainer, GMPTrainer, PrunedVolterraTrainer, LUTTrainer)}
