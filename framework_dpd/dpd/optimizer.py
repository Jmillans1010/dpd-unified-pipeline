"""Búsqueda común: no contiene nombres ni ecuaciones de arquitecturas."""
from __future__ import annotations
import json
import numpy as np
from .contracts import (AlignedInputData, CandidateResult, ModelResult, OptimizationCriteria,
                        OptimizationTrace, ModelConfig)
from .metrics import normalized_error, error_to_db, progress_fraction, bic_penalty_db, bic_score_db
from .trainers import Trainer
from .search import SearchPolicy
from .hardware import HardwareComplexityEstimator


class HyperparameterOptimizer:
    def __init__(self, trainer: Trainer, policy: SearchPolicy,
                 estimator: HardwareComplexityEstimator, criteria: OptimizationCriteria):
        self.trainer, self.policy, self.estimator, self.criteria = trainer, policy, estimator, criteria
        self.candidates: dict[str, CandidateResult] = {}
        self.baseline_error = self.reference_error = 0.0

    @staticmethod
    def key(config: ModelConfig) -> str:
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def evaluate_candidate(self, config: ModelConfig, data: AlignedInputData) -> CandidateResult:
        key = self.key(config)
        if key in self.candidates:
            return self.candidates[key]
        result = CandidateResult(key, self.trainer.name, config, None, None, None, None, None,
                                 None if self.criteria.stopping_rule == "bic" else 0.0,
                                 self.policy.structural_complexity(config), None)
        self.candidates[key] = result
        try:
            result.model = self.trainer.fit(data.train, config)
            result.hardware = self.estimator.estimate(result.model, config)
            if not result.model.diagnostics.stable:
                result.valid = False
                result.failure = result.model.diagnostics.reason
                return result
            result.train_error_linear = normalized_error(data.train.target, self.trainer.predict(data.train, result.model))
            result.normalized_error_linear = normalized_error(data.validation.target, self.trainer.predict(data.validation, result.model))
            result.nmse_train_db = error_to_db(result.train_error_linear)
            result.nmse_validation_db = error_to_db(result.normalized_error_linear)
            if self.criteria.stopping_rule == "bic":
                result.bic_num_coefficients = result.model.diagnostics.rank
                result.bic_score_db = bic_score_db(result.train_error_linear, result.bic_num_coefficients,
                                                  len(data.train.indices))
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            result.valid, result.failure = False, str(exc)
        return result

    def calculate_progress(self, candidate: CandidateResult) -> float:
        return progress_fraction(self.baseline_error, candidate.normalized_error_linear, self.reference_error) if candidate.valid else 0.0

    def estimate_reference_performance(self, data: AlignedInputData) -> list[CandidateResult]:
        return [self.evaluate_candidate(c, data) for c in self.policy.reference_configs() if self.policy.is_valid(c)]

    def calculate_marginal_efficiency(self, current: CandidateResult, candidate: CandidateResult) -> float | None:
        delta = candidate.structural_complexity-current.structural_complexity
        if delta < 0:
            raise ValueError("La política redujo complejidad durante crecimiento.")
        return (candidate.progress_fraction-current.progress_fraction)/delta if delta else None

    @staticmethod
    def hardware_key(candidate: CandidateResult) -> tuple:
        h = candidate.hardware
        return (4*h.complex_multipliers+h.real_multipliers, h.memory_bits,
                2*h.complex_adders+h.real_adders, candidate.candidate_id)

    def select_next_candidate(self, current: CandidateResult, candidates: list[CandidateResult]) -> tuple[CandidateResult, str]:
        tolerance = self.criteria.tie_tolerance
        # Equal-cost alternatives first collapse to the lowest validation error.
        groups: dict[float, list[CandidateResult]] = {}
        for c in candidates:
            groups.setdefault(c.structural_complexity, []).append(c)
        representatives = []
        for group in groups.values():
            best_error = min(c.normalized_error_linear for c in group)
            equivalent = [c for c in group if c.normalized_error_linear <= best_error+tolerance*self.baseline_error]
            representatives.append(min(equivalent, key=self.hardware_key))
        free_improvements = [c for c in representatives if c.structural_complexity == current.structural_complexity and
                             c.normalized_error_linear < current.normalized_error_linear]
        if free_improvements:
            return min(free_improvements, key=lambda c: (c.normalized_error_linear, self.hardware_key(c))), "lower_validation_error_at_equal_complexity"
        growing = [c for c in representatives if c.structural_complexity > current.structural_complexity]
        pool = growing or representatives
        best_efficiency = max(c.marginal_efficiency or 0.0 for c in pool)
        efficient = [c for c in pool if (c.marginal_efficiency or 0.0) >= best_efficiency-tolerance]
        lowest_error = min(c.normalized_error_linear for c in efficient)
        equivalent = [c for c in efficient if c.normalized_error_linear <= lowest_error+tolerance*self.baseline_error]
        return min(equivalent, key=self.hardware_key), "marginal_progress_per_structural_cost_then_hardware"

    def calculate_pareto_front(self) -> None:
        valid = [c for c in self.candidates.values() if c.valid]
        for a in valid:
            a.pareto_optimal = not any(b.structural_complexity <= a.structural_complexity and
                                     b.normalized_error_linear <= a.normalized_error_linear and
                                     (b.structural_complexity < a.structural_complexity or
                                      b.normalized_error_linear < a.normalized_error_linear) for b in valid)

    def optimize(self, data: AlignedInputData) -> ModelResult:
        if self.criteria.stopping_rule == "bic":
            return self.optimize_bic(data)
        self.candidates = {}
        baseline = self.evaluate_candidate(self.policy.minimal_config(), data)
        trace = OptimizationTrace(baseline.candidate_id, [], None, None, self.criteria.target_fraction,
                                  self.candidates)
        if not baseline.valid:
            trace.stop_reason = "numerical_instability"
            return ModelResult(self.trainer.name, None, None, None, None, None, None, 0, None, None, trace)
        current = best = baseline
        self.baseline_error = baseline.normalized_error_linear
        references = self.estimate_reference_performance(data)
        trace.reference_candidates = [c.candidate_id for c in references]
        valid_references = [c for c in references if c.valid]
        if not valid_references:
            trace.stop_reason = "numerical_instability"
            trace.notes.append("No hay referencias numéricamente válidas; se conserva baseline.")
            self.reference_error = self.baseline_error
        else:
            self.reference_error = min(c.normalized_error_linear for c in valid_references)
            trace.reference_error_linear = self.reference_error
            trace.reference_nmse_db = error_to_db(self.reference_error)
        for c in self.candidates.values():
            c.progress_fraction = self.calculate_progress(c)
        if not trace.stop_reason and self.baseline_error-self.reference_error <= np.finfo(float).eps*max(self.baseline_error, np.finfo(float).tiny):
            trace.stop_reason = "performance_plateau"
            trace.notes.append("La referencia no mejora baseline; fracción de mejora indefinida, se informa 0.")
        visited = {baseline.candidate_id}
        stale = 0
        for iteration in range(self.criteria.max_iterations):
            if trace.stop_reason:
                break
            configs = [c for c in self.policy.generate_neighbors(current.config)
                       if self.policy.is_valid(c) and self.key(c) not in visited]
            if not configs:
                trace.stop_reason = "no_valid_neighbors"
                break
            neighbors = [self.evaluate_candidate(c, data) for c in configs]
            valid = [c for c in neighbors if c.valid]
            entry = {"iteration": iteration+1, "current": current.candidate_id,
                     "candidates": [c.candidate_id for c in neighbors], "chosen": None,
                     "reason": "all_candidates_numerically_invalid", "changes": []}
            trace.iterations.append(entry)
            if not valid:
                trace.stop_reason = "numerical_instability"
                break
            for c in valid:
                c.progress_fraction = self.calculate_progress(c)
                c.marginal_efficiency = self.calculate_marginal_efficiency(current, c)
                entry["changes"].append({"candidate": c.candidate_id,
                                         "delta_progress": c.progress_fraction-current.progress_fraction,
                                         "delta_complexity": c.structural_complexity-current.structural_complexity,
                                         "marginal_efficiency": c.marginal_efficiency})
            chosen, reason = self.select_next_candidate(current, valid)
            entry.update(chosen=chosen.candidate_id, reason=reason, progress_fraction=chosen.progress_fraction)
            improvement = (current.normalized_error_linear-chosen.normalized_error_linear)/max(self.baseline_error, np.finfo(float).tiny)
            stale = stale+1 if improvement <= self.criteria.min_improvement else 0
            current = chosen
            visited.add(chosen.candidate_id)
            if chosen.normalized_error_linear < best.normalized_error_linear:
                best = chosen
            if chosen.progress_fraction >= self.criteria.target_fraction:
                best = chosen
                trace.stop_reason = "target_reached"
            elif stale >= self.criteria.plateau_patience:
                trace.stop_reason = "performance_plateau"
        if not trace.stop_reason:
            trace.stop_reason = "max_iterations"
        trace.selected = best.candidate_id
        self.calculate_pareto_front()
        # Exactly one test evaluation, after selection. Test cannot reach search logic.
        test_error = normalized_error(data.test.target, self.trainer.predict(data.test, best.model))
        return ModelResult(self.trainer.name, best.config, best.model, best.nmse_train_db,
                           best.nmse_validation_db, error_to_db(test_error), test_error,
                           best.progress_fraction, best.structural_complexity, best.hardware, trace)

    def bic_neighbors(self, config: ModelConfig) -> list[ModelConfig]:
        # EMP P=1 has identical bases at every M. Likewise Volterra D=0,M=0
        # cannot grow D yet. Cross zero-basis-change steps without interpreting
        # them as evidence of a BIC plateau; still fit every actual new basis.
        count = self.trainer.count_basis_terms(config)
        queue = list(self.policy.generate_neighbors(config))
        seen = {self.key(config)}
        result = []
        while queue:
            neighbor = queue.pop(0)
            key = self.key(neighbor)
            if key in seen or not self.policy.is_valid(neighbor):
                continue
            seen.add(key)
            if self.trainer.count_basis_terms(neighbor) == count and hasattr(self.trainer, "terms"):
                queue.extend(self.policy.generate_neighbors(neighbor))
            else:
                result.append(neighbor)
        return result

    def optimize_bic(self, data: AlignedInputData) -> ModelResult:
        """Hill-climbing with Becerra's BIC stopping rule, not a global optimum."""
        self.candidates = {}
        current = self.evaluate_candidate(self.policy.minimal_config(), data)
        samples = len(data.train.indices)
        trace = OptimizationTrace(current.candidate_id, [], None, None, None, self.candidates,
                                  stopping_rule="bic", identification_samples=samples,
                                  bic_penalty_db_per_coefficient=bic_penalty_db(samples))
        trace.notes.extend([
            "Becerra et al. 2020, DOI 10.1109/LMWC.2020.3027878, eqs. 6,9,13.",
            "BIC uses identification/training RSS and independent fitted complex coefficients; validation is diagnostic.",
            "Greedy structure search: a local stopping decision, not proof of global optimality.",
            "The RSS form assumes zero-mean residuals; correlated/periodic IQ samples may affect BIC interpretation."])
        if self.trainer.name == "LUT":
            trace.notes.append("LUT extension conditional on its training-derived grid: n=rank of fitted gains; grid estimation is not penalized.")
        if not current.valid:
            trace.stop_reason = "numerical_instability"
            return ModelResult(self.trainer.name, None, None, None, None, None, None, None, None, None, trace)
        visited = {current.candidate_id}
        for iteration in range(self.criteria.max_iterations):
            configs = [c for c in self.bic_neighbors(current.config) if self.key(c) not in visited]
            if not configs:
                trace.stop_reason = "search_limit_reached"
                trace.boundary_reached = True
                break
            neighbors = [self.evaluate_candidate(c, data) for c in configs]
            valid = [c for c in neighbors if c.valid]
            entry = {"iteration": iteration+1, "current": current.candidate_id,
                     "candidates": [c.candidate_id for c in neighbors], "chosen": None,
                     "reason": "", "changes": []}
            trace.iterations.append(entry)
            if not valid:
                trace.stop_reason = "numerical_instability"
                entry["reason"] = "all_candidates_numerically_invalid"
                break
            for c in valid:
                delta_n = c.bic_num_coefficients-current.bic_num_coefficients
                gain = (c.bic_score_db-current.bic_score_db) - delta_n*trace.bic_penalty_db_per_coefficient
                entry["changes"].append({"candidate": c.candidate_id,
                    "delta_coefficients": delta_n, "delta_nmse_train_db": gain,
                    "required_improvement_db": delta_n*trace.bic_penalty_db_per_coefficient,
                    "delta_bic_score_db": c.bic_score_db-current.bic_score_db,
                    "bic_improves": c.bic_score_db < current.bic_score_db-1e-10})
            chosen = min(valid, key=lambda c: (c.bic_score_db, self.hardware_key(c)))
            if chosen.bic_score_db >= current.bic_score_db-1e-10:
                # Rejected numerical candidates leave part of the frontier unknown.
                trace.stop_reason = "numerical_frontier_incomplete" if len(valid) < len(neighbors) else "bic_minimum_reached"
                entry["reason"] = trace.stop_reason
                break
            entry.update(chosen=chosen.candidate_id, reason="lower_identification_bic")
            current = chosen
            visited.add(current.candidate_id)
        if not trace.stop_reason:
            trace.stop_reason = "max_iterations"
        trace.selected = current.candidate_id
        trace.selected_bic_score_db = current.bic_score_db
        self.calculate_pareto_front()
        test_error = normalized_error(data.test.target, self.trainer.predict(data.test, current.model))
        return ModelResult(self.trainer.name, current.config, current.model, current.nmse_train_db,
                           current.nmse_validation_db, error_to_db(test_error), test_error,
                           None, current.structural_complexity, current.hardware, trace)
