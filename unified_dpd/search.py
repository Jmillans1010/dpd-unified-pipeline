"""BIC structure search. This module receives training only, never validation."""
from dataclasses import asdict
import json
import numpy as np
from dpd.metrics import normalized_error, error_to_db, bic_score_db, bic_penalty_db
from dpd.optimizer import HyperparameterOptimizer
from dpd.contracts import OptimizationCriteria


class TrainingBICSearch:
    def __init__(self, trainer, policy, estimator, settings):
        self.trainer, self.policy, self.estimator = trainer, policy, estimator
        self.settings = settings
        self.trace = {}
        self.best = None
        self.budget_hit = False
        # Reuse neighbor generation that crosses identical-basis structural steps.
        self.neighbor_builder = HyperparameterOptimizer(trainer, policy, estimator,
                                                       OptimizationCriteria(stopping_rule="bic"))

    @staticmethod
    def key(config):
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def evaluate(self, config, train):
        key = self.key(config)
        if key in self.trace:
            return self.trace[key]
        if len(self.trace) >= self.settings["max_candidates"]:
            self.budget_hit = True
            return None
        terms = self.trainer.count_basis_terms(config)
        # Conservative planning estimate, not a hard allocator/OS memory limit.
        mib = 6 * len(train.indices) * terms * 16 / 2**20
        entry = dict(config=config, terms=terms, estimated_working_mib=mib,
                     valid=False, reason="", bic_score_db=None)
        self.trace[key] = entry
        if terms > self.settings["max_basis_terms"] or mib > self.settings["max_working_mib"]:
            entry["reason"] = "resource_limit"
            return entry
        if terms > len(train.indices):
            entry["reason"] = "more_coefficients_than_training_samples"
            return entry
        try:
            model = self.trainer.fit(train, config)
            entry["diagnostics"] = asdict(model.diagnostics)
            if not model.diagnostics.stable:
                entry["reason"] = model.diagnostics.reason
                return entry
            error = normalized_error(train.target, model.predict(train))
            score = bic_score_db(error, model.diagnostics.rank, len(train.indices))
            entry.update(valid=True, nmse_train_db=error_to_db(error), bic_score_db=score,
                         num_coefficients=model.diagnostics.rank)
            if self.best is None or score < self.best[0] - 1e-10:
                self.best = (score, model, key)
        except (np.linalg.LinAlgError, FloatingPointError, MemoryError) as exc:
            entry["reason"] = type(exc).__name__ + ": " + str(exc)
        return entry

    def run(self, train):
        current = self.policy.minimal_config()
        initial = self.evaluate(current, train)
        if not initial or not initial["valid"]:
            raise ValueError(f"{self.trainer.name}: baseline unavailable: {initial}")
        iterations, stop = [], "max_iterations"
        expanded = set()
        for step in range(self.settings["max_iterations"]):
            if not self.neighbor_builder.bic_neighbors(current):
                stop = "search_limit_reached"
                break
            previous_best = self.best[0]
            frontier = [current]
            improved = False
            for depth in range(1 + self.settings["plateau_extra_depth"]):
                next_layer = {}
                for origin in frontier:
                    origin_key = self.key(origin)
                    if origin_key in expanded:
                        continue
                    expanded.add(origin_key)
                    for config in self.neighbor_builder.bic_neighbors(origin):
                        entry = self.evaluate(config, train)
                        if entry and entry["valid"]:
                            next_layer[self.key(config)] = entry
                iterations.append(dict(step=step, extra_depth=depth, evaluated=list(next_layer),
                                       best_bic=self.best[0]))
                if self.best[0] < previous_best - 1e-10:
                    current = self.best[1].config
                    improved = True
                    break
                if self.budget_hit or not next_layer:
                    break
                frontier = [e["config"] for e in sorted(next_layer.values(), key=lambda e: e["bic_score_db"])
                            if self.key(e["config"]) not in expanded][:self.settings["beam_width"]]
                if not frontier:
                    break
            if self.budget_hit:
                stop = "candidate_budget"
                break
            if not improved:
                stop = "bounded_bic_frontier_incomplete" if any(not e["valid"] for e in self.trace.values()) else "bounded_bic_search_exhausted"
                break
        score, model, key = self.best
        blocked = [e for e in self.trace.values() if not e["valid"]]
        entry = self.trace[key]
        return dict(model_name=self.trainer.name, best_config=model.config, model=model,
                    nmse_train_db=entry["nmse_train_db"], hardware=self.estimator.estimate(model, model.config),
                    optimization_trace=dict(stopping_rule="training_bic", selected=key, stop_reason=stop,
                        best_bic_score_db=score, identification_samples=len(train.indices),
                        penalty_db_per_coefficient=bic_penalty_db(len(train.indices)),
                        settings=self.settings, candidates=self.trace, iterations=iterations,
                        incomplete_frontier=bool(blocked), rejected_candidates=len(blocked),
                        scope="bounded local search; no global optimum guarantee; validation not accessed"))
