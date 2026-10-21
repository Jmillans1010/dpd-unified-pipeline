import unittest
import numpy as np

from dpd.contracts import AlignedInputData, OptimizationCriteria
from dpd.hardware import DirectHardwareEstimator
from dpd.metrics import bic_penalty_db, bic_score_db
from dpd.optimizer import HyperparameterOptimizer
from dpd.search import MPSearchPolicy, EMPSearchPolicy
from dpd.trainers import MPTrainer
from test_dpd import ScriptedTrainer, partition, random_part


class BICTrainer(ScriptedTrainer):
    def __init__(self, errors, validation_errors=None, invalid=()):
        super().__init__(errors)
        self.validation_errors = validation_errors
        self.invalid = invalid
        self.fitted = []

    def fit(self, data, config):
        self.fitted.append(config["P"])
        model = super().fit(data, config)
        n = (config["P"]+1)//2
        model.diagnostics.rank = model.diagnostics.num_coefficients = n
        model.diagnostics.stable = config["P"] not in self.invalid
        model.diagnostics.reason = "scripted_instability" if not model.diagnostics.stable else ""
        return model

    def predict(self, data, model):
        if self.validation_errors and data is self.validation_data:
            return data.target*(1-np.sqrt(self.validation_errors[model.config["P"]]))
        return super().predict(data, model)


class BICTests(unittest.TestCase):
    def run_search(self, errors, validation_errors=None, invalid=(), **kwargs):
        # Exactly 1000 identification samples, independent of validation size.
        data = AlignedInputData(partition(np.ones(1000)), random_part(seed=2), random_part(seed=3), 1)
        trainer = BICTrainer(errors, validation_errors, invalid)
        trainer.test_data, trainer.validation_data = data.test, data.validation
        optimizer = HyperparameterOptimizer(trainer, MPSearchPolicy(max_P=max(errors), max_M=0),
            DirectHardwareEstimator(), OptimizationCriteria(**kwargs))
        return optimizer.optimize(data), trainer

    def test_paper_example_and_score_equivalence(self):
        self.assertAlmostEqual(bic_penalty_db(5500), .0073, places=4)
        m, n, error = 1000, 4, .01
        self.assertAlmostEqual(bic_score_db(error, n, m), -20+n*10*np.log10(2*m)/m)
        # Paper eq. 6 with unit target energy/sample, converted to dB units.
        raw_bic = 2*m*np.log(error)+2*n*np.log(2*m)
        self.assertAlmostEqual(raw_bic*10/(2*m*np.log(10)), bic_score_db(error, n, m))

    def test_stop_when_increment_below_sample_dependent_threshold(self):
        penalty = bic_penalty_db(1000)
        # First coefficient earns 1 dB, next earns half the BIC threshold.
        result, trainer = self.run_search({1: 1, 3: 10**(-1/10), 5: 10**((-1-penalty/2)/10), 7: .01})
        self.assertEqual(result.best_config["P"], 3)
        self.assertEqual(result.optimization_trace.stop_reason, "bic_minimum_reached")
        self.assertNotIn(7, trainer.fitted)  # no maximum reference fit
        self.assertIsNone(result.progress_fraction)
        self.assertIsNone(result.optimization_trace.target_fraction)
        self.assertEqual(result.optimization_trace.reference_candidates, [])
        self.assertEqual(trainer.test_calls, 1)
        self.assertEqual(result.optimization_trace.identification_samples, 1000)

    def test_training_bic_not_validation_or_percentage(self):
        result, trainer = self.run_search({1: 1, 3: .1, 5: .01}, {1: .01, 3: 1, 5: 10}, target_fraction=.01)
        self.assertEqual(result.best_config["P"], 5)
        self.assertEqual(result.optimization_trace.stop_reason, "search_limit_reached")
        self.assertTrue(result.optimization_trace.boundary_reached)
        self.assertEqual(trainer.test_calls, 1)

    def test_numerical_failure_is_not_bic_plateau(self):
        result, _ = self.run_search({1: 1, 3: .1, 5: .01}, invalid=(3,))
        self.assertEqual(result.optimization_trace.stop_reason, "numerical_instability")
        self.assertEqual(result.best_config["P"], 1)
        self.assertTrue(all(c.progress_fraction is None for c in result.optimization_trace.candidates.values()))

    def test_exact_zero_error_is_finite_and_prefers_fewer_coefficients(self):
        result, _ = self.run_search({1: 0, 3: 0, 5: 0})
        self.assertEqual(result.best_config["P"], 1)
        self.assertTrue(np.isfinite(result.optimization_trace.selected_bic_score_db))

    def test_emp_zero_change_memory_does_not_cause_false_stop(self):
        optimizer = HyperparameterOptimizer(MPTrainer(), EMPSearchPolicy(max_M=3),
            DirectHardwareEstimator(), OptimizationCriteria())
        # Use the actual EMP basis count.
        from dpd.trainers import EMPTrainer
        optimizer.trainer = EMPTrainer()
        candidates = optimizer.bic_neighbors({"P": 1, "M": 0})
        self.assertIn({"P": 3, "M": 3}, candidates)
        self.assertFalse(any(c["P"] == 1 for c in candidates))

    def test_invalid_criterion_inputs(self):
        with self.assertRaises(ValueError):
            bic_penalty_db(0)
        with self.assertRaises(ValueError):
            OptimizationCriteria(stopping_rule="unknown")


if __name__ == "__main__":
    unittest.main()
