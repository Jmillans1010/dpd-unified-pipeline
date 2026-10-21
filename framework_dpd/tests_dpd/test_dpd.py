import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dpd.basis import generate_volterra_terms, volterra_terms, PruningRule
from dpd.contracts import (RawIQData, DataPartition, AlignedInputData, OptimizationCriteria,
                           FitDiagnostics, TrainedModel)
from dpd.data import DataProcessor, CSVReader
from dpd.hardware import DirectHardwareEstimator, LUTAssistedHardwareEstimator, MultiplierSharingEstimator
from dpd.metrics import normalized_error, nmse, db_to_error, progress_fraction
from dpd.optimizer import HyperparameterOptimizer
from dpd.pipeline import TrainingPipeline
from dpd.search import (MPSearchPolicy, EMPSearchPolicy, GMPSearchPolicy,
                       PrunedVolterraSearchPolicy, LUTSearchPolicy)
from dpd.serialization import JSONWriter
from dpd.trainers import MPTrainer, EMPTrainer, GMPTrainer, PrunedVolterraTrainer, LUTTrainer


def partition(z, x=None, past=0, future=0):
    z = np.asarray(z, dtype=complex)
    return DataPartition(z.copy() if x is None else np.asarray(x, dtype=complex), z,
                         np.arange(len(z), dtype=float), np.arange(past, len(z)-future))


def random_part(n=200, past=5, future=2, seed=7):
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=n) + 1j*rng.normal(size=n))*.3
    return partition(z, z*(1+.2*np.abs(z)**2), past, future)


class BasisTests(unittest.TestCase):
    def setUp(self):
        self.data = partition([1+1j, 2-1j, -.5+2j, .2-.8j, 1.2+.3j], past=2, future=1)

    def test_mp_values_and_count(self):
        trainer, c = MPTrainer(), {"P": 3, "M": 1}
        z, i = self.data.z, self.data.indices
        expected = np.column_stack([z[i], z[i-1], z[i]*abs(z[i])**2, z[i-1]*abs(z[i-1])**2])
        np.testing.assert_allclose(trainer.build_regressors(self.data, c), expected)
        self.assertEqual(trainer.count_basis_terms(c), 4)

    def test_emp_no_duplicated_linear_terms(self):
        z, i = self.data.z, self.data.indices
        expected = np.column_stack([z[i], z[i]*abs(z[i])**2, z[i]*abs(z[i-1])**2])
        c = {"P": 3, "M": 1}
        np.testing.assert_allclose(EMPTrainer().build_regressors(self.data, c), expected)
        self.assertEqual(EMPTrainer().count_basis_terms({"P": 1, "M": 5}), 1)

    def test_gmp_aligned_lagging_leading(self):
        z, i = self.data.z, self.data.indices
        c = {"aligned": {"P": 1, "M": 0}, "lagging": {"P": 3, "M": 1, "Q": 1},
             "leading": {"P": 3, "M": 1, "Q": 1}}
        expected = np.column_stack([z[i], z[i]*abs(z[i-1])**2, z[i-1]*abs(z[i-2])**2,
                                    z[i]*abs(z[i+1])**2, z[i-1]*abs(z[i])**2])
        np.testing.assert_allclose(GMPTrainer().build_regressors(self.data, c), expected)

    def test_missing_context_rejected(self):
        with self.assertRaises(ValueError):
            MPTrainer().build_regressors(partition([1, 2, 3]), {"P": 1, "M": 1})

    def test_volterra_generation_pruning(self):
        c = {"P": 3, "M": 1, "D": 0}
        self.assertEqual(len(list(generate_volterra_terms(c))), 8)
        terms = volterra_terms(c)
        self.assertEqual(len(terms), 4)
        self.assertEqual(len(volterra_terms({**c, "D": 1})), 8)
        actual = PrunedVolterraTrainer().build_regressors(self.data, c)
        z, i = self.data.z, self.data.indices
        expected = np.column_stack([z[i], z[i-1], z[i]**2*z[i].conj(), z[i-1]**2*z[i-1].conj()])
        np.testing.assert_allclose(actual, expected)

    def test_polynomial_fit_recovers_coefficients(self):
        data = random_part(n=400)
        trainer, c = MPTrainer(), {"P": 3, "M": 1}
        coefficients = np.array([1+.1j, .2-.1j, .4+.03j, -.1+.01j])
        data.x[data.indices] = trainer.build_regressors(data, c) @ coefficients
        model = trainer.fit(data, c)
        self.assertTrue(model.diagnostics.stable)
        np.testing.assert_allclose(model.coefficients, coefficients, atol=1e-12)

    def test_condition_protection(self):
        model = MPTrainer().fit(partition(np.ones(50), past=2), {"P": 3, "M": 1})
        self.assertFalse(model.diagnostics.stable)
        self.assertLess(model.diagnostics.rank, model.diagnostics.num_coefficients)
        self.assertEqual(len(model.diagnostics.singular_values), 4)


class LUTTests(unittest.TestCase):
    def test_indexing_saturates_and_interpolates(self):
        amplitudes, nodes = np.array([0, 1, 1.5, 2.5, 4]), np.array([1., 2., 3.])
        left, right, w = LUTTrainer.indexing(amplitudes, nodes, "linear")
        np.testing.assert_allclose((1-w)*nodes[left]+w*nodes[right], [1, 1, 1.5, 2.5, 3])
        idx, _, _ = LUTTrainer.indexing(amplitudes, nodes, "nearest")
        np.testing.assert_array_equal(idx, [0, 0, 0, 1, 2])

    def test_gain_fit_and_training_grid_reuse(self):
        train = random_part(n=500)
        train.x = train.z*(.8+.3j)
        for method in LUTSearchPolicy.methods:
            for interp in ("nearest", "linear"):
                c = {"bins": 8, "binning": method, "interpolation": interp}
                trainer = LUTTrainer()
                model = trainer.fit(train, c)
                self.assertTrue(model.diagnostics.stable)
                test = partition([0, .1j, 3+1j, 10j])
                before = model.lut_nodes.copy()
                np.testing.assert_allclose(trainer.predict(test, model), test.z*(.8+.3j), atol=1e-12)
                np.testing.assert_array_equal(before, model.lut_nodes)

    def test_empty_cells_and_zero(self):
        train = partition([0, 1j, -1, 1, -1j])
        model = LUTTrainer().fit(train, {"bins": 8, "binning": "uniform_amplitude", "interpolation": "nearest"})
        self.assertTrue(model.diagnostics.stable)
        self.assertLess(model.diagnostics.rank, 8)
        np.testing.assert_allclose(model.predict(train), train.target)


class MetricTests(unittest.TestCase):
    def test_nmse_and_linear_conversion(self):
        target = np.array([1+1j, -2j])
        self.assertAlmostEqual(normalized_error(target, target*.9), .01)
        self.assertAlmostEqual(nmse(target, target*.9), -20)
        self.assertAlmostEqual(db_to_error(-20), .01)
        self.assertEqual(nmse(target, target), -np.inf)

    def test_progress_is_linear_error_not_db(self):
        self.assertAlmostEqual(progress_fraction(1, .28, .1), .8)
        self.assertGreater(progress_fraction(1, 0, .1), 1)
        self.assertEqual(progress_fraction(1, 1, 1), 0)
        self.assertEqual(progress_fraction(1, 1, 2), 0)

    def test_bad_target(self):
        with self.assertRaises(ValueError):
            nmse(np.zeros(5), np.ones(5))
        with self.assertRaises(ValueError):
            nmse(np.ones(5), np.ones(4))


class SearchTests(unittest.TestCase):
    def test_one_dimension_neighbors(self):
        policies = [MPSearchPolicy(), EMPSearchPolicy(), PrunedVolterraSearchPolicy(), LUTSearchPolicy()]
        for policy in policies:
            start = policy.minimal_config()
            for neighbor in policy.generate_neighbors(start):
                self.assertEqual(sum(start[k] != neighbor[k] for k in start), 1)
                self.assertGreaterEqual(policy.structural_complexity(neighbor), policy.structural_complexity(start))

    def test_gmp_growth(self):
        policy = GMPSearchPolicy()
        start = policy.minimal_config()
        neighbors = policy.generate_neighbors(start)
        self.assertEqual(len(neighbors), 4)
        self.assertIn({"aligned": {"P": 1, "M": 0}, "lagging": {"P": 3, "M": 1, "Q": 1}, "leading": None}, neighbors)
        for c in neighbors:
            self.assertEqual(sum(start[k] != c[k] for k in start), 1)
            self.assertGreater(policy.structural_complexity(c), policy.structural_complexity(start))
        full = policy.reference_configs()[0]
        self.assertEqual(policy.generate_neighbors(full), [])

    def test_counts_match_all_trainers(self):
        for policy, trainer in [(MPSearchPolicy(), MPTrainer()), (EMPSearchPolicy(), EMPTrainer()),
                                (GMPSearchPolicy(), GMPTrainer()), (PrunedVolterraSearchPolicy(), PrunedVolterraTrainer())]:
            for c in [policy.minimal_config(), *policy.reference_configs()]:
                self.assertEqual(policy.structural_complexity(c), trainer.count_basis_terms(c))

    def test_lut_geometric_growth_and_limits(self):
        policy = LUTSearchPolicy()
        c = policy.minimal_config()
        for bins in (8, 16, 32, 64, 128, 256, 512):
            c["bins"] = bins
            self.assertIn({**c, "bins": bins*2}, policy.generate_neighbors(c))
        self.assertFalse(policy.is_valid({**c, "bins": 9}))
        self.assertFalse(policy.is_valid({**c, "bins": 2048}))


class DataTests(unittest.TestCase):
    @staticmethod
    def raw(seed=1, n=100):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)+1j*rng.normal(size=n)
        return RawIQData(np.arange(n)*1e-9, x, (2+.3j)*x)

    def test_training_only_gain(self):
        processor = DataProcessor(3, 1)
        a, b, c = self.raw(1), self.raw(2), self.raw(3)
        b.y *= 100
        c.y *= 1000
        data = processor.prepare(a, b, c)
        self.assertAlmostEqual(data.gain, 2+.3j)
        np.testing.assert_allclose(data.train.z, data.train.x)

    def test_duplicates_rejected(self):
        a, b = self.raw(1), self.raw(2)
        with self.assertRaisesRegex(ValueError, "idénticos"):
            DataProcessor(2, 0).prepare(a, b, a)

    def test_guard_chronological(self):
        processor = DataProcessor(3, 2)
        a, b, c = processor.split_chronologically(self.raw())
        self.assertGreater(b.time[0]-a.time[-1], 3e-9)
        self.assertGreater(c.time[0]-b.time[-1], 3e-9)
        with self.assertRaises(ValueError):
            DataProcessor(5, 2, guard=3)

    def test_integer_alignment(self):
        raw = self.raw(n=200)
        raw.y = np.concatenate([np.zeros(4), raw.y[:-4]])
        aligned = DataProcessor(0, 0).align_signals(raw, 10)
        self.assertEqual(aligned.metadata["dpd_alignment_delay"], 4)
        np.testing.assert_allclose(aligned.y, (2+.3j)*aligned.x)

    def test_invalid_times(self):
        raw = self.raw()
        raw.time[3] = raw.time[2]
        with self.assertRaises(ValueError):
            DataProcessor.validate(raw)


class HardwareTests(unittest.TestCase):
    def test_direct_mp_graph_counts(self):
        c = {"P": 3, "M": 1}
        model = MPTrainer().fit(random_part(), c)
        h = DirectHardwareEstimator().estimate(model, c)
        self.assertEqual((h.num_parameters, h.complex_multipliers, h.real_multipliers), (4, 4, 8))
        self.assertEqual((h.complex_adders, h.real_adders, h.delay_elements), (3, 2, 1))
        self.assertEqual(h.memory_bits, 2*18*(4+1))

    def test_direct_volterra_graph_counts(self):
        c = {"P": 3, "M": 1, "D": 0}
        model = PrunedVolterraTrainer().fit(random_part(), c)
        h = DirectHardwareEstimator().estimate(model, c)
        self.assertEqual(h.complex_multipliers, 8)
        self.assertEqual(h.real_multipliers, 0)

    def test_lut_interpolation_counts(self):
        c = {"bins": 8, "binning": "uniform_amplitude", "interpolation": "linear"}
        h = DirectHardwareEstimator().estimate(LUTTrainer().fit(random_part(), c), c)
        self.assertEqual((h.complex_multipliers, h.real_multipliers, h.lut_entries), (1, 5, 23))
        self.assertEqual(h.memory_bits, 18*31)

    def test_alternative_estimators(self):
        c = {"P": 5, "M": 1}
        model = MPTrainer().fit(random_part(), c)
        direct = DirectHardwareEstimator().estimate(model, c)
        lut = LUTAssistedHardwareEstimator().estimate(model, c)
        sharing = MultiplierSharingEstimator().estimate(model, c)
        self.assertLess(lut.real_multipliers, direct.real_multipliers)
        self.assertGreater(lut.memory_bits, direct.memory_bits)
        self.assertLess(sharing.complex_multipliers, direct.complex_multipliers)
        self.assertGreater(sharing.cycles_per_sample, 1)


class ScriptedTrainer(MPTrainer):
    """Controlled error landscape verifies optimizer independently of fitting."""
    def __init__(self, errors):
        super().__init__()
        self.errors = errors
        self.test_calls = 0
        self.test_data = None

    def fit(self, train_data, config):
        return TrainedModel("MP", config, np.array([1+0j]), FitDiagnostics(1, [1.], 1, 1, 1, True),
                            basis_terms=[{"kind": "envelope"}])

    def predict(self, data, model):
        if data is self.test_data:
            self.test_calls += 1
        return data.target*(1-np.sqrt(self.errors[model.config["P"]]))


class OptimizerTests(unittest.TestCase):
    def optimize(self, errors, **kwargs):
        kwargs.setdefault("stopping_rule", "reference_fraction")
        trainer = ScriptedTrainer(errors)
        data = AlignedInputData(random_part(seed=1), random_part(seed=2), random_part(seed=3), 1+0j)
        trainer.test_data = data.test
        optimizer = HyperparameterOptimizer(trainer, MPSearchPolicy(max_P=max(errors), max_M=0),
                                             DirectHardwareEstimator(), OptimizationCriteria(**kwargs))
        return optimizer.optimize(data), trainer

    def test_stop_at_80_and_test_once(self):
        result, trainer = self.optimize({1: 1., 3: .28, 5: .2, 7: .1}, target_fraction=.8)
        self.assertEqual(result.best_config["P"], 3)
        self.assertEqual(result.optimization_trace.stop_reason, "target_reached")
        self.assertAlmostEqual(result.progress_fraction, .8)
        self.assertEqual(trainer.test_calls, 1)

    def test_plateau_and_keep_best(self):
        result, _ = self.optimize({1: 1., 3: 1., 5: .1}, plateau_patience=1)
        self.assertEqual(result.optimization_trace.stop_reason, "performance_plateau")
        self.assertEqual(result.best_config["P"], 1)

    def test_reference_not_better(self):
        result, _ = self.optimize({1: .1, 3: .2, 5: .3})
        self.assertEqual(result.optimization_trace.stop_reason, "performance_plateau")
        self.assertEqual(result.progress_fraction, 0)

    def test_max_iterations(self):
        result, _ = self.optimize({1: 1., 3: .7, 5: .5, 7: .1}, max_iterations=1)
        self.assertEqual(result.optimization_trace.stop_reason, "max_iterations")

    def test_no_valid_neighbors(self):
        class DeadEndPolicy(MPSearchPolicy):
            def generate_neighbors(self, config):
                return []
        trainer = ScriptedTrainer({1: 1., 3: .1})
        data = AlignedInputData(random_part(), random_part(seed=2), random_part(seed=3), 1)
        result = HyperparameterOptimizer(trainer, DeadEndPolicy(max_P=3), DirectHardwareEstimator(),
                                         OptimizationCriteria(stopping_rule="reference_fraction")).optimize(data)
        self.assertEqual(result.optimization_trace.stop_reason, "no_valid_neighbors")

    def test_equal_cost_error_and_hardware_tie_break(self):
        trainer = ScriptedTrainer({1: 1., 3: .5, 5: .1})
        data = AlignedInputData(random_part(), random_part(seed=2), random_part(seed=3), 1)
        optimizer = HyperparameterOptimizer(trainer, MPSearchPolicy(), DirectHardwareEstimator(), OptimizationCriteria())
        current = optimizer.evaluate_candidate({"P": 1, "M": 0}, data)
        a = copy.deepcopy(optimizer.evaluate_candidate({"P": 3, "M": 0}, data))
        b = copy.deepcopy(optimizer.evaluate_candidate({"P": 5, "M": 0}, data))
        optimizer.baseline_error = 1.
        a.structural_complexity = b.structural_complexity = current.structural_complexity
        self.assertIs(optimizer.select_next_candidate(current, [a, b])[0], b)
        a.normalized_error_linear = b.normalized_error_linear
        a.hardware.real_multipliers = 10
        b.hardware.real_multipliers = 1
        self.assertIs(optimizer.select_next_candidate(current, [a, b])[0], b)

    def test_baseline_instability(self):
        data = AlignedInputData(partition(np.zeros(20), np.ones(20)), random_part(), random_part(), 1)
        optimizer = HyperparameterOptimizer(MPTrainer(), MPSearchPolicy(), DirectHardwareEstimator(), OptimizationCriteria())
        self.assertEqual(optimizer.optimize(data).optimization_trace.stop_reason, "numerical_instability")

    def test_test_changes_do_not_change_selection(self):
        train, val, test = random_part(seed=1), random_part(seed=2), random_part(seed=3)
        def run():
            return HyperparameterOptimizer(MPTrainer(), MPSearchPolicy(), DirectHardwareEstimator(),
                                           OptimizationCriteria()).optimize(AlignedInputData(train, val, test, 1))
        first = run()
        test.x *= 100
        second = run()
        self.assertEqual(first.best_config, second.best_config)
        self.assertEqual(first.nmse_validation_db, second.nmse_validation_db)
        self.assertNotEqual(first.nmse_test_db, second.nmse_test_db)


class SerializationTests(unittest.TestCase):
    def test_custom_pruning_basis_survives_reload(self):
        class OnlyDelayed(PruningRule):
            def keep(self, term, config):
                return min(term.delays()) == 1
        trainer = PrunedVolterraTrainer(pruning_rule=OnlyDelayed())
        data = random_part()
        model = trainer.fit(data, {"P": 3, "M": 2, "D": 1})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"custom.json"
            JSONWriter.write(model, path)
            loaded = JSONWriter.load_model(path)
            np.testing.assert_allclose(loaded.predict(data), trainer.predict(data, model))

    def test_model_roundtrip(self):
        data = random_part()
        for trainer, c in [(MPTrainer(), {"P": 3, "M": 1}),
                           (LUTTrainer(), {"bins": 8, "binning": "quantile", "interpolation": "linear"})]:
            model = trainer.fit(data, c)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/"model.json"
                JSONWriter.write(model, path)
                loaded = JSONWriter.load_model(path)
                np.testing.assert_allclose(loaded.predict(data), model.predict(data))

    def test_strict_json_nonfinite_and_complex(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"data.json"
            JSONWriter.write({"v": complex(1, 2), "inf": np.inf, "negative": -np.inf}, path)
            raw = path.read_text()
            json.loads(raw, parse_constant=lambda c: self.fail("Nonstandard JSON token"))
            values = JSONWriter.read(path)
            self.assertEqual(values["v"], 1+2j)
            self.assertEqual(values["inf"], np.inf)


class RealInputTests(unittest.TestCase):
    def test_agreed_split_gain_and_serialized_models(self):
        root = Path(__file__).resolve().parents[1]
        pipeline = TrainingPipeline(root/"dpd_config.json")
        data = pipeline.prepare_data()
        self.assertEqual(data.provenance["split_mode"], "training_folder_plus_chronological_validation_holdout")
        self.assertGreater(data.test.time[0], data.validation.time[-1])
        self.assertAlmostEqual(data.gain, DataProcessor.estimate_gain(data.train.x, data.train.z*data.gain))
        for name in pipeline.policies:
            result = pipeline.run_model(name)
            self.assertIsNotNone(result.model)
            self.assertTrue(np.isfinite(result.nmse_test_db))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/"result.json"
                JSONWriter.write(result, path)
                loaded = JSONWriter.load_model(path)
                self.assertAlmostEqual(nmse(data.test.target, loaded.predict(data.test)), result.nmse_test_db)


if __name__ == "__main__":
    unittest.main()
