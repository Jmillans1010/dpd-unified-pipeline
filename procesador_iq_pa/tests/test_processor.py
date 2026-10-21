"""Pruebas numéricas y de integración; unittest no requiere pytest."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iq_processor.models import ProcessingConfig, ProcessingError, IQSignal, InputData, InputMetadata
from iq_processor.processing import (SampleAligner, EVMCalculator, SignalValidator, SymbolTimingSelector,
                                     complex_correlation, sinc_interpolate, DataProcessor)
from iq_processor.readers import CSVReader, InputSCSReader, MetadataResolver, spectre_number
from iq_processor.outputs import CSVWriter, JSONWriter


def config(**kwargs):
    return ProcessingConfig(**kwargs)


def data(x, y, t0=8e-9, out_t0=None):
    period = 250e-12
    out_t0 = t0 if out_t0 is None else out_t0
    return InputData(IQSignal(t0+np.arange(len(x))*period, np.asarray(x), np.arange(len(x))),
                     IQSignal(out_t0+np.arange(len(y))*period, np.asarray(y), np.arange(len(y))))


class NumericalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.x = rng.normal(size=1024)+1j*rng.normal(size=1024)

    def test_fft_matches_direct_complex_definition(self):
        y = self.x[:51]*(1.3+.7j)
        x = self.x[:37]
        lags, actual = complex_correlation(x, y)
        expected = np.array([sum(np.conj(x[n])*y[n+lag] for n in range(len(x)) if 0 <= n+lag < len(y)) for lag in lags])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_positive_and_negative_delay_rotation_and_indices(self):
        gain = 1.2*np.exp(1j*1.3)
        for delay in (9, -7, 0):
            with self.subTest(delay=delay):
                if delay >= 0:
                    y = np.concatenate([np.zeros(delay), gain*self.x])
                    x = self.x
                else:
                    x = np.concatenate([np.zeros(-delay), self.x])
                    y = gain*self.x
                pair = data(x, y)
                cfg = config(max_delay_samples=20)
                sampling = SignalValidator().validate(pair, cfg)
                aligned = SampleAligner().align(pair, sampling, cfg)
                self.assertEqual(aligned.integer_delay, delay)
                np.testing.assert_allclose(aligned.y, gain*aligned.x, atol=1e-12)
                np.testing.assert_allclose(aligned.output_positions-aligned.input_indices, delay)
                self.assertAlmostEqual(aligned.delay_s, delay*250e-12, places=18)

    def test_time_origins_enter_physical_delay(self):
        pair = data(self.x, self.x, out_t0=10e-9)
        cfg = config(max_delay_samples=5)
        a = SampleAligner().align(pair, SignalValidator().validate(pair, cfg), cfg)
        self.assertEqual(a.integer_delay, 0)
        self.assertAlmostEqual(a.delay_s, 2e-9, places=18)

    def test_evm_gain_rotation_and_known_orthogonal_error(self):
        x = np.array([1, 1j, -1, -1j], dtype=complex)
        error = np.array([.1, .1, .1, .1])
        e = EVMCalculator().calculate(x, 2j*x+error)
        self.assertAlmostEqual(e.alpha.real, 0)
        self.assertAlmostEqual(e.alpha.imag, 2)
        self.assertAlmostEqual(e.rms_percent, 5.)
        self.assertAlmostEqual(e.rms_db, 20*np.log10(.05))
        self.assertLess(EVMCalculator().calculate(self.x, (.3+2j)*self.x).rms_percent, 1e-12)

    def test_zero_energy_rejected_and_nonlinearity_retained(self):
        with self.assertRaises(ProcessingError):
            EVMCalculator().calculate(np.zeros(8), np.ones(8))
        with self.assertRaises(ProcessingError):
            EVMCalculator().calculate(np.ones(8), np.zeros(8))
        y = self.x*(1-.05*np.abs(self.x)**2)
        self.assertGreater(EVMCalculator().calculate(self.x, y).rms_percent, 1)

    def test_invalid_time_grids_rejected(self):
        for kind in ("duplicate", "drift"):
            pair = data(self.x, self.x)
            if kind == "duplicate":
                pair.input.time[1] = pair.input.time[0]
            else:
                pair.output.time += np.arange(len(self.x))*1e-16
            with self.assertRaises(ProcessingError):
                SignalValidator().validate(pair, config())

    def test_symbol_phase_known_best(self):
        rng = np.random.default_rng(44)
        symbols = self.x[:128]
        x = np.repeat(symbols, 4)
        y = 2j*x + .1*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
        y[2::4] = 2j*x[2::4]
        pair = data(x, y)
        cfg = config(max_delay_samples=0, minimum_symbols=16)
        a = SampleAligner().align(pair, SignalValidator().validate(pair, cfg), cfg)
        selection = SymbolTimingSelector().select(a, 4, cfg)
        self.assertEqual(selection.phase, 2)
        self.assertEqual(len(selection.x), 128)
        self.assertEqual(len(selection.candidates), 4)
        self.assertLess(selection.evm.rms_percent, 1e-12)

    def test_fractional_interpolator_on_bandlimited_signal(self):
        n = np.arange(500)
        y = np.exp(.08j*n)+.3*np.exp(.17j*n)
        positions = np.arange(30, 450)+.27
        reference = np.exp(.08j*positions)+.3*np.exp(.17j*positions)
        actual = sinc_interpolate(y, positions, 24)
        self.assertLess(float(np.max(np.abs(actual-reference))), 2e-4)

    def test_missing_sps_returns_aligned_partial_result(self):
        pair = data(self.x, self.x)
        cfg = config(max_delay_samples=8)
        result = DataProcessor().process(pair, InputMetadata(), cfg, SignalValidator().validate(pair, cfg), None)
        self.assertIsNone(result.selection)
        self.assertEqual(result.metadata["processing"]["stage_status"]["symbols_evm"]["status"], "skipped")

    def test_fractional_refinement_improves_known_delayed_waveform(self):
        rng = np.random.default_rng(135)
        n = np.arange(8192)
        frequencies = rng.uniform(.015, .45, 45)
        coefficients = rng.normal(size=45)+1j*rng.normal(size=45)
        x = np.exp(1j*np.outer(n, frequencies)) @ coefficients
        y = np.exp(1j*np.outer(n-3.3, frequencies)) @ coefficients
        pair = data(x, y)
        cfg = config(max_delay_samples=8)
        sampling = SignalValidator().validate(pair, cfg)
        integer = SampleAligner().align(pair, sampling, cfg)
        cfg.fractional_refinement = True
        refined = SampleAligner().align(pair, sampling, cfg)
        self.assertEqual(refined.integer_delay, 3)
        self.assertLess(abs(refined.fractional_delay-.3), .03)
        self.assertLess(EVMCalculator().calculate(refined.x, refined.y).rms_percent,
                        EVMCalculator().calculate(integer.x, integer.y).rms_percent/5)
        self.assertLess(len(refined.x), len(integer.x))


class ReaderAndIntegrationTests(unittest.TestCase):
    def test_spectre_suffixes_and_unresolved_expressions(self):
        self.assertEqual(spectre_number("250p"), 250e-12)
        self.assertEqual(spectre_number("1G"), 1e9)
        self.assertEqual(spectre_number("1M"), 1e6)
        self.assertEqual(spectre_number("1m"), .001)
        self.assertIsNone(spectre_number("1/Fs"))

    def test_netlist_continuations_context_and_quotes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input"
            path.write_text('simulator lang=spectre\nsubckt example a b\n'
                            'W9 (a b) wsource oversample=999\nends example\n'
                            'W0 (i 0 q 0) wsource MSource_ModType=QAM16 \\\n'
                            ' oversample=4 MSource_DataRate=1G\n'
                            'tran tran outputstart=8n strobeperiod=250p stop=2.2u\n'
                            'include "folder//name.scs" section=tt // comment\n', encoding="utf-8")
            result = InputSCSReader().read(path)
            self.assertEqual(result.signal_config["source_oversample"], 4)
            self.assertEqual(result.simulation_config["strobe_period_s"], 250e-12)
            self.assertIn("folder//name.scs", result.simulation_config["includes_not_expanded"][0])

    def test_real_data_export_reconstructs_evm_and_preserves_sources(self):
        cfg = ProcessingConfig.read(ROOT/"config.json")
        raw = CSVReader().read(cfg.input_files, ROOT, cfg)
        sampling = SignalValidator().validate(raw, cfg)
        metadata = InputSCSReader().read(ROOT/cfg.netlist)
        sps = MetadataResolver().resolve(metadata, cfg, ROOT, raw.sources, sampling)
        result = DataProcessor().process(raw, metadata, cfg, sampling, sps)
        self.assertAlmostEqual(sampling["sample_period_s"], 250e-12, places=18)
        self.assertAlmostEqual(raw.input.time[0], 8e-9, places=18)
        self.assertIsNotNone(result.selection)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            CSVWriter().write(result, directory)
            JSONWriter().write(result.metadata, directory/"metadata.json")
            saved = np.genfromtxt(directory/"selected_symbols.csv", delimiter=",", names=True)
            numerator = np.sum(saved["error_I"]**2+saved["error_Q"]**2)
            denominator = np.sum(saved["I_ref"]**2+saved["Q_ref"]**2)
            actual = 100*np.sqrt(numerator/denominator)
            self.assertAlmostEqual(actual, result.selection.evm.rms_percent, places=12)
            document = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(set(document), {"simulation_config", "signal_config", "processing", "statistics"})
            aligned = np.genfromtxt(directory/"aligned_waveform.csv", delimiter=",", names=True)
            np.testing.assert_allclose(aligned["I_in"], raw.input.values.real[aligned["input_original_index"].astype(int)])
            np.testing.assert_allclose(aligned["I_out"], raw.output.values.real[aligned["output_original_position"].astype(int)])

    def test_conflicting_sps_rejected(self):
        cfg = config(samples_per_symbol=16)
        metadata = InputMetadata(signal_config={"symbol_rate_hz": 1e9})
        with self.assertRaises(ProcessingError):
            MetadataResolver().resolve(metadata, cfg, ROOT, [], {"sample_rate_hz": 4e9, "sample_period_s": 250e-12})

    def test_missing_netlist_manual_rate_and_stale_metadata(self):
        metadata = InputSCSReader().read(None)
        cfg = config(manual_metadata={"signal_config": {"symbol_rate_hz": 250e6}})
        sps = MetadataResolver().resolve(metadata, cfg, ROOT, [], {"sample_rate_hz": 4e9, "sample_period_s": 250e-12})
        self.assertEqual(sps, 16)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"old.json").write_text('{"processing": {"sources": []}}', encoding="utf-8")
            cfg.previous_metadata = "old.json"
            with self.assertRaises(ProcessingError):
                MetadataResolver().resolve(InputMetadata(), cfg, root, [{"role": "I_in", "sha256": "different"}], {})

    def test_invalid_configuration_types(self):
        cfg = ProcessingConfig.read(ROOT/"config.json")
        cfg.minimum_overlap_fraction = "0.8"
        with self.assertRaises(ProcessingError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
