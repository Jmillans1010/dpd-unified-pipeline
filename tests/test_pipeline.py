import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from make_demo import create_demo
from unified_dpd.pipeline import run_pipeline, preprocess, load_config, voltage_policy, digest
from unified_dpd.feedback import run_feedback
from dpd.serialization import JSONWriter
from dpd.contracts import DataPartition
from dpd.trainers import MPTrainer
from dpd.search import MPSearchPolicy
from dpd.hardware import DirectHardwareEstimator
from unified_dpd.search import TrainingBICSearch
from unified_dpd.layout import RUN_DIRECTORIES
from unified_dpd.results import prepare_results


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch=Path(__file__).resolve().parents[1]/"runs"
        scratch.mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=scratch)
        cls.root=Path(cls.temp.name)
        cls.config_path=create_demo(cls.root/"demo",symbols=80)
        cls.config=load_config(cls.config_path)[1]
        cls.source_hashes={str(p):digest(p) for p in (cls.root/"demo").glob("*/*") if p.is_file()}
        cls.result_dir=run_pipeline(cls.config_path,make_plots=False)

    @classmethod
    def tearDownClass(cls):
        assert cls.root.resolve().is_relative_to(Path(__file__).resolve().parents[1]/"runs")
        cls.temp.cleanup()

    def test_e2e_contract_and_no_test_split(self):
        m=JSONWriter.read(self.result_dir/"manifest.json")
        self.assertEqual(m["status"],"completed")
        self.assertEqual(m["artifact_language"],"en")
        self.assertEqual({p.name for p in self.result_dir.iterdir() if p.is_dir()},set(RUN_DIRECTORIES))
        self.assertTrue((self.result_dir/"model_comparisons/report.md").is_file())
        self.assertIn("# Reconstruction comparison",(self.result_dir/"model_comparisons/report.md").read_text())
        self.assertTrue((self.result_dir/"diagnostics/training/sources/input").is_file())
        self.assertNotIn("test",m["datasets"])
        comparison=JSONWriter.read(self.result_dir/"model_comparisons/metrics.json")
        self.assertEqual(len(comparison["rows"]),5)
        self.assertEqual(comparison["winner"],min(comparison["rows"],key=lambda r:r["nmse_waveform_db"])["model"])
        for row in comparison["rows"]:
            saved=JSONWriter.read(self.result_dir/"model_results"/(row["model"]+".json"))
            trace=saved["optimization_trace"]
            valid=[x for x in trace["candidates"].values() if x["valid"]]
            self.assertAlmostEqual(trace["best_bic_score_db"],min(x["bic_score_db"] for x in valid))
            self.assertFalse(any("validation" in key for key in trace["candidates"][trace["selected"]]))
        for p,h in self.source_hashes.items():
            self.assertEqual(digest(p),h)

    def test_symbol_indices_and_input_inference(self):
        meta=JSONWriter.read(self.result_dir/"diagnostics/validation/metadata.json")
        self.assertEqual(meta["signal_config"]["samples_per_symbol"],16)
        self.assertEqual(meta["signal_config"]["symbol_rate_hz"],250e6)
        comparison=JSONWriter.read(self.result_dir/"model_comparisons/metrics.json")
        indices=comparison["symbol_indices"]
        self.assertTrue(np.all(np.diff(indices)==16))
        for name in self.config["models"]:
            a=np.genfromtxt(self.result_dir/"model_comparisons"/f"{name}_symbols.csv",delimiter=",",names=True)
            np.testing.assert_array_equal(a["aligned_index"],indices)

    def test_voltage_limit_after_scale_and_pwl(self):
        x=np.array([3+1j,-3-3j,3.001+0j,.2-3.01j,1+1j])
        before,after,idx=voltage_policy(x,1,3,"zero_pair")
        np.testing.assert_array_equal(idx,[2,3])
        np.testing.assert_array_equal(after[:2],x[:2])
        np.testing.assert_array_equal(after[2:4],[0j,0j])
        self.assertEqual(voltage_policy(x,2,3,"zero_pair")[2].size,4)
        with self.assertRaises(ValueError):
            voltage_policy(np.array([1+1j,np.nan]),1,3,"zero_pair")
        for name in ["no_dpd",*self.config["models"]]:
            dest=self.result_dir/"outputs"/name
            samples=np.genfromtxt(dest/"samples.csv",delimiter=",",names=True)
            i=np.loadtxt(dest/"I.txt"); q=np.loadtxt(dest/"Q.txt")
            self.assertTrue(np.all(np.diff(i[:,0])>0))
            np.testing.assert_array_equal(i[:,0],q[:,0])
            np.testing.assert_array_equal(i[:,1],samples["I_export_V"])
            np.testing.assert_array_equal(q[:,1],samples["Q_export_V"])
            self.assertLessEqual(max(abs(i[:,1]).max(),abs(q[:,1]).max()),3)

    def test_missing_input_and_nonuniform_clock_rejected(self):
        src=self.root/"invalid"
        shutil.copytree(self.root/"demo/validation",src)
        (src/"input").rename(src/"input.saved")
        with self.assertRaisesRegex(ValueError,"required files"):
            preprocess(src,"validation",self.root/"bad_missing",self.config,False)
        (src/"input.saved").rename(src/"input")
        for p in src.glob("*.csv"):
            a=np.loadtxt(p,delimiter=",",skiprows=1); a[4,0]+=10e-12
            header=p.read_text().splitlines()[0]
            np.savetxt(p,a,delimiter=",",header=header,comments="",fmt="%.17g")
        with self.assertRaisesRegex(ValueError,"non-uniform"):
            preprocess(src,"validation",self.root/"bad_clock",self.config,False)

    def test_reuse_has_identical_model_hashes_and_does_not_refit(self):
        for source in (self.result_dir,self.result_dir/"model_results"):
            with self.subTest(source=source):
                reused=run_pipeline(self.config_path,stage="compare",models_dir=source,make_plots=False)
                for name in self.config["models"]:
                    self.assertEqual(digest(reused/"model_results"/f"{name}.json"),digest(self.result_dir/"model_results"/f"{name}.json"))
                self.assertIn("reused_models",JSONWriter.read(reused/"manifest.json"))
                self.assertTrue((reused/"model_comparisons/metrics.json").is_file())
                self.assertEqual(list((reused/"outputs").iterdir()),[])

    def test_legacy_model_folders_remain_readable(self):
        legacy=self.root/"legacy_models_run"
        shutil.copytree(self.result_dir/"model_results",legacy/"models")
        shutil.copyfile(self.result_dir/"manifest.json",legacy/"manifest.json")
        before={p.name:digest(p) for p in (legacy/"models").iterdir()}
        expected=JSONWriter.read(self.result_dir/"model_comparisons/metrics.json")
        for source in (legacy,legacy/"models"):
            with self.subTest(source=source):
                reused=run_pipeline(self.config_path,stage="compare",models_dir=source,make_plots=False)
                actual=JSONWriter.read(reused/"model_comparisons/metrics.json")
                self.assertEqual(actual["winner"],expected["winner"])
                self.assertEqual([r["nmse_waveform_db"] for r in actual["rows"]],
                                 [r["nmse_waveform_db"] for r in expected["rows"]])
                for name,sha in before.items():
                    self.assertEqual(digest(reused/"model_results"/name),sha)
                    self.assertEqual(digest(legacy/"models"/name),sha)

    def test_preprocess_uses_the_same_categories(self):
        result=run_pipeline(self.config_path,stage="preprocess",make_plots=False)
        self.assertEqual({p.name for p in result.iterdir() if p.is_dir()},set(RUN_DIRECTORIES))
        self.assertTrue((result/"diagnostics/validation/summary.txt").is_file())
        self.assertEqual(list((result/"outputs").iterdir()),[])
        self.assertEqual(list((result/"model_results").iterdir()),[])

    def test_results_hold_desired_references_and_preserve_simulation_files(self):
        for name in ["no_dpd",*self.config["models"]]:
            samples=np.genfromtxt(self.result_dir/"outputs"/name/"samples.csv",delimiter=",",names=True)
            folder=self.result_dir/"results"/name
            for channel in ("I","Q"):
                reference=np.genfromtxt(folder/f"{channel}_in.csv",delimiter=",",names=True)
                np.testing.assert_array_equal(reference["time_s"],samples["time_s"])
                np.testing.assert_array_equal(reference["voltage_V"],samples[f"{channel}_desired_V"])
                self.assertFalse((folder/f"{channel}_out.csv").exists())
        output=self.result_dir/"results/EMP/I_out.csv"
        output.write_text("simulation output placeholder for overwrite protection test",encoding="utf-8")
        original=digest(output)
        with self.assertRaises(FileExistsError):
            prepare_results(self.result_dir)
        self.assertEqual(digest(output),original)

    def test_feedback_uses_desired_reference_and_fixed_gain(self):
        source=self.root/"feedback"; source.mkdir()
        data=np.genfromtxt(self.result_dir/"outputs/MP/samples.csv",delimiter=",",names=True)
        gain=JSONWriter.read(self.result_dir/"manifest.json")["gain"]
        x=data["I_desired_V"]+1j*data["Q_desired_V"]
        actual=data["I_export_V"]+1j*data["Q_export_V"]
        y=np.r_[np.zeros(3,complex),gain*x[:-3]]
        for ch,values in (("I_in",actual.real),("Q_in",actual.imag),("I_out",y.real),("Q_out",y.imag)):
            np.savetxt(source/f"{ch}_validation.csv",np.column_stack((data["time_s"],values)),delimiter=",",header=f"/{ch.upper()} X,/{ch.upper()} Y",comments="")
        shutil.copyfile(self.root/"demo/validation/input",source/"input")
        out=run_feedback(self.config_path,self.result_dir,"MP",source,False)
        metrics=JSONWriter.read(out/"model_comparisons"/"metrics.json")
        self.assertLess(metrics["evm_percent"],1e-10)
        self.assertEqual({p.name for p in out.iterdir() if p.is_dir()},set(RUN_DIRECTORIES))
        self.assertTrue((out/"diagnostics/aligned_feedback.csv").is_file())
        legacy=self.root/"legacy_feedback_run"
        shutil.copytree(self.result_dir/"outputs",legacy/"exports")
        # This is a temporary fixture, not a migration of the user's saved runs.
        source_baseline=legacy/"exports/no_dpd"
        self.assertTrue(source_baseline.resolve().is_relative_to(self.root.resolve()))
        source_baseline.rename(legacy/"exports/sin_dpd")
        shutil.copyfile(self.result_dir/"manifest.json",legacy/"manifest.json")
        for reference,name in ((legacy,"MP"),(legacy,"no_dpd"),(self.result_dir,"sin_dpd")):
            with self.subTest(reference=reference,name=name):
                feedback=run_feedback(self.config_path,reference,name,source,False)
                result=JSONWriter.read(feedback/"model_comparisons/metrics.json")
                self.assertLess(result["evm_percent"],1e-10)
                saved=JSONWriter.read(feedback/"manifest.json")
                self.assertNotEqual(saved["model"],"sin_dpd")

    def test_plateau_extension_does_not_force_larger_winner(self):
        rng=np.random.default_rng(9); z=rng.normal(size=128)+1j*rng.normal(size=128)
        d=DataPartition(2*z,z,np.arange(len(z)),np.arange(4,124))
        cfg={**self.config["search"],"plateau_extra_depth":2,"max_candidates":20}
        result=TrainingBICSearch(MPTrainer(),MPSearchPolicy(5,2),DirectHardwareEstimator(),cfg).run(d)
        self.assertGreater(len(result["optimization_trace"]["candidates"]),1)
        valid=[x for x in result["optimization_trace"]["candidates"].values() if x["valid"]]
        self.assertAlmostEqual(result["optimization_trace"]["best_bic_score_db"],min(x["bic_score_db"] for x in valid))

    def test_extra_depth_crosses_a_worse_intermediate_candidate(self):
        from dpd.contracts import FitDiagnostics
        class FixtureTrainer:
            name="fixture"
            def count_basis_terms(self,c): return c["P"]
            def fit(self,data,c):
                err={1:.1,3:.11,5:.001}[c["P"]]
                return SimpleNamespace(config=c,diagnostics=FitDiagnostics(c["P"],[1],1,c["P"],1,True,""),
                    predict=lambda partition: partition.target*(1+np.sqrt(err)))
        estimator=SimpleNamespace(estimate=lambda m,c:None)
        data=DataPartition(np.ones(100,complex),np.ones(100,complex),np.arange(100),np.arange(100))
        cfg={**self.config["search"],"plateau_extra_depth":0}
        basic=TrainingBICSearch(FixtureTrainer(),MPSearchPolicy(5,0),estimator,cfg).run(data)
        extended=TrainingBICSearch(FixtureTrainer(),MPSearchPolicy(5,0),estimator,{**cfg,"plateau_extra_depth":1}).run(data)
        self.assertEqual(basic["best_config"]["P"],1)
        self.assertEqual(extended["best_config"]["P"],5)


if __name__=="__main__":
    unittest.main()
