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
        self.assertNotIn("test",m["datasets"])
        comparison=JSONWriter.read(self.result_dir/"comparison/metrics.json")
        self.assertEqual(len(comparison["rows"]),5)
        self.assertEqual(comparison["winner"],min(comparison["rows"],key=lambda r:r["nmse_waveform_db"])["model"])
        for row in comparison["rows"]:
            saved=JSONWriter.read(self.result_dir/"models"/(row["model"]+".json"))
            trace=saved["optimization_trace"]
            valid=[x for x in trace["candidates"].values() if x["valid"]]
            self.assertAlmostEqual(trace["best_bic_score_db"],min(x["bic_score_db"] for x in valid))
            self.assertFalse(any("validation" in key for key in trace["candidates"][trace["selected"]]))
        for p,h in self.source_hashes.items():
            self.assertEqual(digest(p),h)

    def test_symbol_indices_and_input_inference(self):
        meta=JSONWriter.read(self.result_dir/"preprocessed/validation/metadata.json")
        self.assertEqual(meta["signal_config"]["samples_per_symbol"],16)
        self.assertEqual(meta["signal_config"]["symbol_rate_hz"],250e6)
        comparison=JSONWriter.read(self.result_dir/"comparison/metrics.json")
        indices=comparison["symbol_indices"]
        self.assertTrue(np.all(np.diff(indices)==16))
        for name in self.config["models"]:
            a=np.genfromtxt(self.result_dir/"comparison"/f"{name}_symbols.csv",delimiter=",",names=True)
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
        for name in ["sin_dpd",*self.config["models"]]:
            dest=self.result_dir/"exports"/name
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
        with self.assertRaisesRegex(ValueError,"no uniformes"):
            preprocess(src,"validation",self.root/"bad_clock",self.config,False)

    def test_reuse_has_identical_model_hashes_and_does_not_refit(self):
        reused=run_pipeline(self.config_path,stage="compare",models_dir=self.result_dir,make_plots=False)
        for name in self.config["models"]:
            self.assertEqual(digest(reused/"models"/f"{name}.json"),digest(self.result_dir/"models"/f"{name}.json"))
        self.assertIn("reused_models",JSONWriter.read(reused/"manifest.json"))

    def test_feedback_uses_desired_reference_and_fixed_gain(self):
        source=self.root/"feedback"; source.mkdir()
        data=np.genfromtxt(self.result_dir/"exports/MP/samples.csv",delimiter=",",names=True)
        gain=JSONWriter.read(self.result_dir/"manifest.json")["gain"]
        x=data["I_desired_V"]+1j*data["Q_desired_V"]
        actual=data["I_export_V"]+1j*data["Q_export_V"]
        y=np.r_[np.zeros(3,complex),gain*x[:-3]]
        for ch,values in (("I_in",actual.real),("Q_in",actual.imag),("I_out",y.real),("Q_out",y.imag)):
            np.savetxt(source/f"{ch}_validation.csv",np.column_stack((data["time_s"],values)),delimiter=",",header=f"/{ch.upper()} X,/{ch.upper()} Y",comments="")
        shutil.copyfile(self.root/"demo/validation/input",source/"input")
        out=run_feedback(self.config_path,self.result_dir,"MP",source,False)
        metrics=JSONWriter.read(out/"metrics.json")
        self.assertLess(metrics["evm_percent"],1e-10)

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
