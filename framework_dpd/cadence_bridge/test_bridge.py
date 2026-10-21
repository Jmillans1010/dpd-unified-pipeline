import ast
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .core import DPDModel, export_to_cadence, load_iq
from .feedback import analyze_feedback


class BridgeTests(unittest.TestCase):
    def test_native_portable_and_boundary(self):
        root=Path(__file__).resolve().parents[1]
        run=root/'data/dpd_results/run_20261001T174609_515799Z'
        x,fs,t0=load_iq(root/'data/input/Validation/aligned_waveform.csv')
        for name in ['MP','EMP','GMP','PrunedVolterra','LUT']:
            for boundary in ['zero_pad','trim']:
                a=DPDModel(run/(name+'.json'),'native',boundary)
                b=DPDModel(run/(name+'.json'),'portable',boundary)
                ya,yb=a.apply(x),b.apply(x)
                np.testing.assert_allclose(ya,yb,rtol=1e-8,atol=1e-9)
                self.assertEqual(len(ya),len(x) if boundary=='zero_pad' else len(x)-a.past-a.future)
        self.assertAlmostEqual(fs/1e9,4.,places=8)

    def test_pwl(self):
        x=np.array([.1+.2j,-.2+.1j,.3-.1j])
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            export_to_cadence(x,1e9,d/'i',d/'q',2.,1.)
            t=np.loadtxt(str(d/'i')); q=np.loadtxt(str(d/'q'))
            np.testing.assert_allclose(t[:,0],np.arange(3)/1e9,atol=1e-20)
            np.testing.assert_allclose(t[:,1],2*x.real)
            np.testing.assert_array_equal(t[:,0],q[:,0])
            export_to_cadence(x,1e9,d/'iz',d/'qz',1.,1.,'zoh',1e-12)
            z=np.loadtxt(str(d/'iz'))
            self.assertTrue(np.all(np.diff(z[:,0])>0))
            np.testing.assert_allclose(z[:,1],[.1,.1,-.2,-.2,.3,.3])
            with self.assertRaises(ValueError): export_to_cadence(x,1e9,d/'bad',d/'badq',1.,.01)
            self.assertFalse((d/'bad').exists())
            with self.assertRaises(ValueError): export_to_cadence([np.nan,1],1e9,d/'bad',d/'badq',1.,1.)

    def test_feedback(self):
        x=np.array([1+1j,2+1j,1-1j]); g=2+.3j
        r=analyze_feedback(x,g*x,g,np.array([0,2]))
        self.assertLess(r['relative_rms_percent'],1e-12)
        self.assertLess(r['evm_percent'],1e-12)

    def test_python36_syntax(self):
        for p in Path(__file__).parent.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,6))


if __name__=='__main__': unittest.main()
