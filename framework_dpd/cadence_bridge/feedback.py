"""Analysis of explicitly aligned, demodulated feedback; no hidden gain fitting."""
import numpy as np
from .core import checked_signal


def analyze_feedback(reference, measured, gain, symbol_indices=None):
    x,y=checked_signal(reference),checked_signal(measured)
    if len(x)!=len(y): raise ValueError('Align and resample signals before analysis')
    if not np.isfinite(gain) or abs(gain)==0: raise ValueError('Explicit finite nonzero gain required')
    if np.sum(abs(x)**2)==0: raise ValueError('Reference has zero energy')
    z=y/gain; e=z-x
    ratio=float(np.sum(abs(e)**2)/np.sum(abs(x)**2))
    valid=abs(x)>max(abs(x).max()*1e-9,1e-15)
    result={'nmse_db':float(10*np.log10(ratio)) if ratio else -float('inf'),
            'relative_rms_percent':100*np.sqrt(ratio),
            'am_am_input_V':abs(x[valid]),'am_am_output_V':abs(y[valid]),
            'am_pm_deg':np.angle(z[valid]*x[valid].conjugate(),deg=True),
            'evm_percent':None,'evm_scope':'not computed: explicit symbol indices required'}
    if symbol_indices is not None:
        raw=np.asarray(symbol_indices)
        if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.integer) or len(raw)==0 or np.any(raw<0) or np.any(raw>=len(x)):
            raise ValueError('Invalid symbol indices')
        energy=np.sum(abs(x[raw])**2)
        if energy==0: raise ValueError('Zero symbol reference energy')
        result['evm_percent']=float(100*np.sqrt(np.sum(abs(e[raw])**2)/energy))
        result['evm_scope']='relative to supplied reference at explicit symbol indices; no decision or fitted gain'
    return result
