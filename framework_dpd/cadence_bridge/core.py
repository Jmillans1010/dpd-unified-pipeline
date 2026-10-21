"""No training, coefficient changes, implicit normalization, or clipping."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np


def decode(value):
    if '__complex__' in value:
        return complex(*value['__complex__'])
    if '__ndarray__' in value:
        return np.asarray(value['values'], dtype=value['dtype'])
    if '__nonfinite__' in value:
        return float(value['__nonfinite__'])
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_hook=decode)


def checked_signal(signal):
    a = np.asarray(signal, dtype=complex)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError('Signal must be a finite complex vector with at least two samples')
    return a


def load_iq(path):
    data = np.genfromtxt(str(path), delimiter=',', names=True, encoding='utf-8-sig')
    required = {'time_reference_s', 'I_in', 'Q_in'}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError('Required CSV columns: time_reference_s,I_in,Q_in')
    x = checked_signal(data['I_in'] + 1j * data['Q_in'])
    time = np.asarray(data['time_reference_s'])
    delta = np.diff(time)
    if not np.all(np.isfinite(time)) or np.any(delta <= 0):
        raise ValueError('Times must be finite and strictly increasing')
    ts = float(np.median(delta))
    if not np.allclose(delta, ts, rtol=1e-5, atol=1e-18):
        raise ValueError('Nonuniform sampling: resampling must be explicit upstream')
    return x, 1.0 / ts, float(time[0])


class DPDModel(object):
    """Adapter around trained artifacts. apply(x) means predistortion F(x)."""
    def __init__(self, model_path, backend='auto', boundary='zero_pad'):
        self.path = Path(model_path)
        saved = read_json(self.path)
        self.model = saved.get('model', saved)
        if not self.model or self.model['model_name'] not in ['MP','EMP','GMP','PrunedVolterra','LUT']:
            raise ValueError('Unsupported or unavailable model')
        self.name = self.model['model_name']
        self.boundary = boundary
        if boundary not in ['zero_pad','trim']:
            raise ValueError('Boundary must be zero_pad or trim')
        delays = [0]
        for term in self.model.get('basis_terms', []):
            if term['kind'] == 'envelope':
                delays.append(term['signal_delay'])
                if term['envelope_power']:
                    delays.append(term['envelope_delay'])
            elif term['kind'] == 'volterra':
                delays.extend(term['direct_delays'] + term['conjugate_delays'])
            else:
                raise ValueError('Unknown serialized basis')
        self.past, self.future = max(delays), max(0, -min(delays))
        self.backend = ('native' if sys.version_info >= (3,10) else 'portable') if backend == 'auto' else backend
        if self.backend not in ['native','portable']:
            raise ValueError('Unknown backend')
        self.native = None
        if self.backend == 'native':
            from dpd.serialization import JSONWriter
            self.native = JSONWriter.load_model(self.path)

    def apply(self, x):
        x = checked_signal(x)
        if len(x) <= self.past + self.future + 1:
            raise ValueError('Insufficient samples for model context')
        if self.boundary == 'zero_pad':
            z = np.pad(x, (self.past,self.future), mode='constant')
            idx = np.arange(self.past,self.past+len(x))
            self.original_indices = np.arange(len(x))
        else:
            z = x
            idx = np.arange(self.past,len(x)-self.future)
            self.original_indices = idx
        if self.native is not None:
            from dpd.contracts import DataPartition
            result = self.native.predict(DataPartition(np.zeros_like(z),z,np.arange(len(z)),idx))
        else:
            result = self._portable(z,idx)
        return checked_signal(result)

    def _portable(self,z,idx):
        m = self.model
        if self.name == 'LUT':
            nodes, values = m['lut_nodes'], m['lut_values']
            mode = m['config']['interpolation']
            if mode == 'nearest':
                return z[idx]*values[np.searchsorted((nodes[:-1]+nodes[1:])/2,abs(z[idx]))]
            if mode != 'linear':
                raise ValueError('Unknown LUT interpolation')
            return z[idx]*(np.interp(abs(z[idx]),nodes,values.real)+1j*np.interp(abs(z[idx]),nodes,values.imag))
        terms, coefficients = m['basis_terms'], m['coefficients']
        if len(terms) != len(coefficients):
            raise ValueError('Coefficient/basis mismatch')
        result = np.zeros(len(idx),complex)
        for c,t in zip(coefficients,terms):
            if t['kind'] == 'envelope':
                value = z[idx-t['signal_delay']] * abs(z[idx-t['envelope_delay']])**t['envelope_power']
            else:
                value = np.ones(len(idx),complex)
                for d in t['direct_delays']:
                    value *= z[idx-d]
                for d in t['conjugate_delays']:
                    value *= z[idx-d].conjugate()
            result += c*value
        return result


def signal_stats(signal, resistance_ohm=None):
    v = checked_signal(signal)
    power = abs(v)**2
    mean_square = float(np.mean(power))
    out = dict(peak_I_V=float(abs(v.real).max()), peak_Q_V=float(abs(v.imag).max()),
               peak_envelope_V=float(abs(v).max()), complex_rms_V=float(np.sqrt(mean_square)),
               mean_square_V2=mean_square,
               papr_db=float(10*np.log10(power.max()/mean_square)) if mean_square else None)
    if resistance_ohm is not None:
        if not np.isfinite(resistance_ohm) or resistance_ohm <= 0:
            raise ValueError('Positive finite resistance required')
        out['ideal_RF_average_W'] = mean_square/(2*resistance_ohm)
        out['power_assumption'] = 'vRF=I*cos-Q*sin across R; ideal modulation, no modulator gain or mismatch'
    return out


def export_to_cadence(signal, fs, output_I, output_Q, voltage_scale,
                      max_abs_voltage, interpolation='linear', transition_s=None):
    signal = checked_signal(signal)
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(voltage_scale) or voltage_scale <= 0:
        raise ValueError('Positive finite Fs and voltage_scale required')
    if not np.isfinite(max_abs_voltage) or max_abs_voltage <= 0:
        raise ValueError('Explicit positive voltage limit required')
    v = checked_signal(signal*voltage_scale)
    stats = signal_stats(v)
    if max(stats['peak_I_V'],stats['peak_Q_V']) > max_abs_voltage:
        raise ValueError('Voltage limit exceeded: I={:.6g} V, Q={:.6g} V, limit={:.6g} V. No clipping.'.format(stats['peak_I_V'],stats['peak_Q_V'],max_abs_voltage))
    t = np.arange(len(v),dtype=float)/fs
    if interpolation == 'zoh':
        if transition_s is None or not np.isfinite(transition_s) or not 0 < transition_s < 1/fs:
            raise ValueError('ZOH requires 0 < transition_s < Ts')
        # Hold x[n-1] until t[n], then ramp to x[n] during transition_s.
        t = np.r_[0., np.column_stack((t[1:],t[1:]+transition_s)).ravel(),len(v)/fs]
        v = np.r_[v[0],np.column_stack((v[:-1],v[1:])).ravel(),v[-1]]
    elif interpolation != 'linear':
        raise ValueError('Interpolation must be linear or zoh')
    if np.any(np.diff(t) <= 0):
        raise ValueError('PWL timestamps not representably increasing')
    pi,pq = Path(output_I),Path(output_Q)
    if pi.resolve() == pq.resolve():
        raise ValueError('I and Q files must differ')
    if pi.exists() or pq.exists():
        raise FileExistsError('Refusing to overwrite existing stimuli')
    pi.parent.mkdir(parents=True,exist_ok=True); pq.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(str(pi),np.column_stack((t,v.real)),fmt='%.17e',delimiter=' ')
    np.savetxt(str(pq),np.column_stack((t,v.imag)),fmt='%.17e',delimiter=' ')
    return dict(num_samples=len(signal),pwl_points=len(t),sampling_rate_hz=fs,sampling_period_s=1/fs,
                voltage_scale=voltage_scale,max_abs_voltage=max_abs_voltage,
                interpolation=interpolation,transition_s=transition_s,end_time_s=float(t[-1]),
                end_behavior='hold final voltage; no implicit repeat',I_file=pi.name,Q_file=pq.name,
                units={'time':'s','amplitude':'V'},statistics=stats)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
