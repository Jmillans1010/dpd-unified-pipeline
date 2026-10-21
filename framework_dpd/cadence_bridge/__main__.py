import argparse
import json
from pathlib import Path
import numpy as np
from .core import DPDModel, load_iq, signal_stats, export_to_cadence, sha256


def plots(original, dpd, fs, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(len(original))/fs*1e9
    fig,axes=plt.subplots(3,1,figsize=(11,8))
    for ax,a,b,label in zip(axes,[original.real,original.imag,abs(original)],[dpd.real,dpd.imag,abs(dpd)],['I (V)','Q (V)','Magnitud (V)']):
        ax.plot(t,a,label='Original',lw=.8); ax.plot(t,b,label='DPD',lw=.8,alpha=.7)
        ax.set(xlabel='Tiempo (ns)',ylabel=label); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(str(directory/'tiempo.png'),dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    window=np.hanning(len(original)); f=np.fft.fftshift(np.fft.fftfreq(len(original),1/fs))/1e9
    for v,label in [(original,'Original'),(dpd,'DPD')]:
        spec=abs(np.fft.fftshift(np.fft.fft(v*window)))/window.sum()
        axes[0].plot(f,20*np.log10(np.maximum(spec,1e-15)),label=label,lw=.8)
        axes[1].scatter(v.real,v.imag,label=label,s=3,alpha=.4)
    axes[0].set(xlabel='Frecuencia banda base (GHz)',ylabel='FFT, dB respecto a 1 V (Hann)',title='Espectro de verificación; no PSD ni ACLR')
    axes[1].set(xlabel='I (V)',ylabel='Q (V)',title='Constelación de muestras'); axes[1].set_aspect('equal',adjustable='box')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(str(directory/'espectro_constelacion.png'),dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description='Apply saved DPD F(x), then export two Spectre PWL sources')
    p.add_argument('--model',required=True); p.add_argument('--input',required=True)
    p.add_argument('--output',required=True); p.add_argument('--voltage-scale',required=True,type=float)
    p.add_argument('--max-abs-voltage',type=float)
    p.add_argument('--boundary',choices=['zero_pad','trim'],default='zero_pad')
    p.add_argument('--backend',choices=['auto','native','portable'],default='auto')
    p.add_argument('--interpolation',choices=['linear','zoh'],default='linear')
    p.add_argument('--transition-s',type=float); p.add_argument('--resistance-ohm',type=float)
    p.add_argument('--diagnose-only',action='store_true'); p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    if not a.diagnose_only and a.max_abs_voltage is None: p.error('--max-abs-voltage required for export')
    if not np.isfinite(a.voltage_scale) or a.voltage_scale<=0: p.error('Positive finite voltage scale required')
    directory=Path(a.output)
    if directory.exists(): p.error('Use a new output directory to avoid mixing runs')
    x,fs,source_t0=load_iq(a.input)
    model=DPDModel(a.model,a.backend,a.boundary)
    dpd=model.apply(x); reference=x[model.original_indices]
    directory.mkdir(parents=True)
    meta=dict(model=model.name,config=model.model['config'],backend=model.backend,
              model_sha256=sha256(a.model),input_sha256=sha256(a.input),direction='F(x), no y/G division',
              source_time_origin_s=source_t0,export_time_origin_s=0.,
              first_original_index=int(model.original_indices[0]),last_original_index=int(model.original_indices[-1]),
              boundary=a.boundary,past_samples=model.past,future_samples=model.future,
              offline_lookahead_s=model.future/fs,streaming_latency_needed_samples=model.future,
              boundary_affected_output_indices=([0,model.past,len(x)-model.future,len(x)] if a.boundary=='zero_pad' else []),
              sampling_rate_hz=fs,num_samples=len(dpd),voltage_scale=a.voltage_scale,
              source_units='V, user confirmed; no RMS or peak normalization',
              original_stats=signal_stats(reference*a.voltage_scale,a.resistance_ohm),
              dpd_stats=signal_stats(dpd*a.voltage_scale,a.resistance_ohm),status='diagnostics_only')
    np.savetxt(str(directory/'samples.csv'),np.column_stack((np.arange(len(dpd))/fs,reference.real,reference.imag,dpd.real,dpd.imag)),delimiter=',',header='time_s,I_original,Q_original,I_dpd_before_scale,Q_dpd_before_scale',comments='')
    if not a.no_plots: plots(reference*a.voltage_scale,dpd*a.voltage_scale,fs,directory)
    if not a.diagnose_only:
        try:
            exported=export_to_cadence(dpd,fs,directory/'I_DPD.txt',directory/'Q_DPD.txt',a.voltage_scale,a.max_abs_voltage,a.interpolation,a.transition_s)
            meta['export']=exported; meta['status']='exported_not_simulated'
            (directory/'sources_example.scs').write_text('simulator lang=spectre\n// Replace paths with absolute paths on the Cadence server. Disable W0 first.\nV_I_DPD (I_IN 0) vsource type=pwl file="/ABSOLUTE/SERVER/PATH/I_DPD.txt"\nV_Q_DPD (Q_IN 0) vsource type=pwl file="/ABSOLUTE/SERVER/PATH/Q_DPD.txt"\n',encoding='ascii')
        except ValueError as error:
            meta['status']='export_blocked'; meta['reason']=str(error)
            (directory/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            raise
    (directory/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))


if __name__=='__main__': main()
