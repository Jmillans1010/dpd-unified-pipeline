"""Static scientific figures; final constellation plots contain symbols only."""
import numpy as np


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def diagnostics(raw, fs, directory):
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for signal, label in ((raw.input, "Input"), (raw.output, "Output")):
        n = min(len(signal.values), 512)
        t = (signal.time[:n]-signal.time[0])*1e9
        axes[0, 0].plot(t, signal.values[:n].real, label=label)
        axes[0, 1].plot(t, signal.values[:n].imag, label=label)
        v = signal.values
        amplitude = np.abs(np.fft.fftshift(np.fft.fft(v*np.hanning(len(v)))))/max(np.hanning(len(v)).sum(), 1)
        axes[1, 0].plot(np.fft.fftshift(np.fft.fftfreq(len(v), 1/fs))/1e6,
                        20*np.log10(np.maximum(amplitude, 1e-15)), label=label)
        axes[1, 1].hist(np.abs(v), bins=50, histtype="step", label=label)
    axes[0, 0].set(xlabel="Relative time (ns)", ylabel="I (V)")
    axes[0, 1].set(xlabel="Relative time (ns)", ylabel="Q (V)")
    axes[1, 0].set(xlabel="Baseband frequency (MHz)", ylabel="FFT amplitude (dB re 1 V)")
    axes[1, 1].set(xlabel="Envelope (V)", ylabel="Samples")
    for ax in axes.flat:
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Diagnostics before alignment / DPD")
    fig.savefig(directory/"00_raw_diagnostics.png", dpi=150)
    plt.close(fig)


def constellations(reference, predictions, rows, directory):
    plt = pyplot()
    # Caller passes only common symbol instants, never the full waveform.
    all_values = [reference, *predictions.values()]
    global_limit = 1.08*max(float(max(abs(v.real).max(), abs(v.imag).max())) for v in all_values)
    zoom_limit = 1.12*max(float(abs(reference.real).max()), float(abs(reference.imag).max()), 1e-9)
    for suffix, limit in (("global", global_limit), ("zoom", zoom_limit)):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        axes[0, 0].scatter(reference.real, reference.imag, s=9, alpha=.6, label="Measured input")
        axes[0, 0].set_title(f"Reference · {len(reference)} symbols")
        for ax, (name, predicted) in zip(list(axes.flat)[1:], predictions.items()):
            row = next(r for r in rows if r["model"] == name)
            outside = int(np.sum((abs(predicted.real)>limit) | (abs(predicted.imag)>limit)))
            ax.scatter(reference.real, reference.imag, s=10, alpha=.5, label="Measured input")
            ax.scatter(predicted.real, predicted.imag, s=9, alpha=.6, label="Reconstructed")
            ax.set_title(f"{name} · Symbol NMSE {row['nmse_symbols_db']:.2f} dB\n{outside} outside view")
        for ax in axes.flat:
            if not ax.has_data():
                ax.axis("off"); continue
            ax.set(xlabel="I (V)", ylabel="Q (V)", xlim=(-limit, limit), ylim=(-limit, limit))
            ax.set_aspect("equal"); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle("Validation: symbol instants only · same indices for all models")
        fig.savefig(directory/f"constellations_symbols_{suffix}.png", dpi=160)
        plt.close(fig)


def errors(indices, reference, predictions, directory):
    plt = pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for name, prediction in predictions.items():
        e = np.abs(prediction-reference)
        axes[0].plot(indices, e, label=name, alpha=.75)
        axes[1].plot(np.arange(len(e))/max(len(e)-1, 1), np.sort(e), label=name)
    axes[0].set(xlabel="Aligned sample index", ylabel="Absolute error (V)")
    axes[1].set(xlabel="Cumulative fraction", ylabel="Sorted absolute error (V)")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(directory/"reconstruction_errors.png", dpi=150)
    plt.close(fig)
