
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from typing import Tuple

def compute_stft(signal: np.ndarray, fs: float,
                 nperseg: int = 1024, noverlap: int = 1000, nfft: int = 2048):
    """
    Compute STFT magnitude (log-scaled) similar to the paper's settings:
    - window length: 1024 samples
    - overlap: 1000 samples
    - FFT length: 2048
    Returns (T, F, Sxx_db), where T: seconds, F: Hz.
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None)
    S = np.abs(Zxx)
    S_db = 20 * np.log10(np.maximum(S, 1e-12))
    return t, f, S_db

def overlay_curve_on_spectrogram(t_spec: np.ndarray, f_spec: np.ndarray, S_db: np.ndarray,
                                 tprime_curve: np.ndarray, f_curve: np.ndarray,
                                 vmin=None, vmax=None, title="Spectrogram with Doppler Curve"):
    """
    Plot STFT spectrogram and overlay (t', f(t')) curve.
    """
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_spec, f_spec, S_db, shading='auto')
    plt.plot(tprime_curve, f_curve, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if title:
        plt.title(title)
    if vmin is not None or vmax is not None:
        plt.clim(vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.show()

# -------- Optional demo (synthetic): make an AM signal that roughly follows f(t') --------
def synth_from_curve(tprime: np.ndarray, f_curve: np.ndarray, fs: float, duration: float) -> np.ndarray:
    """
    Very simple synthesis: generate a sinusoid whose instantaneous frequency follows a time-frequency curve.
    We resample f_curve onto uniform time grid [0, duration]. This is not a physical propagation model,
    but sufficient to visualize an overlay demo.
    """
    t = np.linspace(0.0, duration, int(duration * fs), endpoint=False)
    f_interp = np.interp(t, np.clip(tprime, tprime.min(), tprime.max()), f_curve)
    phase = 2 * np.pi * np.cumsum(f_interp) / fs
    sig = 0.5 * np.sin(phase)
    sig += 0.2 * np.sin(2 * phase)
    return sig

if __name__ == "__main__":
    # Example if run directly (requires forward_models.py to be importable)
    from forward_models import forward_accel
    tprime, f_obs, _, _ = forward_accel(f0=25, l=1000, v0=100, a=-4.0, t0=20, t_min=0, t_max=40, dt=0.02)
    fs = 44100.0
    duration = float(tprime.max() - tprime.min())
    sig = synth_from_curve(tprime - tprime.min(), f_obs, fs=fs, duration=duration)
    t_spec, f_spec, S_db = compute_stft(sig, fs=fs)
    overlay_curve_on_spectrogram(t_spec, f_spec, S_db, tprime - tprime.min(), f_obs,
                                 title="Demo: Deceleration Doppler (synthetic)")
