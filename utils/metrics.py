import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import Tuple, Union


def magnitude_spectrum(signal: np.ndarray, sample_rate: int, nfft: int, shift: bool = False):
    """Compute FFT of input signal."""
    spectrum = np.fft.fft(signal, n=nfft, axis=-1)

    if shift:
        spectrum = np.fft.fftshift(spectrum, axes=-1)
        freq = np.fft.fftshift(np.fft.fftfreq(signal.shape[-1], d=1 / sample_rate))
    else:
        freq = np.linspace(0, sample_rate, signal.shape[-1])

    return freq, spectrum


def IQ_to_complex(iq_signal: np.ndarray) -> np.ndarray:
    """Convert I/Q pairs to complex signals."""
    return iq_signal[..., 0] + 1j * iq_signal[..., 1]


def power_spectrum(complex_signal: np.ndarray, fs: float = 800e6,
                   nperseg: int = 2560, axis: int = -1):
    """Compute PSD using Welch method."""
    freq, psd = welch(complex_signal, fs=fs, nperseg=nperseg,
                      return_onesided=False, scaling='spectrum', axis=axis)

    half = nperseg // 2
    freq = np.concatenate([freq[half:], freq[:half]])
    psd = np.concatenate([psd[..., half:], psd[..., :half]], axis=-1)

    return freq, np.mean(psd, axis=0)


def NMSE(pred: np.ndarray, truth: np.ndarray) -> float:
    """Calculate Normalized Mean Square Error in dB."""
    mse = np.mean((truth[..., 0] - pred[..., 0]) ** 2 + (truth[..., 1] - pred[..., 1]) ** 2, axis=-1)
    energy = np.mean(truth[..., 0] ** 2 + truth[..., 1] ** 2, axis=-1)
    return np.mean(10 * np.log10(mse / energy))


def EVM(pred: np.ndarray, truth: np.ndarray, sample_rate: float = 800e6,
        bw_main: float = 200e6, n_sub: int = 10, nperseg: int = 2560) -> float:
    """Calculate Error Vector Magnitude in dB."""
    pred_c = IQ_to_complex(pred)
    truth_c = IQ_to_complex(truth)

    freq, spec_pred = magnitude_spectrum(pred_c, sample_rate, nperseg, shift=True)
    _, spec_truth = magnitude_spectrum(truth_c, sample_rate, nperseg, shift=True)

    idx_l = np.argmin(np.abs(freq + bw_main / 2))
    idx_r = np.argmin(np.abs(freq - bw_main / 2))
    ch_len = (idx_r - idx_l) // n_sub

    error = np.zeros((pred.shape[0], n_sub))
    for c in range(n_sub):
        slc = slice(idx_l + c * ch_len, idx_l + (c + 1) * ch_len)
        error[:, c] = np.mean(np.abs(spec_pred[:, slc] - spec_truth[:, slc]), axis=-1)
        error[:, c] /= np.mean(np.abs(spec_truth[:, slc]), axis=-1)

    return 20 * np.log10(np.mean(error.mean(axis=-1)))


def ACLR(pred: np.ndarray, fs: float = 800e6, nperseg: int = 2560,
         bw_main: float = 200e6, n_sub: int = 10) -> Tuple[float, float]:
    """Calculate Adjacent Channel Leakage Ratio."""
    freq, psd = power_spectrum(IQ_to_complex(pred), fs, nperseg)

    idx_l = np.argmin(np.abs(freq + bw_main / 2))
    idx_r = np.argmin(np.abs(freq - bw_main / 2))
    ch_len = (idx_r - idx_l) // n_sub

    sub_power = [np.sum(psd[idx_l + i * ch_len:idx_l + (i + 1) * ch_len]) for i in range(n_sub)]
    max_power = max(sub_power)

    aclr_l = 10 * np.log10(np.sum(psd[idx_l - ch_len:idx_l]) / max_power)
    aclr_r = 10 * np.log10(np.sum(psd[idx_r:idx_r + ch_len]) / max_power)

    return aclr_l, aclr_r


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average."""
    return pd.Series(data).rolling(window).mean().to_numpy()[window - 1:]


def plot_psd(sig1: np.ndarray, sig2: np.ndarray, label1: str = "wo_DPD",
             label2: str = "with_DPD", fs: float = 800e6, nperseg: int = 2560,
             smooth: int = 10):
    """Plot normalized PSD of two signals."""

    def compute_psd(sig):
        freq, psd = welch(sig[:nperseg], fs=fs, nperseg=nperseg, return_onesided=False)
        half = nperseg // 2
        freq = np.concatenate([freq[half:], freq[:half]])
        psd = np.concatenate([psd[half:], psd[:half]])
        psd_norm = 10 * np.log10(psd / np.max(psd))
        return freq, moving_average(psd_norm, smooth)

    f1, p1 = compute_psd(sig1)
    f2, p2 = compute_psd(sig2)

    plt.figure(figsize=(10, 6))
    plt.plot(f1[smooth // 2:-smooth // 2 + 1] / 1e6, p1, label=label1, color='blue')
    plt.plot(f2[smooth // 2:-smooth // 2 + 1] / 1e6, p2, label=label2, color='red', linestyle='--')
    plt.xlabel('Frequency (MHz)');
    plt.ylabel('Normalized PSD (dB)')
    plt.title('Normalized Power Spectral Density')
    plt.legend();
    plt.grid(True);
    plt.tight_layout();
    plt.show()


def plot_constellation(sig1: np.ndarray, sig2: np.ndarray, nperseg: int = 2560,
                       n_subc: int = 64, n_ch: int = 10):
    """Plot constellation diagram."""
    spec1 = np.fft.fftshift(np.fft.fft(sig1, n=nperseg, axis=-1), axes=-1)
    spec2 = np.fft.fftshift(np.fft.fft(sig2, n=nperseg, axis=-1), axes=-1)

    left = nperseg // 2 - n_subc * (n_ch // 2)
    slices = [slice(left + i * n_subc, left + (i + 1) * n_subc) for i in range(n_ch)]

    def collect(spec):
        parts = [spec[s] for s in slices]
        X, y = [], []
        for p in parts:
            p = p / max(np.abs(p.real).max(), np.abs(p.imag).max())
            X.append(p.real);
            y.append(p.imag)
        return np.concatenate(X), np.concatenate(y)

    X1, y1 = collect(spec1)
    X2, y2 = collect(spec2)

    plt.figure()
    plt.scatter(X1, y1, c='blue', label='DPD', alpha=0.3, edgecolors='none')
    plt.scatter(X2, y2, c='red', label='Standard', alpha=0.3, edgecolors='none')
    plt.legend();
    plt.grid(True);
    plt.show()