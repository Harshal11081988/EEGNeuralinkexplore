"""
Shared constants and signal-processing helpers for the EEG BCI Visualizer.
"""

import numpy as np
from scipy.signal import welch

# Standard EEG frequency bands (Hz) relevant to motor imagery
FREQ_BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha/Mu (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
}

# Human-readable class labels for the PhysioNet EEGMMIDB motor imagery task
# (runs 4, 8, 12 -> imagined left fist vs imagined right fist)
CLASS_LABELS = {0: "Imagined Left Fist", 1: "Imagined Right Fist"}


def bandpower(signal_1d: np.ndarray, sfreq: float, band: tuple) -> float:
    """
    Compute average power of a 1D signal within a frequency band using Welch's method.

    Args:
        signal_1d: 1D numpy array, a single channel's time series.
        sfreq: sampling frequency in Hz.
        band: (low_freq, high_freq) tuple.

    Returns:
        Average power (float) within the band.
    """
    low, high = band
    nperseg = min(len(signal_1d), int(sfreq * 2))
    freqs, psd = welch(signal_1d, sfreq, nperseg=nperseg)
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx):
        return 0.0
    # np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid_fn(psd[idx], freqs[idx]))


def epoch_bandpowers(epoch: np.ndarray, sfreq: float) -> dict:
    """
    Compute average band power (averaged across all channels) for each frequency band.

    Args:
        epoch: array of shape (n_channels, n_times).
        sfreq: sampling frequency in Hz.

    Returns:
        Dict mapping band name -> average power across channels.
    """
    results = {}
    for band_name, band_range in FREQ_BANDS.items():
        powers = [bandpower(epoch[ch, :], sfreq, band_range) for ch in range(epoch.shape[0])]
        results[band_name] = float(np.mean(powers))
    return results
