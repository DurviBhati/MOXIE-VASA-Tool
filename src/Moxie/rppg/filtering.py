"""
rppg/filtering.py  —  Signal Detrending and Bandpass Filtering
===============================================================
Two-stage filtering applied to the combined RGB signal before POS:

  Stage 1 — Detrending
      Removes slow illumination drift (e.g. fluorescent light flicker,
      subject moving toward/away from a window).  Uses a simple
      moving-average subtraction rather than a polynomial fit —
      faster and sufficient for 30-second windows.

  Stage 2 — Butterworth Bandpass
      4th-order zero-phase Butterworth filter (applied with filtfilt
      so there is no phase shift that would distort the BVP waveform).
      Cutoffs: 0.7 Hz (42 bpm) to 3.0 Hz (180 bpm).
      These cover the full physiological HR range including stress.

Why IIR (Butterworth) over FIR
-------------------------------
At ~29 fps a FIR filter needs a very high order (~150+ taps) to achieve
a sharp cutoff at 0.7 Hz, requiring hundreds of frames of startup.
A 4th-order Butterworth achieves clean rolloff with far fewer samples
and is the standard choice in the rPPG literature (de Haan & Jeanne 2013,
Wang et al. POS 2017).

Public interface
----------------
    from rppg.filtering import detrend_signal, bandpass_filter

    detrended = detrend_signal(rgb_signal)
    filtered  = bandpass_filter(detrended, fps=29.97)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, butter


def detrend_signal(signal: np.ndarray, window_size: int = 33) -> np.ndarray:
    """
    Remove slow illumination drift from an RGB signal using
    moving-average subtraction.

    Parameters
    ----------
    signal      : np.ndarray  shape (N, 3)   raw RGB trace
    window_size : int  length of the moving average window in frames.
                  Default 33 ≈ 1 second at 30fps.  Must be odd.

    Returns
    -------
    detrended : np.ndarray  shape (N, 3)
    """
    if window_size % 2 == 0:
        window_size += 1   # enforce odd

    n = signal.shape[0]
    detrended = np.zeros_like(signal, dtype=np.float64)

    for ch in range(3):
        channel = signal[:, ch].astype(np.float64)
        # Pad edges with reflection to avoid boundary artifacts
        pad = window_size // 2
        padded = np.pad(channel, pad, mode="reflect")
        trend = np.convolve(padded, np.ones(window_size) / window_size, mode="valid")
        trend = trend[:n]   # trim to original length
        detrended[:, ch] = channel - trend

    return detrended


def bandpass_filter(
    signal:     np.ndarray,
    fps:        float,
    low_hz:     float = 0.7,
    high_hz:    float = 3.0,
    order:      int   = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to each RGB channel.

    Parameters
    ----------
    signal  : np.ndarray  shape (N, 3)   detrended RGB signal
    fps     : float  actual video frame rate (from preprocessing metadata)
    low_hz  : float  lower cutoff in Hz  (default 0.7 = 42 bpm)
    high_hz : float  upper cutoff in Hz  (default 3.0 = 180 bpm)
    order   : int    filter order  (default 4)

    Returns
    -------
    filtered : np.ndarray  shape (N, 3)

    Notes
    -----
    Uses filtfilt (zero-phase) so the BVP waveform shape is preserved —
    important for accurate peak detection in hr_estimation.py.
    Requires at least  3 * (order + 1) = 15  samples to be stable.
    """
    nyq = fps / 2.0

    # Guard: cutoffs must be strictly within (0, Nyquist)
    low  = max(0.01, min(low_hz  / nyq, 0.99))
    high = max(0.01, min(high_hz / nyq, 0.99))

    if low >= high:
        raise ValueError(
            f"Invalid bandpass range after normalization: "
            f"low={low:.3f} high={high:.3f}. "
            f"Check fps={fps} and cutoff settings."
        )

    # Minimum samples needed for filtfilt stability
    min_samples = 3 * (order + 1)
    if signal.shape[0] < min_samples:
        raise ValueError(
            f"Signal too short for bandpass filter: "
            f"{signal.shape[0]} samples, need at least {min_samples}. "
            f"Use a longer video window."
        )

    # Design filter as second-order sections (more numerically stable than ba)
    sos = butter(order, [low, high], btype="bandpass", output="sos")

    filtered = np.zeros_like(signal, dtype=np.float64)
    for ch in range(3):
        filtered[:, ch] = sosfiltfilt(sos, signal[:, ch].astype(np.float64))

    return filtered