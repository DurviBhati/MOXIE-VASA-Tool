"""
rppg/signal_quality.py  —  Signal Quality Index (SQI) for BVP Windows
======================================================================
Overview
--------
Evaluates the quality of each BVP (blood volume pulse) window produced
by the POS algorithm.  Low-quality windows — caused by head movement,
illumination changes, blinks, or poor ROI tracking — are flagged so
that downstream HR estimation can exclude them, reducing MAE.

Quality Metrics (per window)
-----------------------------
Three complementary metrics are computed for each BVP window:

  1. **Spectral SNR** (weight 0.5)
     Ratio of power at the dominant frequency (±0.2 Hz) to total power
     in the 0.7–3.0 Hz band.  A clean pulse signal concentrates most
     energy at one frequency; noisy signals spread power across the
     spectrum.  Range: 0.0 (pure noise) to 1.0 (pure sinusoid).

  2. **Kurtosis score** (weight 0.3)
     Excess kurtosis of the BVP waveform.  A clean pulse-like signal
     has kurtosis in the range [1.5, 6.0] (leptokurtic, with sharp
     peaks from systolic pulses).  Values outside this range indicate
     either Gaussian noise (kurtosis ≈ 0) or extreme spikes (>10).
     The raw kurtosis is mapped to a 0–1 score via a Gaussian kernel
     centered at 3.5.

  3. **Periodicity score** (weight 0.2)
     Normalized autocorrelation at the dominant lag.  A periodic
     signal (heartbeat) produces strong autocorrelation at the lag
     corresponding to the cardiac period.  Random noise has near-zero
     autocorrelation.  Range: 0.0 to 1.0.

Composite SQI = 0.5 * spectral_snr + 0.3 * kurtosis_score + 0.2 * periodicity

Windows with SQI < threshold (default 0.4) are marked as low quality.

Public interface
----------------
    from rppg.signal_quality import compute_window_sqi, filter_bvp_by_quality

    sqi = compute_window_sqi(bvp_window, fps=29.97)
    # sqi : dict with keys 'spectral_snr', 'kurtosis_score',
    #        'periodicity', 'composite', 'is_good'

    bvp_filtered, mask, report = filter_bvp_by_quality(
        bvp_full, fps=29.97, window_sec=30, step_sec=10, threshold=0.4
    )
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


# ── Public API ────────────────────────────────────────────────────────────────

def compute_window_sqi(
    bvp_window: np.ndarray,
    fps:        float,
    low_hz:     float = 0.7,
    high_hz:    float = 3.0,
    peak_band:  float = 0.2,
    threshold:  float = 0.4,
) -> dict:
    """
    Compute Signal Quality Index for a single BVP window.

    Parameters
    ----------
    bvp_window : np.ndarray, shape (T,)
        One window of BVP waveform from the POS algorithm.
    fps : float
        Video frame rate (e.g. 29.97).
    low_hz : float
        Lower bound of physiological HR range in Hz (default 0.7 = 42 bpm).
    high_hz : float
        Upper bound of physiological HR range in Hz (default 3.0 = 180 bpm).
    peak_band : float
        Half-bandwidth around the dominant frequency for SNR computation
        (default ±0.2 Hz).
    threshold : float
        SQI threshold below which the window is marked low quality
        (default 0.4).

    Returns
    -------
    dict with keys:
        'spectral_snr'    : float  0.0–1.0
        'kurtosis_score'  : float  0.0–1.0
        'periodicity'     : float  0.0–1.0
        'composite'       : float  0.0–1.0  (weighted combination)
        'dominant_freq_hz': float  dominant frequency in Hz
        'is_good'         : bool   True if composite >= threshold
    """
    if len(bvp_window) < 10:
        return _empty_sqi()

    # Remove DC offset
    bvp = bvp_window - np.mean(bvp_window)

    # Guard: all-zero or constant signal
    if np.std(bvp) < 1e-10:
        return _empty_sqi()

    # ── Metric 1: Spectral SNR ──────────────────────────────────────────
    snr, dom_freq = _spectral_snr(bvp, fps, low_hz, high_hz, peak_band)

    # ── Metric 2: Kurtosis score ────────────────────────────────────────
    kurt_score = _kurtosis_score(bvp)

    # ── Metric 3: Periodicity (autocorrelation at dominant lag) ─────────
    period_score = _periodicity_score(bvp, fps, dom_freq, low_hz, high_hz)

    # ── Composite SQI ───────────────────────────────────────────────────
    composite = 0.5 * snr + 0.3 * kurt_score + 0.2 * period_score

    return {
        "spectral_snr":     round(float(snr), 4),
        "kurtosis_score":   round(float(kurt_score), 4),
        "periodicity":      round(float(period_score), 4),
        "composite":        round(float(composite), 4),
        "dominant_freq_hz": round(float(dom_freq), 4),
        "is_good":          bool(composite >= threshold),
    }


def filter_bvp_by_quality(
    bvp:          np.ndarray,
    fps:          float,
    window_sec:   float = 30.0,
    step_sec:     float = 10.0,
    threshold:    float = 0.4,
    low_hz:       float = 0.7,
    high_hz:      float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Evaluate quality of each overlapping window in a full BVP signal
    and zero out low-quality regions.

    This function mirrors the windowing used in pos_algorithm.py
    (same window_sec and step_sec defaults) so that quality assessment
    aligns with the windows used for BVP reconstruction.

    Parameters
    ----------
    bvp : np.ndarray, shape (N,)
        Full-length BVP waveform from apply_pos().
    fps : float
        Video frame rate.
    window_sec : float
        Window length in seconds (default 30.0, matching POS).
    step_sec : float
        Step between windows in seconds (default 10.0, matching POS).
    threshold : float
        SQI threshold for accepting a window (default 0.4).
    low_hz : float
        Lower HR frequency bound (default 0.7 Hz).
    high_hz : float
        Upper HR frequency bound (default 3.0 Hz).

    Returns
    -------
    bvp_filtered : np.ndarray, shape (N,)
        BVP with low-quality regions zeroed out.
    quality_mask : np.ndarray, shape (N,), dtype bool
        True for frames that belong to at least one good-quality window.
    report : dict
        Summary with keys:
            'n_windows'      : int   total windows evaluated
            'n_good'         : int   windows passing threshold
            'n_rejected'     : int   windows below threshold
            'pct_good'       : float percentage of good windows
            'per_window'     : list  of per-window SQI dicts
            'mean_sqi'       : float mean composite SQI across all windows
    """
    n_frames   = len(bvp)
    window_len = int(window_sec * fps)
    step_len   = int(step_sec * fps)

    # Clamp window length
    window_len = min(window_len, n_frames)
    step_len   = max(1, min(step_len, window_len))

    quality_mask = np.zeros(n_frames, dtype=bool)
    per_window   = []

    start = 0
    while start + window_len <= n_frames:
        end        = start + window_len
        bvp_window = bvp[start:end]
        sqi        = compute_window_sqi(
            bvp_window, fps, low_hz, high_hz, threshold=threshold
        )
        sqi["start_frame"] = int(start)
        sqi["end_frame"]   = int(end)
        per_window.append(sqi)

        if sqi["is_good"]:
            quality_mask[start:end] = True

        start += step_len

    # Handle remaining frames at the end
    if start < n_frames and (n_frames - start) >= 10:
        bvp_window = bvp[start:]
        sqi        = compute_window_sqi(
            bvp_window, fps, low_hz, high_hz, threshold=threshold
        )
        sqi["start_frame"] = int(start)
        sqi["end_frame"]   = int(n_frames)
        per_window.append(sqi)

        if sqi["is_good"]:
            quality_mask[start:] = True

    # Zero out low-quality regions
    bvp_filtered = bvp.copy()
    bvp_filtered[~quality_mask] = 0.0

    n_good     = sum(1 for w in per_window if w["is_good"])
    n_total    = len(per_window)
    composites = [w["composite"] for w in per_window]

    report = {
        "n_windows":  n_total,
        "n_good":     n_good,
        "n_rejected": n_total - n_good,
        "pct_good":   round(100.0 * n_good / max(n_total, 1), 1),
        "mean_sqi":   round(float(np.mean(composites)) if composites else 0.0, 4),
        "per_window": per_window,
    }

    return bvp_filtered, quality_mask, report


# ── Private helpers ───────────────────────────────────────────────────────────

def _spectral_snr(
    bvp:       np.ndarray,
    fps:       float,
    low_hz:    float,
    high_hz:   float,
    peak_band: float,
) -> tuple[float, float]:
    """
    Compute spectral signal-to-noise ratio.

    SNR = power within ±peak_band of dominant frequency /
          total power in [low_hz, high_hz].

    Returns (snr, dominant_frequency_hz).
    """
    nperseg = min(len(bvp), 512)
    freqs, psd = welch(bvp, fs=fps, nperseg=nperseg)

    # Mask to physiological range
    hr_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not hr_mask.any():
        return 0.0, 0.0

    psd_hr   = psd[hr_mask]
    freqs_hr = freqs[hr_mask]

    # Find dominant frequency
    peak_idx  = np.argmax(psd_hr)
    dom_freq  = freqs_hr[peak_idx]

    # Power at dominant frequency ± peak_band
    signal_mask = (freqs_hr >= dom_freq - peak_band) & \
                  (freqs_hr <= dom_freq + peak_band)
    # np.trapezoid is the NumPy 2.0+ replacement for np.trapz
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    signal_power = _trapz(psd_hr[signal_mask], freqs_hr[signal_mask]) \
                   if signal_mask.any() else 0.0

    total_power = _trapz(psd_hr, freqs_hr)

    snr = signal_power / total_power if total_power > 1e-10 else 0.0
    return float(np.clip(snr, 0.0, 1.0)), float(dom_freq)


def _kurtosis_score(bvp: np.ndarray) -> float:
    """
    Map excess kurtosis to a 0–1 quality score.

    Clean BVP signals have kurtosis ~ 1.5–6.0 (peaked waveform).
    Score is highest at kurtosis=3.5 and falls off using a Gaussian
    kernel with σ=2.5.
    """
    k = float(kurtosis(bvp, fisher=True))  # excess kurtosis (normal = 0)

    # Gaussian kernel centered at 3.5 (ideal for pulse-like waveforms)
    # σ=2.5 gives a smooth rolloff — kurtosis 0 scores ~0.37, kurtosis 8 scores ~0.33
    score = np.exp(-((k - 3.5) ** 2) / (2 * 2.5**2))
    return float(np.clip(score, 0.0, 1.0))


def _periodicity_score(
    bvp:     np.ndarray,
    fps:     float,
    dom_freq: float,
    low_hz:  float,
    high_hz: float,
) -> float:
    """
    Normalized autocorrelation at the dominant cardiac period.

    A clean periodic signal (heartbeat) shows strong autocorrelation
    at lag = 1/dominant_frequency.  Noise shows near-zero correlation.
    """
    if dom_freq < low_hz:
        return 0.0

    # Expected lag in samples for one cardiac cycle
    expected_lag = int(round(fps / dom_freq))

    if expected_lag <= 0 or expected_lag >= len(bvp) // 2:
        return 0.0

    # Normalized autocorrelation
    bvp_norm = bvp - np.mean(bvp)
    autocorr_0 = np.dot(bvp_norm, bvp_norm)

    if autocorr_0 < 1e-10:
        return 0.0

    # Correlation at expected cardiac lag
    shifted = np.roll(bvp_norm, -expected_lag)
    # Exclude the wrapped-around portion
    valid_len = len(bvp_norm) - expected_lag
    autocorr_lag = np.dot(bvp_norm[:valid_len], shifted[:valid_len])

    score = autocorr_lag / autocorr_0
    return float(np.clip(score, 0.0, 1.0))


def _empty_sqi() -> dict:
    """Return a zero-quality SQI dict for invalid/empty windows."""
    return {
        "spectral_snr":     0.0,
        "kurtosis_score":   0.0,
        "periodicity":      0.0,
        "composite":        0.0,
        "dominant_freq_hz": 0.0,
        "is_good":          False,
    }