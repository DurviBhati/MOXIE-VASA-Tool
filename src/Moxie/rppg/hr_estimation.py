"""
rppg/hr_estimation.py  —  Heart Rate and HRV Feature Extraction
================================================================
Takes the BVP waveform from POS and extracts:

  HR metrics
  ----------
  hr_mean      Mean heart rate over the video  (bpm)
  hr_std       Standard deviation of HR across windows

  HRV time-domain metrics  (computed from inter-beat intervals)
  --------------------------------
  sdnn         Standard deviation of NN intervals  (ms)
  rmssd        Root mean square of successive differences  (ms)
                — primary marker of parasympathetic activity,
                  drops under acute stress

  HRV frequency-domain  (Welch PSD on NN intervals)
  --------------------------------------------------
  lf_power     Low-frequency power  (0.04–0.15 Hz)
  hf_power     High-frequency power (0.15–0.40 Hz)
  lf_hf_ratio  LF/HF ratio — elevates under sympathetic stress activation

All of these feed into feature_extractor.py as stress-relevant features.

Public interface
----------------
    from rppg.hr_estimation import estimate_hr_features

    features = estimate_hr_features(bvp, fps=29.97)
    # features: dict with hr_mean, hr_std, sdnn, rmssd, lf_power,
    #           hf_power, lf_hf_ratio, n_beats_detected
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch


def estimate_hr_features(
    bvp:            np.ndarray,
    fps:            float,
    window_seconds: float = 30.0,
    step_seconds:   float = 10.0,
    min_duration_s: float = 30.0,
) -> dict:
    """
    Extract HR and HRV features from a BVP waveform.

    Parameters
    ----------
    bvp            : np.ndarray  shape (N,)   normalized BVP from POS
    fps            : float  actual video frame rate
    window_seconds : float  window for per-window HR  (default 30s)
    step_seconds   : float  step between windows      (default 10s)
    min_duration_s : float  minimum signal length for HRV features

    Returns
    -------
    dict with all HR/HRV features, plus a 'warnings' list for any
    features that could not be computed reliably.
    """
    duration_s = len(bvp) / fps
    warnings   = []

    features = {
        "hr_mean":      None,
        "hr_std":       None,
        "sdnn":         None,
        "rmssd":        None,
        "lf_power":     None,
        "hf_power":     None,
        "lf_hf_ratio":  None,
        "n_beats_detected": 0,
        "duration_seconds": round(duration_s, 2),
        "warnings":     warnings,
    }

    if duration_s < min_duration_s:
        warnings.append(
            f"Video too short for reliable HRV: {duration_s:.1f}s "
            f"(minimum {min_duration_s}s recommended)"
        )

    # ── Step 1: Detect peaks (heartbeats) ────────────────────────────────────
    peaks, peak_props = _detect_peaks(bvp, fps)
    features["n_beats_detected"] = len(peaks)

    if len(peaks) < 4:
        warnings.append(
            f"Only {len(peaks)} peaks detected — "
            "too few for reliable HR/HRV. "
            "Check ROI quality and signal length."
        )
        return features

    # ── Step 2: HR from peak intervals ───────────────────────────────────────
    rr_intervals_frames = np.diff(peaks)                  # frame counts
    rr_intervals_ms     = (rr_intervals_frames / fps) * 1000.0  # milliseconds
    hr_per_beat         = 60000.0 / rr_intervals_ms       # bpm per interval

    features["hr_mean"] = round(float(hr_per_beat.mean()), 2)
    features["hr_std"]  = round(float(hr_per_beat.std()),  2)

    # ── Step 3: Time-domain HRV ───────────────────────────────────────────────
    # Filter physiologically implausible intervals (< 300ms or > 2000ms)
    valid_rr = rr_intervals_ms[
        (rr_intervals_ms > 300) & (rr_intervals_ms < 2000)
    ]

    if len(valid_rr) < 3:
        warnings.append("Too few valid RR intervals for HRV — possible artefact.")
        return features

    features["sdnn"]  = round(float(valid_rr.std()), 2)
    features["rmssd"] = round(
        float(np.sqrt(np.mean(np.diff(valid_rr) ** 2))), 2
    )

    # ── Step 4: Frequency-domain HRV (Welch on RR series) ────────────────────
    if len(valid_rr) >= 10:
        lf, hf, ratio = _compute_lf_hf(valid_rr)
        features["lf_power"]    = round(float(lf),    4)
        features["hf_power"]    = round(float(hf),    4)
        features["lf_hf_ratio"] = round(float(ratio), 4)
    else:
        warnings.append(
            f"Only {len(valid_rr)} RR intervals — "
            "frequency-domain HRV requires at least 10."
        )

    return features


# ── Private helpers ───────────────────────────────────────────────────────────

def _detect_peaks(bvp: np.ndarray, fps: float):
    """
    Detect systolic peaks in the BVP signal.

    Minimum peak distance set to the frame equivalent of 40 bpm (1.5s)
    to prevent detecting noise as heartbeats.
    """
    min_distance = int(fps * 0.4)   # ~150 bpm max = 0.4s between peaks
    min_height   = bvp.std() * 0.3  # peaks must be at least 0.3 SD above mean

    peaks, props = find_peaks(
        bvp,
        distance=max(1, min_distance),
        height=bvp.mean() + min_height,
        prominence=bvp.std() * 0.2,
    )
    return peaks, props


def _compute_lf_hf(rr_ms: np.ndarray) -> tuple[float, float, float]:
    """
    Compute LF and HF power from an RR interval series using Welch PSD.

    RR intervals are resampled to 4 Hz before spectral analysis
    (standard in HRV literature).

    LF band: 0.04–0.15 Hz
    HF band: 0.15–0.40 Hz
    """
    # Resample RR series to evenly spaced 4 Hz signal
    resample_hz = 4.0
    cumulative_t = np.cumsum(rr_ms) / 1000.0   # seconds
    duration     = cumulative_t[-1]
    t_uniform    = np.arange(0, duration, 1.0 / resample_hz)
    rr_resampled = np.interp(t_uniform, cumulative_t, rr_ms)

    # Welch PSD
    freqs, psd = welch(
        rr_resampled,
        fs=resample_hz,
        nperseg=min(len(rr_resampled), 256),
    )

    freq_res = freqs[1] - freqs[0]   # Hz per bin

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0

    ratio = lf_power / hf_power if hf_power > 1e-10 else 0.0

    return lf_power, hf_power, ratio