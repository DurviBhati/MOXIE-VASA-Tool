"""
rppg/pos_algorithm.py  —  Plane-Orthogonal-to-Skin (POS) Algorithm
====================================================================
Implements the POS rPPG method from:

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic Principles of Remote PPG.
    IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491.

POS works by projecting the normalized RGB signal onto a plane
orthogonal to the skin-tone vector in a rotating temporal window.
This projection separates the blood-volume-pulse signal from
specular reflections and illumination changes.

Algorithm steps (per window)
-----------------------------
  1. Normalize each channel by its temporal mean  →  C_n
  2. Project onto two orthogonal skin-plane axes:
       S1 =  C_n[R] - C_n[G]
       S2 =  C_n[R] + C_n[G] - 2 * C_n[B]
  3. Scale S2 by the ratio of std(S1) / std(S2)
  4. BVP_window = S1 - α * S2
  5. Overlap-add windows with a Hanning taper to produce
     the full-length BVP waveform

Public interface
----------------
    from rppg.pos_algorithm import apply_pos

    bvp = apply_pos(filtered_rgb, fps=29.97, window_seconds=30)
    # bvp : np.ndarray  shape (N,)   normalized BVP waveform
"""

from __future__ import annotations

import numpy as np


def apply_pos(
    filtered_rgb:   np.ndarray,
    fps:            float,
    window_seconds: float = 30.0,
    step_seconds:   float = 10.0,
) -> np.ndarray:
    """
    Apply the POS algorithm with overlapping Hanning windows.

    Parameters
    ----------
    filtered_rgb   : np.ndarray  shape (N, 3)
                     Detrended + bandpass-filtered RGB signal (float64).
    fps            : float  actual video frame rate
    window_seconds : float  window length in seconds  (default 30)
    step_seconds   : float  step between windows in seconds  (default 10)

    Returns
    -------
    bvp : np.ndarray  shape (N,)
        Full-length BVP waveform, normalized to zero mean unit variance.
        Frames not covered by any window are set to 0.
    """
    n_frames    = filtered_rgb.shape[0]
    window_len  = int(window_seconds * fps)
    step_len    = int(step_seconds   * fps)

    # Clamp window to available signal length
    window_len = min(window_len, n_frames)
    step_len   = max(1, min(step_len, window_len))

    bvp_accum  = np.zeros(n_frames)
    weight_acc = np.zeros(n_frames)
    hann       = np.hanning(window_len)

    start = 0
    while start + window_len <= n_frames:
        end    = start + window_len
        window = filtered_rgb[start:end]          # (window_len, 3)

        bvp_w  = _pos_window(window)              # (window_len,)

        # Taper with Hanning window before overlap-add
        bvp_accum[start:end]  += bvp_w * hann
        weight_acc[start:end] += hann

        start += step_len

    # Handle any remaining frames at the end with a shorter window
    if start < n_frames:
        remaining = filtered_rgb[start:]
        if len(remaining) >= 6:                   # minimum for a stable window
            bvp_w    = _pos_window(remaining)
            hann_r   = np.hanning(len(remaining))
            bvp_accum[start:]  += bvp_w * hann_r
            weight_acc[start:] += hann_r

    # Normalize by overlap weights (avoid division by zero)
    mask = weight_acc > 0
    bvp_accum[mask] /= weight_acc[mask]

    # Z-score normalize the final signal
    std = bvp_accum[mask].std()
    if std > 0:
        bvp_accum[mask] = (bvp_accum[mask] - bvp_accum[mask].mean()) / std

    return bvp_accum


def _pos_window(window: np.ndarray) -> np.ndarray:
    """
    Apply POS to a single temporal window of RGB signal.

    Parameters
    ----------
    window : np.ndarray  shape (T, 3)   one window of RGB (R, G, B columns)

    Returns
    -------
    bvp : np.ndarray  shape (T,)
    """
    # Step 1: Normalize each channel by its temporal mean
    mean_rgb = window.mean(axis=0)                # (3,)

    # Guard against zero mean (e.g. pure black patch)
    mean_rgb = np.where(mean_rgb == 0, 1e-6, mean_rgb)
    C_n = window / mean_rgb                       # (T, 3)  normalized

    R = C_n[:, 0]
    G = C_n[:, 1]
    B = C_n[:, 2]

    # Step 2: Project onto two orthogonal skin-plane axes
    S1 = R - G
    S2 = R + G - 2.0 * B

    # Step 3: Scale factor alpha
    std_s1 = S1.std()
    std_s2 = S2.std()

    if std_s2 < 1e-8:
        # Degenerate window — return zeros rather than NaN
        return np.zeros(len(window))

    alpha = std_s1 / std_s2

    # Step 4: BVP = S1 - alpha * S2
    bvp = S1 - alpha * S2

    return bvp