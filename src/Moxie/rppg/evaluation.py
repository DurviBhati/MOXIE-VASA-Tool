"""
rppg/evaluation.py  —  rPPG Validation Against Ground Truth
============================================================
Used when a ground-truth BVP signal is available (UBFC-rPPG).
Computes standard rPPG validation metrics:

  MAE    Mean Absolute Error between estimated and reference HR (bpm)
  RMSE   Root Mean Square Error
  r      Pearson correlation coefficient between BVP waveforms

For personal videos (no ground truth), this module is not called.
The dispatcher passes ground_truth_bvp=None and pipeline.py skips it.

Public interface
----------------
    from rppg.evaluation import evaluate_rppg

    metrics = evaluate_rppg(estimated_hr, bvp_signal,
                            gt_bvp_csv_path, fps)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import resample


def evaluate_rppg(
    estimated_hr_bpm:   float,
    estimated_bvp:      np.ndarray,
    gt_bvp_csv_path:    str,
    fps:                float,
    gt_format:          str = "ubfc_rppg",
) -> dict:
    """
    Compare estimated rPPG output against UBFC-rPPG ground truth.

    Parameters
    ----------
    estimated_hr_bpm : float   mean HR estimate from hr_estimation.py
    estimated_bvp    : np.ndarray  shape (N,)  normalized BVP waveform
    gt_bvp_csv_path  : str  path to the UBFC-Phys BVP CSV for this subject
    fps              : float  video frame rate (for BVP resampling)

    Returns
    -------
    dict with keys: hr_mae, hr_rmse, bvp_pearson_r, bvp_pearson_p,
                    gt_hr_bpm, estimated_hr_bpm, valid (bool)
    """
    result = {
        "valid":          False,
        "estimated_hr_bpm": estimated_hr_bpm,
        "gt_hr_bpm":      None,
        "hr_mae":         None,
        "hr_rmse":        None,
        "bvp_pearson_r":  None,
        "bvp_pearson_p":  None,
        "error":          None,
        "gt_format":      gt_format,
    }

    try:
        if gt_format == "ubfc_rppg":
            lines = Path(gt_bvp_csv_path).read_text().strip().split('\n')
            # Row 0 = BVP, Row 1 = HR per frame, Row 2 = timestamps
            gt_bvp = np.array([float(v) for v in lines[0].split()], dtype=np.float64)
            hr_values = [float(v) for v in lines[1].split()]
            gt_hr_bpm = float(np.mean(hr_values))
        else:
            gt_bvp, gt_fps = _load_ubfc_bvp(gt_bvp_csv_path, gt_format=gt_format)
            gt_hr_bpm = _hr_from_bvp(gt_bvp, gt_fps)

        result["gt_hr_bpm"] = round(float(gt_hr_bpm), 2)
        result["hr_mae"] = round(abs(estimated_hr_bpm - gt_hr_bpm), 2)
        result["hr_rmse"]   = round(
            float(np.sqrt((estimated_hr_bpm - gt_hr_bpm) ** 2)), 2
        )

        # Pearson r between BVP waveforms (resample gt to match estimated length)
        if len(estimated_bvp) > 10 and len(gt_bvp) > 10:
            gt_resampled = resample(gt_bvp, len(estimated_bvp))
            r, p = pearsonr(estimated_bvp, gt_resampled)
            result["bvp_pearson_r"] = round(float(r), 4)
            result["bvp_pearson_p"] = round(float(p), 6)

        result["valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _load_ubfc_bvp(csv_path: str, gt_format: str = "ubfc_rppg") -> tuple[np.ndarray, float]:
    """
    Load UBFC BVP ground truth.

    Supports two formats:
      - "ubfc_rppg":  ground_truth.txt with space-separated float values
                      (one long row or multiple rows of space-separated values).
                      Sampling rate = same as video fps (passed externally).
      - "ubfc_phys":  CSV with two columns [bvp_value, sampling_rate].
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth BVP not found: {csv_path}")

    if gt_format == "ubfc_rppg":
        lines = path.read_text().strip().split('\n')
        # Row 0 = BVP waveform, Row 1 = HR per frame, Row 2 = timestamps
        hr_values = [float(v) for v in lines[1].split()]
        gt_hr = np.mean(hr_values)
        # Return BVP from row 0 for waveform correlation, HR directly
        bvp = np.array([float(v) for v in lines[0].split()], dtype=np.float64)
        return bvp, fps, gt_hr  # return HR directly, skip _hr_from_bvp

    else:
        # UBFC-Phys CSV format: [bvp_value, sampling_rate]
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] == 2:
            bvp = df.iloc[:, 0].values.astype(float)
            gt_fps = float(df.iloc[0, 1])
        else:
            bvp = df.iloc[:, 0].values.astype(float)
            gt_fps = 64.0
        return bvp, gt_fps


def _hr_from_bvp(bvp: np.ndarray, fps: float) -> float:
    """
    Estimate HR from a BVP waveform using Welch PSD.
    Returns the dominant frequency in bpm.
    """
    from scipy.signal import welch

    freqs, psd = welch(bvp, fs=fps, nperseg=min(len(bvp), 512))

    # Look only in the physiological HR range
    hr_mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not hr_mask.any():
        return 0.0

    dominant_freq = freqs[hr_mask][np.argmax(psd[hr_mask])]
    return dominant_freq * 60.0