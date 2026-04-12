"""
feature_fusion.py  —  MOXIE-VASA Feature Fusion
================================================
Takes the list of per-subject feature dicts from feature_extractor.py
and merges them into a single pandas DataFrame — one row per subject.

This is the master feature matrix used by:
  - stress_inference.py  (rule-based scoring)
  - The analysis notebook (visualisation and reporting)
  - Future ML classifier (Phase 3)

Column groups in the output DataFrame
--------------------------------------
  subject_id          — identifier
  hr_mean, hr_std     — rPPG heart rate
  hr_elevated         — bool flag
  sdnn, rmssd         — HRV (flagged unreliable Phase 1)
  lf_hf_ratio         — autonomic balance
  au04_mean ..        — AU intensities (stress-relevant subset)
  pose_rx_std ..      — head motion variance
  head_motion_total   — combined motion proxy
  gt_hr, hr_mae       — validation metrics (UBFC subjects only)
  duration_s          — video duration
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


# Column order for the output DataFrame
COLUMN_ORDER = [
    # Identity
    "subject_id",
    # rPPG — heart rate
    "hr_mean", "hr_std", "hr_elevated",
    # rPPG — HRV (unreliable Phase 1, kept for transparency)
    "sdnn", "rmssd", "lf_hf_ratio",
    # rPPG — signal quality
    "n_beats", "duration_s",
    # OpenFace — stress-relevant AUs
    "au04_mean", "au07_mean", "au15_mean", "au23_mean",
    "au05_mean", "au09_mean", "au12_mean", "au17_mean",
    # OpenFace — head motion
    "pose_rx_std", "pose_ry_std", "pose_rz_std", "head_motion_total",
    # OpenFace — detection quality
    "n_openface_frames",
    # Validation (UBFC subjects only)
    "gt_hr", "hr_mae", "pearson_r",
]


def fuse_features(features_list: list[dict]) -> pd.DataFrame:
    """
    Merge a list of feature dicts into a single DataFrame.

    Parameters
    ----------
    features_list : list of dicts from feature_extractor.extract_features()

    Returns
    -------
    pd.DataFrame — one row per subject, columns in COLUMN_ORDER.
    Missing values filled with NaN.
    """
    if not features_list:
        return pd.DataFrame(columns=COLUMN_ORDER)

    df = pd.DataFrame(features_list)

    # Ensure all expected columns exist (fill missing with NaN)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns — keep any extra columns at the end
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[COLUMN_ORDER + extra_cols]

    # Cast types
    bool_cols = ["hr_elevated"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    numeric_cols = [c for c in COLUMN_ORDER
                    if c not in ["subject_id"] + bool_cols]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[Fusion] Feature matrix: {df.shape[0]} subjects × {df.shape[1]} features")
    return df


def save_feature_matrix(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the feature matrix to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Fusion] Saved feature matrix → {output_path}")


def load_feature_matrix(path: str | Path) -> pd.DataFrame:
    """Load a previously saved feature matrix CSV."""
    return pd.read_csv(path)