"""
feature_extractor.py  —  MOXIE-VASA Feature Extraction
=======================================================
Reads OpenFace CSV + rPPG metrics JSON for one subject and
returns a unified feature dict used by stress_inference.py.

Features extracted
------------------
rPPG (physiological):
    hr_mean       — mean heart rate (bpm)
    hr_elevated   — bool: HR > 85 bpm
    rmssd         — HRV metric (ms) — flagged unreliable for now
    lf_hf_ratio   — autonomic balance
    sdnn          — HRV spread (ms)

OpenFace (facial behaviour):
    au04_mean     — brow furrow (concentration/distress)
    au07_mean     — lid tightener (tension)
    au15_mean     — lip corner depressor (negative affect)
    au23_mean     — lip presser (concentration/stress)
    au05_mean     — upper lid raiser (surprise/alertness)
    au12_mean     — lip corner puller (smile — inverse stress)
    pose_rx_std   — head nod variance (motion/restlessness)
    pose_ry_std   — head turn variance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# AU columns we care about (stress-relevant subset)
AU_COLS = {
    "AU04_r": "au04_mean",   # brow furrow — concentration/distress
    "AU07_r": "au07_mean",   # lid tightener — tension
    "AU14_r": "au14_mean",   # dimpler/jaw tension — sustained effort
    "AU15_r": "au15_mean",   # lip corner depressor — negative affect
    "AU23_r": "au23_mean",   # lip presser — concentration/stress
    "AU05_r": "au05_mean",   # upper lid raiser — alertness/surprise
    "AU12_r": "au12_mean",   # lip corner puller — smile (inverse stress)
    "AU09_r": "au09_mean",   # nose wrinkler — disgust/discomfort
    "AU17_r": "au17_mean",   # chin raiser — uncertainty/stress
    "AU01_r": "au01_mean",   # inner brow raiser — concern/surprise
}


def extract_features(
    openface_csv:  str | Path,
    metrics_json:  str | Path,
    subject_id:    str = "unknown",
) -> dict:
    """
    Extract all features for one subject into a flat dict.

    Parameters
    ----------
    openface_csv  : path to OpenFace CSV for this subject
    metrics_json  : path to rPPG metrics JSON for this subject
    subject_id    : label for logging

    Returns
    -------
    dict with keys described in module docstring.
    Returns empty dict if either file is missing.
    """
    openface_csv = Path(openface_csv)
    metrics_json = Path(metrics_json)

    if not openface_csv.exists():
        print(f"[Features] OpenFace CSV not found: {openface_csv}")
        return {}
    if not metrics_json.exists():
        print(f"[Features] Metrics JSON not found: {metrics_json}")
        return {}

    features = {"subject_id": subject_id}

    # ── rPPG features ─────────────────────────────────────────────────────────
    with open(metrics_json) as f:
        metrics = json.load(f)

    hr = metrics.get("hr", {})
    features["hr_mean"]     = float(hr.get("hr_mean", 0))
    features["hr_std"]      = float(hr.get("hr_std", 0))
    features["hr_elevated"] = features["hr_mean"] > 85.0
    features["sdnn"]        = float(hr.get("sdnn", 0))
    features["rmssd"]       = float(hr.get("rmssd", 0))
    features["lf_hf_ratio"] = float(hr.get("lf_hf_ratio", 0))
    features["n_beats"]     = int(hr.get("n_beats_detected", 0))
    features["duration_s"]  = float(hr.get("duration_seconds", 0))

    # rPPG validation results if available
    val = metrics.get("validation", {})
    if isinstance(val, dict) and val.get("valid"):
        features["gt_hr"]   = float(val.get("gt_hr_bpm", 0))
        features["hr_mae"]  = float(val.get("hr_mae", 0))
        features["pearson_r"] = float(val.get("bvp_pearson_r", 0))
    else:
        features["gt_hr"]     = None
        features["hr_mae"]    = None
        features["pearson_r"] = None

    # ── OpenFace features ──────────────────────────────────────────────────────
    df = pd.read_csv(openface_csv)
    df.columns = [c.strip() for c in df.columns]

    # Filter to successful detection frames only
    if "success" in df.columns:
        df = df[df["success"] == 1]
    elif "confidence" in df.columns:
        df = df[df["confidence"] > 0.5]

    if len(df) == 0:
        print(f"[Features] No valid OpenFace frames for {subject_id}")
        return features

    features["n_openface_frames"] = len(df)

    # AU mean intensities
    for col, feat_name in AU_COLS.items():
        if col in df.columns:
            features[feat_name] = round(float(df[col].mean()), 4)
        else:
            features[feat_name] = 0.0

    # Head pose variance — proxy for restlessness/discomfort
    for pose_col, feat_name in [("pose_Rx", "pose_rx_std"),
                                  ("pose_Ry", "pose_ry_std"),
                                  ("pose_Rz", "pose_rz_std")]:
        if pose_col in df.columns:
            features[feat_name] = round(float(df[pose_col].std()), 4)
        else:
            features[feat_name] = 0.0

    features["head_motion_total"] = round(
        features.get("pose_rx_std", 0) +
        features.get("pose_ry_std", 0), 4
    )

    print(f"[Features] {subject_id}: "
          f"HR={features['hr_mean']:.1f} bpm  "
          f"AU04={features.get('au04_mean',0):.2f}  "
          f"AU23={features.get('au23_mean',0):.2f}  "
          f"frames={len(df)}")

    return features


def extract_features_batch(
    openface_dir:  str | Path,
    rppg_dir:      str | Path,
    subject_ids:   list[str],
) -> list[dict]:
    """
    Extract features for a list of subjects.
    Returns a list of feature dicts (one per subject).
    """
    openface_dir = Path(openface_dir)
    rppg_dir     = Path(rppg_dir)
    results      = []

    for sid in subject_ids:
        of_csv  = openface_dir / f"{sid}_vid_safe.csv"
        rp_json = rppg_dir    / f"{sid}_vid_safe_metrics.json"
        feat    = extract_features(of_csv, rp_json, subject_id=sid)
        if feat:
            results.append(feat)

    print(f"[Features] Extracted features for {len(results)}/{len(subject_ids)} subjects")
    return results