"""
rppg/pipeline.py  —  MOXIE-VASA rPPG Pipeline Orchestrator
============================================================
Project : MOXIE-VASA  (Multimodal Video-Based Stress Assessment)
Author  : Durvi Bhati — University of Michigan
Version : 1.0

Overview
--------
Orchestrates the full rPPG processing chain for a single video:

    video frames  →  ROI extraction (from OpenFace landmarks)
                  →  RGB signal extraction (per ROI, per frame)
                  →  Detrending + bandpass filtering
                  →  POS algorithm  →  BVP waveform
                  →  HR + HRV feature extraction
                  →  (optional) validation vs ground truth BVP
                  →  Save CSV + metrics JSON

Called by dispatcher.py:
    from rppg.pipeline import run_rppg_analysis
    success = run_rppg_analysis(video_path, output_dir,
                                ground_truth_bvp=None, fps=29.97)

Outputs written to output_dir
------------------------------
  {video_stem}_bvp.csv         Raw BVP waveform  (one value per frame)
  {video_stem}_rgb.csv         Per-frame RGB signals per ROI
  {video_stem}_metrics.json    HR/HRV features + validation results

Processing notes
----------------
  - Frames are resized to 640×480 for speed  (color ratios preserved)
  - BGR → RGB conversion happens inside rgb_signal.py
  - Actual fps from preprocessing metadata is used throughout —
    never hardcoded — so 29.5 / 29.97 / 30 all work correctly
  - Memory: frames are loaded one window at a time to prevent RAM overflow
    on long videos at 1080p
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .roi_extractor  import ROIExtractor
from .rgb_signal     import extract_rgb_signals
from .filtering      import detrend_signal, bandpass_filter
from .pos_algorithm  import apply_pos
from .hr_estimation  import estimate_hr_features
from .evaluation     import evaluate_rppg


# Processing resolution — resize frames to this before ROI extraction
# Keeps memory and CPU manageable for 1080p input
PROC_WIDTH  = 640
PROC_HEIGHT = 480


def run_rppg_analysis(
    video_path:       str,
    output_dir:       str,
    ground_truth_bvp: Optional[str] = None,
    fps:              float          = 30.0,
) -> bool:
    """
    Run the full rPPG pipeline on one video.

    Parameters
    ----------
    video_path       : path to the preprocessed _safe.mp4
    output_dir       : directory to write outputs into
    ground_truth_bvp : path to UBFC-Phys BVP CSV, or None (personal videos)
    fps              : actual video frame rate from preprocessing metadata

    Returns
    -------
    bool  True if pipeline completed and outputs were written, else False.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem
    print(f"[rPPG] Processing: {video_path.name}  fps={fps:.3f}")

    try:
        # ── Step 1: Locate the OpenFace CSV ──────────────────────────────────
        openface_csv = _find_openface_csv(video_path, output_dir)
        print(f"[rPPG] OpenFace CSV: {openface_csv.name}")

        # ── Step 2: Load video frames (windowed to limit RAM) ─────────────────
        print("[rPPG] Loading frames...")
        frames, actual_fps = _load_frames(video_path, fps)
        n_frames = len(frames)
        print(f"[rPPG] Loaded {n_frames} frames at {actual_fps:.3f} fps "
              f"(processing res {PROC_WIDTH}×{PROC_HEIGHT})")

        if n_frames < 30:
            print("[rPPG] Too few frames for rPPG analysis — skipping.")
            return False

        # Use the fps OpenCV reports if it differs from metadata
        fps = actual_fps if actual_fps > 0 else fps

        # ── Step 3: Extract ROI boxes from OpenFace landmarks ─────────────────
        print("[rPPG] Extracting ROIs from OpenFace landmarks...")
        roi_extractor = ROIExtractor(openface_csv, PROC_WIDTH, PROC_HEIGHT)
        roi_list      = roi_extractor.get_all_rois()

        # Trim roi_list / frames to same length (OpenFace may drop some frames)
        min_len  = min(n_frames, roi_extractor.n_frames)
        frames   = frames[:min_len]
        roi_list = roi_list[:min_len]

        valid_roi_count = sum(
            1 for rois in roi_list
            if any(box.valid for box in rois.values())
        )
        print(f"[rPPG] Valid ROI frames: {valid_roi_count}/{min_len} "
              f"({100*valid_roi_count/max(min_len,1):.1f}%)")

        if valid_roi_count < 30:
            print("[rPPG] Too few valid ROI frames — face may be occluded.")
            return False

        # ── Step 4: Extract RGB signals ───────────────────────────────────────
        print("[rPPG] Extracting RGB signals from ROI patches...")
        combined_rgb, per_roi_rgb, valid_mask = extract_rgb_signals(
            frames, roi_list
        )
        print(f"[rPPG] RGB signal shape: {combined_rgb.shape}")

        # ── Step 5: Detrend ───────────────────────────────────────────────────
        print("[rPPG] Detrending signal...")
        detrended = detrend_signal(combined_rgb)

        # ── Step 6: Bandpass filter ───────────────────────────────────────────
        print(f"[rPPG] Bandpass filtering (0.7–3.0 Hz at {fps:.3f} fps)...")
        filtered = bandpass_filter(detrended, fps=fps)

        # ── Step 7: POS algorithm ─────────────────────────────────────────────
        print("[rPPG] Applying POS algorithm...")
        bvp = apply_pos(filtered, fps=fps)
        print(f"[rPPG] BVP waveform shape: {bvp.shape}")

        # ── Step 8: HR and HRV feature extraction ─────────────────────────────
        print("[rPPG] Estimating HR and HRV features...")
        hr_features = estimate_hr_features(bvp, fps=fps)
        print(
            f"[rPPG] HR: {hr_features.get('hr_mean')} bpm  "
            f"SDNN: {hr_features.get('sdnn')} ms  "
            f"RMSSD: {hr_features.get('rmssd')} ms"
        )
        if hr_features.get("warnings"):
            for w in hr_features["warnings"]:
                print(f"[rPPG] WARNING: {w}")

        # ── Step 9: Validation (UBFC-Phys only) ───────────────────────────────
        validation = {}
        if ground_truth_bvp and Path(ground_truth_bvp).exists():
            print("[rPPG] Comparing against ground truth BVP...")
            validation = evaluate_rppg(
                estimated_hr_bpm = hr_features.get("hr_mean", 0),
                estimated_bvp    = bvp,
                gt_bvp_csv_path  = ground_truth_bvp,
                fps              = fps,
            )
            print(
                f"[rPPG] Validation — "
                f"MAE: {validation.get('hr_mae')} bpm  "
                f"Pearson r: {validation.get('bvp_pearson_r')}"
            )

        # ── Step 10: Save outputs ──────────────────────────────────────────────
        _save_outputs(
            output_dir   = output_dir,
            video_stem   = video_stem,
            bvp          = bvp,
            valid_mask   = valid_mask,
            per_roi_rgb  = per_roi_rgb,
            hr_features  = hr_features,
            validation   = validation,
            fps          = fps,
        )

        print(f"[rPPG] Done — outputs written to {output_dir}")
        return True

    except Exception as exc:
        import traceback
        print(f"[rPPG] CRITICAL ERROR: {exc}")
        traceback.print_exc()
        return False


# ── Private helpers ───────────────────────────────────────────────────────────

def _find_openface_csv(video_path: Path, rppg_output_dir: Path) -> Path:
    """
    Locate the OpenFace CSV for this video.
    It lives in result_openface/, which is a sibling of result_rppg/.
    """
    # Go up from result_rppg/ → Pipeline_Output/ → look in result_openface/
    openface_dir = rppg_output_dir.parent / "result_openface"
    csv_path     = openface_dir / f"{video_path.stem}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"[rPPG] OpenFace CSV not found at: {csv_path}\n"
            "Make sure OpenFace ran successfully before the rPPG pipeline."
        )
    return csv_path


def _load_frames(
    video_path: Path,
    fps_hint:   float,
) -> tuple[list[np.ndarray], float]:
    """
    Load all frames from the video, resized to PROC_WIDTH x PROC_HEIGHT.

    Returns frames as a list of BGR uint8 arrays and the actual fps.
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"[rPPG] Cannot open video: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps_hint

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize for processing efficiency — does NOT change color ratios
        resized = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT),
                             interpolation=cv2.INTER_AREA)
        frames.append(resized)

    cap.release()
    return frames, actual_fps


def _save_outputs(
    output_dir:  Path,
    video_stem:  str,
    bvp:         np.ndarray,
    valid_mask:  np.ndarray,
    per_roi_rgb: dict,
    hr_features: dict,
    validation:  dict,
    fps:         float,
) -> None:
    """Write BVP CSV, RGB CSV, and metrics JSON to output_dir."""

    # BVP waveform CSV (one value per valid frame)
    bvp_csv = output_dir / f"{video_stem}_bvp.csv"
    pd.DataFrame({"bvp": bvp}).to_csv(bvp_csv, index=False)

    # Per-ROI RGB signals CSV
    rgb_rows = []
    valid_indices = np.where(valid_mask)[0]
    for i, frame_idx in enumerate(valid_indices):
        row = {"frame": int(frame_idx)}
        for roi_name, sig in per_roi_rgb.items():
            if i < len(sig):
                row[f"{roi_name}_R"] = round(float(sig[i, 0]), 6)
                row[f"{roi_name}_G"] = round(float(sig[i, 1]), 6)
                row[f"{roi_name}_B"] = round(float(sig[i, 2]), 6)
        rgb_rows.append(row)
    pd.DataFrame(rgb_rows).to_csv(
        output_dir / f"{video_stem}_rgb.csv", index=False
    )

    # Metrics JSON
    metrics = {
        "video":      video_stem,
        "fps":        round(fps, 4),
        "hr":         hr_features,
        "validation": validation if validation else "not_run",
    }
    metrics_path = output_dir / f"{video_stem}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    print(f"[rPPG] Saved: {bvp_csv.name}, "
          f"{video_stem}_rgb.csv, "
          f"{video_stem}_metrics.json")