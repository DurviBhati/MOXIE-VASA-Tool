"""
rppg/rgb_signal.py  —  Per-Frame RGB Signal Extraction
=======================================================
For each video frame, reads the pixel values inside each valid ROI
patch and computes the spatial mean R, G, B across that patch.

Then combines the four ROI traces into one weighted signal where
the weight of each ROI is proportional to its pixel area — larger
ROIs contribute more because they average over more skin pixels and
are therefore less susceptible to local motion artifacts.

Combined signal shape: (N_frames, 3)  — one row per frame, columns R G B.

This combined signal is what gets passed to the bandpass filter
and then the POS algorithm.  POS needs the cross-channel color
ratios to be preserved — this is why we average spatially within
each ROI (safe) but never mix channels before POS (would destroy
the ratio information POS depends on).

Public interface
----------------
    from rppg.rgb_signal import extract_rgb_signals

    combined, per_roi = extract_rgb_signals(frames, roi_list)
    # combined : np.ndarray  shape (N, 3)   weighted mean across ROIs
    # per_roi  : dict  {roi_name: np.ndarray shape (N, 3)}
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .roi_extractor import ROIBox


# ROIs to use — ordered by typical reliability for rPPG
ROI_NAMES = ["left_cheek", "right_cheek", "forehead", "glabella"]


def extract_rgb_signals(
    frames:   List[np.ndarray],
    roi_list: List[Dict[str, ROIBox]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract mean RGB per ROI per frame and combine into one signal.

    Parameters
    ----------
    frames   : list of BGR uint8 frames (already resized to processing res)
    roi_list : list of ROI dicts, one per frame, from ROIExtractor

    Returns
    -------
    combined : np.ndarray  shape (N_valid, 3)
        Area-weighted mean RGB signal across all valid ROIs.
        Frames where ALL ROIs are invalid are dropped.
    per_roi  : dict  roi_name → np.ndarray shape (N_valid, 3)
        Individual ROI signals for diagnostics / debugging.
    valid_mask : np.ndarray shape (N_frames,) bool
        True for frames that had at least one valid ROI.
    """
    n_frames = min(len(frames), len(roi_list))
    # Pre-allocate storage: one entry per frame, fill as we go
    # Use lists of arrays rather than a dict of lists to avoid None confusion
    all_combined   = []    # will hold one (3,) array per valid frame
    all_per_roi    = {name: [] for name in ROI_NAMES}
    valid_mask = np.zeros(n_frames, dtype=bool)

    for frame_idx in range(n_frames):
        frame_bgr = frames[frame_idx]
        rois      = roi_list[frame_idx]
 
        # Convert BGR → RGB and normalise to [0, 1]
        frame_rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
 
        weighted_sum  = np.zeros(3, dtype=np.float64)
        total_weight  = 0.0
        roi_means     = {}

        for name in ROI_NAMES:
            box = rois.get(name)

            if box is None or not box.valid or box.area == 0:
                roi_means[name] = None
                continue

            # Extract patch and compute spatial mean RGB
            patch = frame_rgb[box.as_slice()]   # shape (h, w, 3)

            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                roi_means[name] = None
                continue

            mean_rgb = patch.mean(axis=(0, 1))  # shape (3,)
            roi_means[name] = mean_rgb
 
            weighted_sum += mean_rgb * box.area
            total_weight += box.area


        # Frame is valid if at least one ROI contributed
        if total_weight > 0:
            combined_frame = weighted_sum / total_weight
            all_combined.append(combined_frame)
            valid_mask[frame_idx] = True
 
            for name in ROI_NAMES:
                if roi_means[name] is not None:
                    all_per_roi[name].append(roi_means[name])
                else:
                    # Fill with the combined value so all arrays stay same length
                    all_per_roi[name].append(combined_frame)
 
    # Convert to numpy arrays
    if len(all_combined) == 0:
        combined_out = np.zeros((0, 3))
        per_roi_out  = {name: np.zeros((0, 3)) for name in ROI_NAMES}
    else:
        combined_out = np.array(all_combined,    dtype=np.float64)  # (N_valid, 3)
        per_roi_out  = {
            name: np.array(all_per_roi[name], dtype=np.float64)
            for name in ROI_NAMES
        }
 
    n_valid = int(valid_mask.sum())
    print(f"[RGB] Extracted {n_valid}/{n_frames} valid frames across "
          f"{len(ROI_NAMES)} ROIs. Signal shape: {combined_out.shape}")
 
    return combined_out, per_roi_out, valid_mask