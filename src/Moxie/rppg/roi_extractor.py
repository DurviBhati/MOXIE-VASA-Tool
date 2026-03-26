"""
rppg/roi_extractor.py  —  ROI Extraction from OpenFace Landmarks
=================================================================
Reads the OpenFace output CSV and derives four ROI bounding boxes
per frame from the 2D facial landmark coordinates:

    Forehead    — derived upward from brow landmarks 19 & 24
    Left cheek  — landmarks 2, 3, 4, 31 (below eye, above jaw)
    Right cheek — landmarks 14, 15, 16, 35 (mirror)
    Glabella    — landmarks 21, 22 (inner brow, between eyes)

OpenFace landmark numbering follows the 68-point iBUG scheme.
Landmark columns in the CSV are named  x_N  and  y_N  (1-indexed).

Public interface
----------------
    from rppg.roi_extractor import ROIExtractor

    extractor = ROIExtractor(openface_csv_path, frame_width, frame_height)
    rois = extractor.get_rois_for_frame(frame_index)
    # rois: dict  {roi_name: (x1, y1, x2, y2)}  clipped to frame bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── ROI landmark definitions (1-indexed, matching OpenFace CSV columns) ───────

# Forehead: derived from outer brow peaks (landmarks 19, 24)
# The forehead box sits above the brow line by FOREHEAD_OFFSET_RATIO
# of the face height, and spans between the two brow peaks horizontally.
FOREHEAD_LANDMARKS  = [19, 24]
FOREHEAD_OFFSET_RATIO = 0.12   # fraction of face height above brows

# Cheeks: polygon anchor points — we take the bounding rect of these
LEFT_CHEEK_LANDMARKS  = [2, 3, 4, 31, 41]
RIGHT_CHEEK_LANDMARKS = [14, 15, 16, 35, 46]

# Glabella: small box centred between inner brow landmarks 21 & 22
GLABELLA_LANDMARKS  = [21, 22]
GLABELLA_BOX_SIZE   = 30   # pixels — fixed size box, scaled later if resized

# Minimum ROI area to be considered valid (pixels²)
MIN_ROI_AREA = 100


@dataclass
class ROIBox:
    """Axis-aligned bounding box for one ROI in one frame."""
    name:  str
    x1:    int
    y1:    int
    x2:    int
    y2:    int
    valid: bool = True   # False if face was not detected in this frame

    @property
    def area(self) -> int:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    def as_slice(self):
        """Return numpy slice tuple (row_slice, col_slice) for frame indexing."""
        return (slice(self.y1, self.y2), slice(self.x1, self.x2))


class ROIExtractor:
    """
    Derives ROI bounding boxes from an OpenFace landmark CSV.

    Parameters
    ----------
    openface_csv : path to the OpenFace output CSV for this video
    frame_width  : pixel width of the video frames
    frame_height : pixel height of the video frames

    Usage
    -----
        extractor = ROIExtractor("result_openface/video_safe.csv", 640, 480)
        rois = extractor.get_rois_for_frame(42)
        # → {"forehead": ROIBox(...), "left_cheek": ROIBox(...), ...}
    """

    def __init__(
        self,
        openface_csv: str | Path,
        frame_width:  int,
        frame_height: int,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self._df          = self._load_csv(openface_csv)
        self._n_frames    = len(self._df)

        # Log what confidence/success values actually look like
        if "confidence" in self._df.columns:
            conf_stats = self._df["confidence"].describe()
            print(f"[ROIExtractor] Loaded {self._n_frames} frames. "
                  f"Confidence: min={conf_stats['min']:.2f} "
                  f"mean={conf_stats['mean']:.2f} "
                  f"max={conf_stats['max']:.2f}")
        else:
            print(f"[ROIExtractor] WARNING: 'confidence' column not found. "
                  f"Available columns (first 10): {list(self._df.columns[:10])}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_rois_for_frame(self, frame_index: int) -> Dict[str, ROIBox]:
        """
        Return all four ROI boxes for a single frame.

        Parameters
        ----------
        frame_index : 0-based index into the video frame sequence

        Returns
        -------
        dict mapping roi_name → ROIBox
        Invalid ROIs (face not detected or too small) have .valid = False.
        """
        if frame_index >= self._n_frames:
            return self._invalid_rois(f"frame {frame_index} out of range")

        row = self._df.iloc[frame_index]

        # Check face detection success — use 'success' column if present,
        # fall back to 'confidence' threshold, then assume valid if neither exists
        if "success" in self._df.columns:
            success = int(row["success"])
            if success == 0:
                return self._invalid_rois("OpenFace success=0")
        elif "confidence" in self._df.columns:
            confidence = float(row["confidence"])
            if confidence < 0.5:
                return self._invalid_rois(f"low confidence: {confidence:.2f}")
        # If neither column present, proceed — don't discard frames unnecessarily
        lm = self._get_landmarks(row)
 
        if not lm:
            return self._invalid_rois("no landmarks found in row")
 
        face_height = self._estimate_face_height(lm)

        return {
            "forehead":   self._forehead_roi(lm, face_height),
            "left_cheek": self._cheek_roi(lm, LEFT_CHEEK_LANDMARKS,  "left_cheek"),
            "right_cheek":self._cheek_roi(lm, RIGHT_CHEEK_LANDMARKS, "right_cheek"),
            "glabella":   self._glabella_roi(lm),
        }

    def get_all_rois(self) -> List[Dict[str, ROIBox]]:
        """
        Return ROI dicts for every frame in the CSV.
        Convenience wrapper around get_rois_for_frame.
        """
        return [self.get_rois_for_frame(i) for i in range(self._n_frames)]

    @property
    def n_frames(self) -> int:
        return self._n_frames

    # ── Private: CSV loading ──────────────────────────────────────────────────

    def _load_csv(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"[ROIExtractor] OpenFace CSV not found: {path}\n"
                "Make sure OpenFace ran successfully before the rPPG pipeline."
            )
        df = pd.read_csv(path)
        # OpenFace adds a space before column names — strip them
        df.columns = [c.strip() for c in df.columns]

        # Also strip any whitespace from string values in key columns
        for col in ["confidence", "success"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
 
        return df

    # ── Private: landmark helpers ─────────────────────────────────────────────

    def _get_landmarks(self, row: pd.Series) -> Dict[int, Tuple[float, float]]:
        """
        Extract all 68 (x, y) landmark positions from one CSV row.
        Returns dict: {landmark_index_1based: (x, y)}
        """
        lm = {}
        for i in range(1, 69):
            x_col = f"x_{i}"
            y_col = f"y_{i}"
            if x_col in row.index and y_col in row.index:
                x_val = row[x_col]
                y_val = row[y_col]
                # Scale landmarks from original resolution to processing resolution
                # OpenFace outputs coords at the original video resolution (1920x1080)
                # but frames have been resized to 640x480
                x_scaled = float(x_val) * (self.frame_width  / 1920.0)
                y_scaled = float(y_val) * (self.frame_height / 1080.0)
                lm[i] = (x_scaled, y_scaled)
        return lm

    def _estimate_face_height(self, lm: Dict[int, Tuple[float, float]]) -> float:
        """
        Estimate face height as vertical distance from chin (landmark 9)
        to the top of the nose bridge (landmark 28).
        Falls back to 20% of frame height if landmarks are missing.
        """
        if 9 in lm and 28 in lm:
            return abs(lm[9][1] - lm[28][1]) * 2.0
        return self.frame_height * 0.20

    # ── Private: ROI constructors ─────────────────────────────────────────────

    def _forehead_roi(self, lm, face_height) -> ROIBox:
        """
        Forehead: bounding rect of landmarks 19 & 24 (outer brow peaks),
        shifted upward by FOREHEAD_OFFSET_RATIO * face_height.
        Height of the box = same offset value.
        """
        pts = [lm[i] for i in FOREHEAD_LANDMARKS if i in lm]
        if not pts:
            return ROIBox("forehead", 0, 0, 0, 0, valid=False)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        offset = face_height * FOREHEAD_OFFSET_RATIO
        x1 = int(min(xs)) - 10          # slight horizontal padding
        x2 = int(max(xs)) + 10
        y2 = int(min(ys)) - 5           # just above brow line
        y1 = int(y2 - offset)           # box extends upward

        return self._clip_and_build("forehead", x1, y1, x2, y2)

    def _cheek_roi(self, lm, indices, name) -> ROIBox:
        """
        Cheek: bounding rect of the given landmark subset with small padding.
        """
        pts = [lm[i] for i in indices if i in lm]
        if not pts:
            return ROIBox(name, 0, 0, 0, 0, valid=False)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        pad = 5
        
        return self._clip_and_build(
            name,
            int(min(xs)) - pad, int(min(ys)) - pad,
            int(max(xs)) + pad, int(max(ys)) + pad,
        )

    def _glabella_roi(self, lm) -> ROIBox:
        """
        Glabella: fixed-size box centred between inner brow landmarks 21 & 22.
        """
        pts = [lm[i] for i in GLABELLA_LANDMARKS if i in lm]
        if not pts:
            return ROIBox("glabella", 0, 0, 0, 0, valid=False)

        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        half = GLABELLA_BOX_SIZE // 2

        return self._clip_and_build(
            "glabella",
            cx - half, cy - half,
            cx + half, cy + half,
        )

    # ── Private: clipping and validation ─────────────────────────────────────

    def _clip_and_build(self, name, x1, y1, x2, y2) -> ROIBox:
        """Clip box to frame bounds and mark invalid if too small."""
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.frame_width,  x2)
        y2 = min(self.frame_height, y2)

        box = ROIBox(name, x1, y1, x2, y2)
        if box.area < MIN_ROI_AREA:
            box.valid = False
        return box

    @staticmethod
    def _invalid_rois(reason: str) -> Dict[str, ROIBox]:
        return {
            name: ROIBox(name, 0, 0, 0, 0, valid=False)
            for name in ("forehead", "left_cheek", "right_cheek", "glabella")
        }