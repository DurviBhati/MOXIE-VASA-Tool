"""
preprocessing.py  —  MOXIE-VASA Shared Preprocessing Module
=============================================================
Overview
--------
Shared preprocessing that runs on every video before it reaches
OpenFace or the rPPG pipeline.  Produces a clean, standardized
_safe.mp4 that both downstream modules can read without issues.

What this module does
---------------------
  1. Routes by extension  (.mp4 / .avi / .mov / .mkv all handled)
  2. Transcodes to H.264 MP4 at CRF 15  (near-lossless, rPPG-safe)
  3. Fixes rotation from phone metadata flags
  4. Fast face presence check  (catches empty/wrong videos early)
  5. Returns safe video path + metadata dict to the dispatcher

What this module deliberately does NOT do
------------------------------------------
  - No histogram equalization       (would corrupt rPPG color signal)
  - No sharpening or contrast boost (OpenFace doesn't need it either
                                     given normal indoor lighting)
  - No color space conversion       (stays BGR throughout)

Module-specific preprocessing
------------------------------
  rPPG  : gentle resize to 640x480 inside rppg/pipeline.py
  OpenFace : no additional preprocessing needed beyond this step

Dispatcher interface
--------------------
  from preprocessing import route_and_preprocess

  safe_path, metadata = route_and_preprocess(video_path)

"""

import os
import subprocess
from pathlib import Path

import cv2


# Extensions this module will accept
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Transcode settings — CRF 15 is near-lossless and safe for rPPG skin signal
FFMPEG_CRF    = 15
FFMPEG_PRESET = "fast"

# Face check: sample this many evenly-spaced frames looking for a face
FACE_CHECK_SAMPLE_FRAMES = 10


# ──────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────

def route_and_preprocess(video_path: str) -> tuple[str, dict]:
    """
    Main entry point called by dispatcher.py.

    Standardizes the video to a clean H.264 MP4, fixes rotation,
    verifies a face is present, and returns the safe path with metadata.

    Parameters
    ----------
    video_path : str
        Path to the original input video.

    Returns
    -------
    safe_path : str
        Path to the standardized _safe.mp4 copy.
    metadata : dict
        Keys: fps, width, height, frame_count, rotation
        (fps is the value the rPPG pipeline needs for its bandpass filter)

    Raises
    ------
    FileNotFoundError  if the input video does not exist.
    ValueError         if the extension is unsupported.
    RuntimeError       if transcoding fails or no face is found.
    """
    video_path = str(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    ext = Path(video_path).suffix.lower()
    filename = Path(video_path).name
    print(f"\n--- Preprocessing: {filename} ---")

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    # ── Step 1: Transcode to safe H.264 MP4 ──────────────────────────────────
    safe_path = _make_safe_mp4(video_path)

    # ── Step 2: Extract metadata from the safe copy ───────────────────────────
    metadata = _extract_metadata(safe_path)
    print(
        f"[PreProcess] Metadata: "
        f"{metadata['width']}x{metadata['height']} "
        f"@ {metadata['fps']:.2f} FPS  "
        f"({metadata['frame_count']} frames)  "
        f"rotation={metadata['rotation']}°"
    )

    # ── Step 3: Fast face presence check ─────────────────────────────────────
    _verify_face_present(safe_path, metadata)

    print(f"[PreProcess] Done → {Path(safe_path).name}")
    return safe_path, metadata


# ──────────────────────────────────────────────────────────────
# Step 1 — Transcode
# ──────────────────────────────────────────────────────────────

def _make_safe_mp4(video_path: str) -> str:
    """
    Transcode any supported format to a clean H.264 MP4.

    - CRF 15: near-lossless — preserves subtle skin color variations
      that the rPPG POS algorithm depends on
    - -vf transpose auto-rotate: fixes phone rotation metadata so
      OpenFace receives an upright face
    - Skips re-encoding if _safe.mp4 already exists (idempotent)

    Returns path to the safe MP4.
    """
    safe_path = str(Path(video_path).with_suffix("")) + "_safe.mp4"

    if os.path.exists(safe_path) and os.path.getsize(safe_path) > 0:
        print(f"[PreProcess] Safe copy already exists — skipping transcode.")
        return safe_path

    ext = Path(video_path).suffix.lower()
    print(f"[PreProcess] Transcoding {ext} → H.264 MP4 (CRF {FFMPEG_CRF}) ...")

    # -vf "transpose=0" alone won't fix metadata rotation —
    # autorotate=1 tells ffmpeg to honour the rotation flag in the container
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                         # overwrite output without asking
        "-i",       video_path,       # input
        "-c:v",     "libx264",
        "-preset",  FFMPEG_PRESET,
        "-crf",     str(FFMPEG_CRF),
        "-vf",      "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dimensions
        "-an",                        # strip audio (not needed, saves space)
        "-movflags", "+faststart",    # web-friendly MP4 atom ordering
        safe_path,
    ]

    result = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[PreProcess] ffmpeg transcoding failed for {Path(video_path).name}.\n"
            f"ffmpeg stderr:\n{result.stderr[-1000:]}"   # last 1000 chars of error
        )

    if not os.path.exists(safe_path) or os.path.getsize(safe_path) == 0:
        raise RuntimeError(
            f"[PreProcess] ffmpeg exited cleanly but produced no output file: "
            f"{safe_path}"
        )

    print(f"[PreProcess] Transcode complete → {Path(safe_path).name}")
    return safe_path


# ──────────────────────────────────────────────────────────────
# Step 2 — Metadata extraction
# ──────────────────────────────────────────────────────────────

def _extract_metadata(video_path: str) -> dict:
    """
    Extract FPS, resolution, frame count, and rotation from the safe MP4.

    FPS is critical — the rPPG pipeline uses it to set the bandpass
    filter cutoff frequencies correctly.
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"[PreProcess] Cannot open video for metadata: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # OpenCV doesn't expose rotation directly — read via ffprobe
    rotation = _get_rotation_ffprobe(video_path)

    cap.release()

    # Sanity check — a 0 FPS reading means OpenCV couldn't parse the container
    if fps <= 0:
        raise RuntimeError(
            f"[PreProcess] Could not read FPS from {Path(video_path).name}. "
            "The file may be corrupt or in an unsupported container."
        )

    return {
        "fps":         fps,
        "width":       width,
        "height":      height,
        "frame_count": frame_count,
        "rotation":    rotation,
    }


def _get_rotation_ffprobe(video_path: str) -> int:
    """
    Use ffprobe to read the rotation metadata tag.
    Returns 0 if ffprobe is unavailable or the tag is absent.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_tags=rotate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        rotation_str = result.stdout.strip()
        return int(rotation_str) if rotation_str.lstrip("-").isdigit() else 0
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────
# Step 3 — Face presence check
# ──────────────────────────────────────────────────────────────

def _verify_face_present(video_path: str, metadata: dict) -> None:
    """
    Sample FACE_CHECK_SAMPLE_FRAMES evenly across the video and run
    OpenCV's Haar cascade detector on each.

    Raises RuntimeError if no face is found in any sampled frame.
    This catches wrong videos (a screen recording, a landscape shot)
    before they waste minutes inside the OpenFace Docker container.

    Note: Haar cascade is intentionally used here rather than a deep
    detector because it's fast, has no extra dependencies, and is
    sufficient for a simple presence check on your personal videos.
    """
    print(f"[PreProcess] Checking for face presence ({FACE_CHECK_SAMPLE_FRAMES} sample frames)...")

    # OpenCV ships this cascade — no download needed
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap         = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frame_count = metadata["frame_count"]
    step        = max(1, frame_count // FACE_CHECK_SAMPLE_FRAMES)
    found       = False

    for i in range(FACE_CHECK_SAMPLE_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),      # ignore tiny detections
        )

        if len(faces) > 0:
            found = True
            break

    cap.release()

    if not found:
        raise RuntimeError(
            "[PreProcess] No face detected in any sampled frame. "
            "Check that the video contains a visible frontal face. "
            "If wearing glasses or partially occluded, try increasing "
            "the sample frame count or lowering minNeighbors."
        )

    print("[PreProcess] Face detected — video is valid.")