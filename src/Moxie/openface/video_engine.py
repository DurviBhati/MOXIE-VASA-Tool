"""
openface/video_engine.py  —  MOXIE-VASA OpenFace Module
=========================================================
Project : MOXIE-VASA  (Multimodal Video-Based Stress Assessment)
Author  : Durvi Bhati 
Version : 0.3

Overview
--------
Wraps the OpenFace FeatureExtraction tool running inside a Docker
container.  The dispatcher sends a video path and output directory;
this module handles all Docker mechanics and returns True/False.

The function mounts two host directories into the container:
    /in   ← the folder containing the input video  (read-only)
    /out  ← the output directory for CSVs           (read-write)

OpenFace writes one CSV per video named after the video stem, e.g.
    video_safe.csv

Extracted features
------------------
  -aus      Action Unit intensities (_r) and presence (_c), AU01–AU45
  -pose     Head pose: Rx, Ry, Rz, Tx, Ty, Tz
  -gaze     Gaze angles: gaze_angle_x, gaze_angle_y
  -2Dfp     2D facial landmarks (68 points)
  -3Dfp     3D facial landmarks
  -simalign Similarity-aligned face patches

Docker image
------------
  algebr/openface:latest
  Pull manually if not present:  docker pull algebr/openface:latest

Usage (called by dispatcher.py)
--------------------------------
  from openface.video_engine import run_openface_feature_extraction

  success = run_openface_feature_extraction(
      video_path = "/abs/path/to/video_safe.mp4",
      output_dir = "Pipeline_Output/result_openface"
  )
"""

import os
import subprocess
from pathlib import Path


# Formats OpenFace can reliably decode
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

# Kill Docker if it hasn't finished within this many seconds
# (30-min limit — generous for long videos, prevents silent freezes)
DOCKER_TIMEOUT_SECONDS = 1800


def run_openface_feature_extraction(video_path: str, output_dir: str) -> bool:
    """
    Run OpenFace FeatureExtraction on a single video via Docker.

    Parameters
    ----------
    video_path : str
        Absolute path to the preprocessed input video.
    output_dir : str
        Absolute or relative path to the directory where the CSV
        should be written.  Created automatically if it does not exist.

    Returns
    -------
    bool
        True  — Docker exited cleanly AND a non-empty CSV was produced.
        False — Docker failed, timed out, or produced no usable output.
    """

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    video_filename = video_path.name
    video_folder   = video_path.parent
    extension      = video_path.suffix.lower()

    print(f"[VideoEngine] Input  : {video_filename}  (ext={extension})")
    print(f"[VideoEngine] Output : {output_dir}")

    # ── Guard: unsupported extension ─────────────────────────────────────────
    if extension not in SUPPORTED_EXTENSIONS:
        print(f"[VideoEngine] Unsupported format '{extension}'. "
              f"Supported: {SUPPORTED_EXTENSIONS}")
        return False

    # ── Guard: video file must exist ─────────────────────────────────────────
    if not video_path.exists():
        print(f"[VideoEngine] Video file not found: {video_path}")
        return False

    # ── Ensure output directory exists ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build the Docker command 
    OPENFACE_BINARY = "/home/openface-build/build/bin/FeatureExtraction"

    command = [
        "docker", "run",
        "--rm",                            # delete container when done
        "--platform", "linux/amd64",       # explicit platform for M-chip Macs / WSL
        "--entrypoint", OPENFACE_BINARY,     # override /bin/bash with actual binary
        "-w", "/home/openface-build/build/bin",        # set working dir (binary expects this)
        "-v", f"{video_folder}:/in",       # mount video folder   → /in  (read)
        "-v", f"{output_dir}:/out",        # mount output folder  → /out (write)
        "algebr/openface:latest",          # image  (entrypoint already set inside)
        "-f",        f"/in/{video_filename}",  # input file inside container
        "-out_dir",  "/out",               # output dir inside container
        "-aus",                            # Action Units
        "-pose",                           # head pose
        "-gaze",                           # gaze angles
        "-2Dfp",                           # 2D facial landmarks
        "-3Dfp",                           # 3D facial landmarks
        "-simalign",                       # similarity-aligned patches
    ]

    print(f"[VideoEngine] Spinning up Docker container...")
    print(f"[VideoEngine] Command: {' '.join(command)}")

    # ── Run Docker ────────────────────────────────────────────────────────────
    try:
        # stream=True equivalent: use Popen so stdout prints live instead of
        # buffering silently until the process exits (which can take minutes)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout stream
            text=True,
        )

        # Print each line as it arrives so progress is visible in real time
        for line in process.stdout:
            print(f"  [OpenFace] {line}", end="")

        process.wait(timeout=DOCKER_TIMEOUT_SECONDS)
        returncode = process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"[VideoEngine] Docker timed out after "
              f"{DOCKER_TIMEOUT_SECONDS // 60} minutes — container killed.")
        return False

    except FileNotFoundError:
        print("[VideoEngine] 'docker' command not found. "
              "Is Docker installed and running?")
        return False

    except Exception as exc:
        print(f"[VideoEngine] Unexpected error: {exc}")
        return False

    # ── Verify output ─────────────────────────────────────────────────────────
    if returncode != 0:
        print(f"[VideoEngine] Docker exited with code {returncode}.")
        return False

    # OpenFace names the CSV after the input file stem
    video_stem   = video_path.stem
    expected_csv = output_dir / f"{video_stem}.csv"

    if not expected_csv.exists():
        print(f"[VideoEngine] Docker finished cleanly but no CSV found at: "
              f"{expected_csv}")
        print("[VideoEngine] Possible causes: no face detected, corrupt video, "
              "or codec not supported by the container.")
        return False

    if expected_csv.stat().st_size == 0:
        print(f"[VideoEngine] CSV exists but is empty — "
              "OpenFace found no frames to process.")
        return False

    print(f"[VideoEngine] Success — CSV written to {expected_csv}")
    return True