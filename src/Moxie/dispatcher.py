"""
dispatcher.py  —  MOXIE-VASA Pipeline Orchestrator
====================================================
Project : MOXIE-VASA  (Multimodal Video-Based Stress Assessment)
Author  : Durvi Bhati 
Version : 0.5  (personal-video mode — Phase 1)

Overview
--------
The dispatcher is the central coordinator of the MOXIE-VASA processing
pipeline.  It reads video paths from config.yaml, loops over every video
in the personal_videos folder, and routes each one through every active
analysis module in sequence:

    config.yaml → preprocessing → OpenFace → rPPG → (OpenPose: future)

This version (0.5) is scoped to personal videos only so the full
pipeline can be validated end-to-end before the UBFC-Phys dataset
is introduced.  The VideoJob / data_ingestion layer will be wired
back in during Phase 3 when the ML training data is ready.

Pipeline steps
--------------
  1. Preprocessing       (preprocessing.py)
       Applies lighting correction, face-region crop, and codec
       normalization.  Returns the path to a safe processed copy of
       the video that downstream modules can read without modifying
       the original file.

  2. OpenFace analysis   (openface/video_engine.py)
       Calls the OpenFace Docker container via subprocess.
       Extracts per-frame AU intensities + presence (AU01–AU45),
       head pose (Rx/Ry/Rz, Tx/Ty/Tz), and gaze angles.
       Output: a CSV written to Pipeline_Output/result_openface/

  3. rPPG analysis       (rppg/pipeline.py)
       Runs the custom POS-based remote photoplethysmography pipeline.
       Extracts mean HR, SDNN, RMSSD, LF/HF ratio per video.
       Output: a CSV + metrics JSON in Pipeline_Output/result_rppg/

  4. OpenPose            (future — stub commented out)
       Will be activated for the lab dataset once available.

Outputs per video
-----------------
  Pipeline_Output/
  ├── result_openface/   {video_stem}.csv        (AU + pose + gaze, per frame)
  └── result_rppg/       {video_stem}.csv         (HR signal, per window)
                         {video_stem}_metrics.json (HR, HRV summary)

Configuration
-------------
All paths are read from config.yaml at the project root.
No paths are hardcoded in this file.

  Relevant config keys:
    paths.personal_videos  → folder containing your input videos
    paths.openface_output  → where OpenFace CSVs are written
    paths.rppg_output      → where rPPG outputs are written

Usage
-----
  # Run from the src/Moxie directory:
  python dispatcher.py
"""

import argparse
import os
import sys
from pathlib import Path

# ── ensure src/Moxie is importable regardless of working directory ────────────
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_ingestion import load_config
from preprocessing import route_and_preprocess
from openface.video_engine import run_openface_feature_extraction
from rppg.pipeline import run_rppg_analysis
## from openpose.pose_engine import run_openpose_analysis   # Phase 3


# ──────────────────────────────────────────────────────────────
# Valid video extensions
# ──────────────────────────────────────────────────────────────

VALID_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


# ──────────────────────────────────────────────────────────────
# Core: process a single video file
# ──────────────────────────────────────────────────────────────

def process_video(video_path: Path, openface_output: Path, rppg_output: Path) -> dict:
    """
    Run the full pipeline on one video file.

    Parameters
    ----------
    video_path       : absolute path to the input video
    openface_output  : directory where OpenFace CSV will be written
    rppg_output      : directory where rPPG outputs will be written

    Returns
    -------
    result : dict with keys —
        filename, openface_success, rppg_success,
        openface_csv, rppg_metrics, error
    """
    filename = video_path.name
    result = {
        "filename":         filename,
        "openface_success": False,
        "rppg_success":     False,
        "openface_csv":     None,
        "rppg_metrics":     None,
        "error":            None,
    }

    print(f"\n{'='*60}")
    print(f"  PROCESSING: {filename}")
    print(f"  Path : {video_path}")
    print(f"{'='*60}")

    try:
        # ── Step 1: Preprocessing ─────────────────────────────────────
        print("\n>>> [1/3] Preprocessing ...")
        processed_video, metadata = route_and_preprocess(str(video_path))
        print(f"    Preprocessed → {processed_video}")
        result["metadata"] = metadata

        # ── Step 2: OpenFace ──────────────────────────────────────────
        print("\n>>> [2/3] OpenFace feature extraction ...")
        openface_success = run_openface_feature_extraction(
            processed_video,
            str(openface_output)
        )
        result["openface_success"] = openface_success

        if openface_success:
            stem = Path(processed_video).stem
            result["openface_csv"] = str(openface_output / f"{stem}.csv")
            print(f"    OpenFace CSV → {result['openface_csv']}")

        # ── Step 3: rPPG ──────────────────────────────────────────────
        print("\n>>> [3/3] rPPG analysis ...")
        rppg_success = run_rppg_analysis(
            processed_video,
            str(rppg_output),
            ground_truth_bvp=None,   # no ground truth for personal videos
        )
        result["rppg_success"] = rppg_success

        if rppg_success:
            stem = Path(processed_video).stem
            result["rppg_metrics"] = str(rppg_output / f"{stem}_metrics.json")
            print(f"    rPPG metrics → {result['rppg_metrics']}")

        # ── Step 4: OpenPose (future) ─────────────────────────────────
        # print("\n>>> [4/3] OpenPose body pose analysis ...")
        # openpose_success = run_openpose_analysis(processed_video, openpose_output)

        # ── Per-video summary ─────────────────────────────────────────
        if openface_success and rppg_success:
            print(f"\n  [SUCCESS] Pipeline complete for {filename}")
        else:
            print(f"\n  [WARNING] Pipeline finished with errors for {filename}")
            print(f"    OpenFace : {'OK' if openface_success else 'FAILED'}")
            print(f"    rPPG     : {'OK' if rppg_success else 'FAILED'}")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"\n  [CRITICAL ERROR] Pipeline crashed on {filename}: {exc}")

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MOXIE-VASA Dispatcher v0.5")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (defaults to project root)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MOXIE-VASA Dispatcher v0.5  [personal video mode]")
    print("=" * 60)

    # ── Load config ───────────────────────────────────────────────
    cfg = load_config(args.config)

    personal_videos_dir = Path(cfg["paths"]["personal_videos"])
    openface_output     = Path(cfg["paths"]["openface_output"])
    rppg_output         = Path(cfg["paths"]["rppg_output"])

    print(f"  Input  : {personal_videos_dir}")
    print(f"  OpenFace output : {openface_output}")
    print(f"  rPPG output     : {rppg_output}")

    # ── Validate input folder ─────────────────────────────────────
    if not personal_videos_dir.exists():
        print(f"\n[ERROR] Personal video folder not found: {personal_videos_dir}")
        print("  Check data_root and personal_videos in config.yaml")
        sys.exit(1)

    # ── Discover videos ───────────────────────────────────────────
    video_files = sorted([
        f for f in personal_videos_dir.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
        and "_safe" not in f.name        # skip preprocessed copies
        and f.is_file()
    ])

    if not video_files:
        print(f"\n[INFO] No videos found in {personal_videos_dir}")
        print("  Add .mp4 / .avi / .mov / .mkv files to that folder and re-run.")
        sys.exit(0)

    print(f"\n  Found {len(video_files)} video(s) to process:")
    for vf in video_files:
        print(f"    • {vf.name}")

    # ── Process each video ────────────────────────────────────────
    all_results = []
    for video_path in video_files:
        result = process_video(video_path, openface_output, rppg_output)
        all_results.append(result)

    # ── Batch summary ─────────────────────────────────────────────
    total   = len(all_results)
    success = sum(1 for r in all_results if r["openface_success"] and r["rppg_success"])
    failed  = total - success

    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"  Total : {total}   Success : {success}   Failed : {failed}")
    print(f"{'='*60}")

    if failed > 0:
        print("\n  Failed videos:")
        for r in all_results:
            if not (r["openface_success"] and r["rppg_success"]):
                err = r.get("error") or "partial module failure"
                print(f"    {r['filename']} — {err}")

    # Returned so Streamlit app.py can call main() and read results directly
    return all_results


if __name__ == "__main__":
    main()