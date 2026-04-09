"""
data_ingestion.py  —  MOXIE-VASA Data Ingestion Layer
======================================================
Responsibilities
----------------
1. Load and resolve all paths from config.yaml
2. Provide a uniform VideoJob dataclass that dispatcher.py consumes
   regardless of source:
     - Personal videos  (absolute paths from config)
     - UBFC-rPPG        (download-and-purge via gdown, rPPG validation)
     - UBFC-Phys        (future — AWS S3 via IEEE DataPort credentials)
     - COHFACE          (future — API access)
3. Download individual UBFC-rPPG videos on demand, yield the VideoJob,
   then delete the raw .avi to keep disk usage low (~600 MB peak per video)

Public interface (what dispatcher.py calls)
-------------------------------------------
    from data_ingestion import load_config, iter_personal_videos,
                               iter_ubfc_rppg, VideoJob

    cfg = load_config()
    for job in iter_personal_videos(cfg):
        dispatcher_process(job)
    for job in iter_ubfc_rppg(cfg):
        dispatcher_process(job)

Dataset notes
-------------
  UBFC-rPPG : 49 subjects, vid.avi + ground_truth.txt per folder
              Ground truth = single row of space-separated BVP values at 30 Hz
              No stress labels — used for rPPG accuracy validation only
              (MAE and Pearson r vs BVP ground truth)

  UBFC-Phys : 56 subjects, TSST stress protocol, BVP + EDA ground truth
              Access via IEEE DataPort + AWS S3 (credentials required)
              Used for ML stress classifier training in Phase 3
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import yaml


# ──────────────────────────────────────────────────────────────
# 1.  Config loader
# ──────────────────────────────────────────────────────────────

def load_config(config_path=None) -> dict:
    """
    Load config.yaml and resolve all ${data_root} placeholders.

    Parameters
    ----------
    config_path : path to config.yaml.
        Defaults to <project_root>/config.yaml (two levels up from this file).

    Returns
    -------
    dict — fully resolved config with no unexpanded variables.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}.\n"
            "Copy config.template.yaml to config.yaml and set data_root."
        )

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    cfg = _resolve_placeholders(cfg)
    _ensure_output_dirs(cfg)
    return cfg


def _resolve_placeholders(cfg: dict) -> dict:
    """Replace ${data_root} in every string value."""
    data_root = cfg["paths"]["data_root"]

    def _expand(value):
        if isinstance(value, str):
            return value.replace("${data_root}", data_root)
        if isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_expand(i) for i in value]
        return value

    cfg["paths"] = _expand(cfg["paths"])
    return cfg


def _ensure_output_dirs(cfg: dict) -> None:
    """Create all pipeline output directories if they don't exist."""
    for key in ["output_root", "openface_output", "rppg_output",
                "openpose_output", "features_output"]:
        path = Path(cfg["paths"][key])
        path.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 2.  VideoJob  —  the contract between ingestion and dispatcher
# ──────────────────────────────────────────────────────────────

@dataclass
class VideoJob:
    """
    One unit of work for the dispatcher.

    Dispatcher never needs to know WHERE the video came from —
    it just processes VideoJob objects and calls the right modules.

    Attributes
    ----------
    video_path      : absolute path to the video file on disk
    subject_id      : string identifier, e.g. "personal_001" or "ubfc_rppg_s01"
    condition       : "stress" | "rest" | "unknown"
    source          : "personal" | "ubfc_rppg" | "ubfc_phys" | "cohface"
    ground_truth_bvp: path to ground truth BVP file if available, else None
    label           : int  1=stress  0=rest  -1=unknown
    metadata        : any extra info (fps hint, dataset name, etc.)
    _delete_after   : if True, job.cleanup() deletes the video after processing
                      — set True only for downloaded dataset videos
    """
    video_path:       Path
    subject_id:       str
    condition:        str
    source:           str
    ground_truth_bvp: Optional[Path] = None
    label:            int             = -1
    metadata:         dict            = field(default_factory=dict)
    _delete_after:    bool            = False

    def __post_init__(self):
        self.video_path = Path(self.video_path)
        if self.ground_truth_bvp is not None:
            self.ground_truth_bvp = Path(self.ground_truth_bvp)

    def cleanup(self) -> None:
        """
        Delete the raw video file after feature extraction.
        Only acts if _delete_after=True. Safe to call unconditionally —
        personal videos are never deleted.
        """
        if self._delete_after and self.video_path.exists():
            os.remove(self.video_path)
            print(f"  [CLEANUP] Deleted raw video: {self.video_path.name}")


# ──────────────────────────────────────────────────────────────
# 3.  Personal video iterator
# ──────────────────────────────────────────────────────────────

VALID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def iter_personal_videos(cfg: dict) -> Generator[VideoJob, None, None]:
    """
    Yield VideoJob for each video in the personal_videos directory.

    Labels are unknown (-1) — personal videos have no ground truth.
    Videos are never deleted after processing.

    Yields
    ------
    VideoJob with source="personal", label=-1, condition="unknown"
    """
    folder = Path(cfg["paths"]["personal_videos"])

    if not folder.exists():
        print(f"[INGESTION] Personal video folder not found: {folder}")
        print("  Create the folder and add videos, or update config.yaml.")
        return

    video_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
        and "_safe" not in f.name
        and f.is_file()
    ])

    if not video_files:
        print(f"[INGESTION] No videos found in {folder}")
        return

    print(f"[INGESTION] Found {len(video_files)} personal video(s)")

    for idx, video_path in enumerate(video_files, start=1):
        subject_id = f"personal_{idx:03d}"
        print(f"  → {subject_id}: {video_path.name}")

        yield VideoJob(
            video_path       = video_path,
            subject_id       = subject_id,
            condition        = "unknown",
            source           = "personal",
            ground_truth_bvp = None,
            label            = -1,
            metadata         = {"original_filename": video_path.name},
            _delete_after    = False,
        )


# ──────────────────────────────────────────────────────────────
# 4.  UBFC-rPPG iterator  (download → yield → purge)
# ──────────────────────────────────────────────────────────────

def iter_ubfc_rppg(cfg: dict) -> Generator[VideoJob, None, None]:
    """
    For each subject in the configured UBFC-rPPG subset:
      1. Download the subject folder from Google Drive using gdown
      2. Locate vid.avi and ground_truth.txt
      3. Yield a VideoJob with ground_truth_bvp set
      4. After dispatcher calls job.cleanup(), delete vid.avi
         (ground_truth.txt is kept — it's tiny and needed for evaluation)

    Ground truth format
    -------------------
    ground_truth.txt contains a single row of space-separated BVP values
    at 30 Hz, matching the video frame rate.
    evaluation.py reads this with np.fromstring(text, sep=' ').

    No stress labels — UBFC-rPPG is used purely to validate rPPG accuracy
    (MAE in bpm and Pearson r between estimated and reference BVP).

    Yields
    ------
    VideoJob with source="ubfc_rppg", label=-1, ground_truth_bvp set
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for UBFC-rPPG download.\n"
            "Install it with:  pip install gdown"
        )

    ubfc_cfg   = cfg.get("ubfc_rppg", {})
    folder_url = ubfc_cfg.get("drive_folder_url", "")
    subjects   = ubfc_cfg.get("subjects", list(range(1, 11)))
    keep_raw   = ubfc_cfg.get("keep_raw_video", False)

    raw_dir = Path(cfg["paths"]["data_root"]) / "ubfc_rppg" / "raw"
    gt_dir  = Path(cfg["paths"]["data_root"]) / "ubfc_rppg" / "ground_truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    if not folder_url:
        raise ValueError(
            "ubfc_rppg.drive_folder_url not set in config.yaml.\n"
            "Add the Google Drive folder URL for the UBFC-rPPG dataset."
        )

    total = len(subjects)

    for i, subject_num in enumerate(subjects, start=1):
        subject_id  = f"subject{subject_num}"
        tag         = f"ubfc_rppg_{subject_id}"
        video_path  = raw_dir / subject_id / "vid.avi"
        gt_path     = gt_dir  / subject_id / "ground_truth.txt"

        print(f"\n[UBFC-rPPG] ({i}/{total})  {subject_id}")

        # ── Download ground truth (tiny — keep permanently) ───────────────
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        if not gt_path.exists():
            print(f"  Downloading ground_truth.txt ...")
            _gdown_file(
                folder_url = folder_url,
                subfolder  = subject_id,
                filename   = "ground_truth.txt",
                dest       = gt_path,
            )
        else:
            print(f"  Ground truth already cached.")

        if not gt_path.exists():
            print(f"  [SKIP] Could not download ground truth for {tag}.")
            continue

        # ── Download video (large — purge after extraction) ───────────────
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if not video_path.exists():
            print(f"  Downloading vid.avi ...")
            _gdown_file(
                folder_url = folder_url,
                subfolder  = subject_id,
                filename   = "vid.avi",
                dest       = video_path,
            )
        else:
            print(f"  Video already on disk.")

        if not video_path.exists():
            print(f"  [SKIP] Could not download video for {tag}.")
            continue

        # ── Yield the job ─────────────────────────────────────────────────
        yield VideoJob(
            video_path       = video_path,
            subject_id       = tag,
            condition        = "unknown",   # no stress labels in UBFC-rPPG
            source           = "ubfc_rppg",
            ground_truth_bvp = gt_path,
            label            = -1,
            metadata         = {
                "subject_num": subject_num,
                "dataset":     "UBFC-rPPG",
                "gt_fps":      30.0,        # UBFC-rPPG BVP is at 30 Hz
            },
            _delete_after = not keep_raw,
        )
        # NOTE: dispatcher must call job.cleanup() after processing


# ──────────────────────────────────────────────────────────────
# 5.  UBFC-Phys iterator  (future — AWS S3 via IEEE DataPort)
# ──────────────────────────────────────────────────────────────

def iter_ubfc_phys(cfg: dict) -> Generator[VideoJob, None, None]:
    """
    Future: download UBFC-Phys subjects from AWS S3 using IEEE DataPort
    credentials (access key + secret key provided after dataset approval).

    Protocol: T1 (rest) → T2 (speech, skip) → T3 (arithmetic stress)
    Labels: T1 = 0 (rest), T3 = 1 (stress)
    Ground truth: BVP at 64 Hz from Empatica E4 wristband

    Access steps:
      1. Request access at https://ieee-dataport.org/open-access/ubfc-phys-2
      2. Once approved, set AWS credentials in config.yaml under ubfc_phys.aws
      3. Uncomment and implement the aws s3 cp download logic below

    Stub included so dispatcher.py can already reference this function.
    """
    print("[UBFC-Phys] AWS S3 access not yet configured.")
    print("  Request dataset access at: https://ieee-dataport.org/open-access/ubfc-phys-2")
    print("  Once approved, add AWS credentials to config.yaml under ubfc_phys.aws")
    return
    yield   # makes this a generator


# ──────────────────────────────────────────────────────────────
# 6.  COHFACE iterator  (future — API access)
# ──────────────────────────────────────────────────────────────

def iter_cohface(cfg: dict) -> Generator[VideoJob, None, None]:
    """
    Future: stream videos from the COHFACE API without local storage.
    Stub included so dispatcher.py can already reference this function.
    """
    print("[COHFACE] API integration not yet implemented — skipping.")
    return
    yield


# ──────────────────────────────────────────────────────────────
# 7.  gdown helper
# ──────────────────────────────────────────────────────────────

def _gdown_file(
    folder_url: str,
    subfolder:  str,
    filename:   str,
    dest:       Path,
    retries:    int = 3,
) -> bool:
    """
    Download a single file from a Google Drive folder using gdown.

    gdown navigates into the subfolder and downloads the named file.
    Falls back to folder-level download if direct file URL isn't available.

    Parameters
    ----------
    folder_url : base Google Drive folder URL for the dataset
    subfolder  : subject folder name (e.g. "subject1")
    filename   : file to download ("vid.avi" or "ground_truth.txt")
    dest       : local destination path
    retries    : number of attempts before giving up

    Returns
    -------
    bool — True if file was downloaded successfully
    """
    import gdown

    dest = Path(dest)
    tmp_dir = dest.parent / f"_tmp_{subfolder}"

    for attempt in range(1, retries + 1):
        try:
            print(f"    Attempt {attempt}/{retries} — downloading {filename} ...")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Download the entire subject subfolder (small — 1 video + 1 txt)
            # gdown handles the Drive folder navigation automatically
            subject_folder_url = f"{folder_url.rstrip('/')}"
            gdown.download_folder(
                url          = subject_folder_url,
                output       = str(tmp_dir),
                quiet        = False,
                use_cookies  = False,
                remaining_ok = True,
            )

            # Find the downloaded file and move it to dest
            downloaded = list(tmp_dir.rglob(filename))
            if downloaded:
                shutil.move(str(downloaded[0]), str(dest))
                print(f"    Saved → {dest}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return True
            else:
                print(f"    [WARNING] {filename} not found in downloaded folder.")

        except Exception as exc:
            print(f"    [ERROR] Attempt {attempt} failed: {exc}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    Retrying in {wait}s ...")
                time.sleep(wait)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    [FAILED] Could not download {filename} after {retries} attempts.")
    return False


# ──────────────────────────────────────────────────────────────
# UBFC-rPPG local iterator  (reads already-downloaded files)
# ──────────────────────────────────────────────────────────────

def iter_ubfc_rppg_local(cfg: dict) -> Generator[VideoJob, None, None]:
    """
    Yield VideoJob for each UBFC-rPPG subject already downloaded locally.

    Reads from config.paths.ubfc_local — expects structure:
        ubfc_local/
        ├── subject1/
        │   ├── vid.avi
        │   └── ground_truth.txt
        ├── subject3/
        ...

    No downloading — works with files already on disk.
    Videos are NOT deleted after processing since they were downloaded manually.

    Yields
    ------
    VideoJob with source="ubfc_rppg", label=-1, ground_truth_bvp set
    """
    ubfc_dir = Path(cfg["paths"].get("ubfc_local", ""))

    if not ubfc_dir or not ubfc_dir.exists():
        print(f"[UBFC-local] ubfc_local path not set or not found: {ubfc_dir}")
        print("  Set paths.ubfc_local in config.yaml to your moxie_data/ubfc/ folder.")
        return

    # Find all subject folders that contain vid.avi
    subject_dirs = sorted([
        d for d in ubfc_dir.iterdir()
        if d.is_dir() and (d / "vid.avi").exists()
    ])

    if not subject_dirs:
        print(f"[UBFC-local] No subject folders with vid.avi found in {ubfc_dir}")
        return

    print(f"[UBFC-local] Found {len(subject_dirs)} subjects in {ubfc_dir}")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        video_path = subject_dir / "vid.avi"
        gt_path    = subject_dir / "ground_truth.txt"

        if not gt_path.exists():
            print(f"  [SKIP] {subject_id} — ground_truth.txt missing")
            continue

        tag = f"ubfc_rppg_{subject_id}"
        print(f"  → {tag}: vid.avi + ground_truth.txt")

        yield VideoJob(
            video_path       = video_path,
            subject_id       = tag,
            condition        = "unknown",
            source           = "ubfc_rppg",
            ground_truth_bvp = gt_path,
            label            = -1,
            metadata         = {
                "subject_id": subject_id,
                "dataset":    "UBFC-rPPG",
                "gt_fps":     30.0,
                "gt_format":  "ubfc_rppg",
            },
            _delete_after = False,
        )