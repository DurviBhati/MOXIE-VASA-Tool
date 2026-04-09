"""
app.py  —  MOXIE-VASA Streamlit GUI
=====================================
Phase 2 GUI — folder-based pipeline runner.

User flow
---------
  1. Enter path to moxie_data/ root folder
  2. GUI auto-detects My_video/ (personal) and ubfc/ (UBFC-rPPG) subfolders
  3. Click Run Pipeline — processes all videos with per-step progress
  4. View summary table + expandable subject cards with stress scores
  5. Download PDF report per subject

Run
---
    cd src/Moxie
    streamlit run app.py
"""

import sys
import os
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ── Ensure src/Moxie is importable ───────────────────────────────────────────
MOXIE_SRC = Path(__file__).resolve().parent
if str(MOXIE_SRC) not in sys.path:
    sys.path.insert(0, str(MOXIE_SRC))

from data_ingestion        import load_config
from preprocessing         import route_and_preprocess
from openface.video_engine import run_openface_feature_extraction
from rppg.pipeline         import run_rppg_analysis
from feature_extractor     import extract_features
from stress_inference      import compute_stress_score
from report_generator      import generate_report

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_EXT = (".mp4", ".avi", ".mov", ".mkv")

SCORE_COLOUR = {
    "LOW STRESS":      "#27AE60",
    "MODERATE STRESS": "#E67E22",
    "HIGH STRESS":     "#C0392B",
}
SCORE_EMOJI = {
    "LOW STRESS":      "🟢",
    "MODERATE STRESS": "🟡",
    "HIGH STRESS":     "🔴",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOXIE-VASA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MOXIE-VASA")
    st.caption("Video Analysis for Stress Assessment  \n"
               "University of Michigan  \n"
               "Phase 2 — Rule-based Scorer")
    st.markdown("---")

    st.markdown("### Pipeline stages")
    st.markdown(
        "1. **Preprocessing** — codec normalise, face check\n"
        "2. **OpenFace** — AU intensities, head pose *(Docker)*\n"
        "3. **rPPG** — BVP waveform → HR via POS algorithm\n"
        "4. **Scoring** — rule-based stress score 0–100%\n"
        "5. **PDF report** — 3-page per-subject report\n"
    )
    st.markdown("---")

    st.markdown("### Requirements")
    st.markdown(
        "- Docker Desktop running\n"
        "- `algebr/openface:latest` pulled\n"
        "- `config.yaml` set up at project root\n"
    )
    st.markdown("---")

    st.markdown("### Data folder structure")
    st.code(
        "moxie_data/\n"
        "├── My_video/        ← personal videos\n"
        "│   ├── video1.mp4\n"
        "│   └── video2.mp4\n"
        "└── ubfc/            ← UBFC-rPPG subjects\n"
        "    ├── subject1/\n"
        "    │   ├── vid.avi\n"
        "    │   └── ground_truth.txt\n"
        "    └── subject3/\n",
        language="text"
    )
    st.markdown("---")
    st.caption(
        "⚠️ Research tool only.  \n"
        "Not a validated clinical instrument."
    )

# ── Config ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_config():
    try:
        return load_config(), None
    except Exception as e:
        return None, str(e)

cfg, cfg_err = get_config()
if cfg_err:
    st.error(f"Cannot load config.yaml: {cfg_err}")
    st.info("Run `streamlit run app.py` from `src/Moxie/` and ensure "
            "`config.yaml` exists at the project root.")
    st.stop()

OF_DIR     = Path(cfg["paths"]["openface_output"])
RPPG_DIR   = Path(cfg["paths"]["rppg_output"])
REPORT_DIR = Path(cfg["paths"]["output_root"]) / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper: discover videos ───────────────────────────────────────────────────

def discover_sources(data_root: Path) -> dict:
    """
    Scan data_root for known source folders.
    Returns dict: { source_label: [(video_path, gt_path_or_None), ...] }
    """
    sources = {}

    # Personal videos — flat folder of video files
    personal_dir = data_root / "My_video"
    if personal_dir.exists():
        vids = sorted([
            f for f in personal_dir.iterdir()
            if f.suffix.lower() in VALID_EXT
            and "_safe" not in f.name
            and f.is_file()
        ])
        if vids:
            sources["Personal Videos (My_video/)"] = [
                (v, None) for v in vids
            ]

    # UBFC-rPPG — subfolders with vid.avi + ground_truth.txt
    ubfc_dir = data_root / "ubfc"
    if ubfc_dir.exists():
        ubfc_jobs = []
        for subj_dir in sorted(ubfc_dir.iterdir()):
            vid = subj_dir / "vid.avi"
            gt  = subj_dir / "ground_truth.txt"
            if vid.exists():
                ubfc_jobs.append((vid, gt if gt.exists() else None))
        if ubfc_jobs:
            sources[f"UBFC-rPPG (ubfc/) — {len(ubfc_jobs)} subjects"] = ubfc_jobs

    return sources


def _safe_name(video_path: Path) -> str:
    """Predict the safe filename preprocessing.py will produce."""
    par  = video_path.parent.name
    stem = video_path.stem
    if par and par != stem:
        return f"{par}_{stem}_safe.mp4"
    return f"{stem}_safe.mp4"


# ── Process one video ─────────────────────────────────────────────────────────

def process_one(
    video_path: Path,
    gt_path,
    step_containers: tuple,
    log_area,
) -> dict:
    """
    Run preprocessing → OpenFace → rPPG → feature extraction → scoring.
    Updates the three step_containers (st.empty) with progress status.
    Returns a result dict.
    """
    s1, s2, s3 = step_containers
    result = {
        "video": video_path.name,
        "subject_id": None,
        "features": None,
        "score_result": None,
        "report_path": None,
        "error": None,
    }

    def log(msg):
        log_area.text(msg)

    try:
        import cv2

        # ── Step 1: Preprocessing ─────────────────────────────────────────
        s1.info("⏳ Preprocessing")
        safe_name     = _safe_name(video_path)
        expected_safe = video_path.parent / safe_name

        if expected_safe.exists():
            cap = cv2.VideoCapture(str(expected_safe))
            metadata = {
                "fps":         cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            cap.release()
            processed_video = str(expected_safe)
            log(f"  Preprocessing: cached ({safe_name})")
            s1.success("✅ Preprocessing")
        else:
            log(f"  Preprocessing: transcoding {video_path.name} ...")
            processed_video, metadata = route_and_preprocess(str(video_path))
            log(f"  Preprocessing: done → {safe_name}")
            s1.success("✅ Preprocessing")

        stem = Path(processed_video).stem
        fps  = metadata.get("fps", 30.0)
        result["subject_id"] = stem

        # ── Step 2: OpenFace ──────────────────────────────────────────────
        s2.info("⏳ OpenFace")
        expected_csv = OF_DIR / f"{stem}.csv"

        if expected_csv.exists() and expected_csv.stat().st_size > 0:
            log(f"  OpenFace: cached ({stem}.csv)")
            s2.success("✅ OpenFace")
            of_ok = True
        else:
            log(f"  OpenFace: running Docker container (~5 min) ...")
            of_ok = run_openface_feature_extraction(
                processed_video, str(OF_DIR)
            )
            log(f"  OpenFace: {'done' if of_ok else 'FAILED'}")
            if of_ok:
                s2.success("✅ OpenFace")
            else:
                s2.error("❌ OpenFace failed")

        if not of_ok:
            result["error"] = "OpenFace failed"
            s3.warning("⏭ rPPG skipped")
            return result

        # ── Step 3: rPPG ─────────────────────────────────────────────────
        s3.info("⏳ rPPG")
        expected_metrics = RPPG_DIR / f"{stem}_metrics.json"
        expected_bvp     = RPPG_DIR / f"{stem}_bvp.csv"

        gt_bvp_path = str(gt_path) if gt_path else None

        if expected_metrics.exists() and expected_bvp.exists():
            log(f"  rPPG: cached ({stem}_metrics.json)")
            s3.success("✅ rPPG")
            rppg_ok = True
        else:
            log(f"  rPPG: running POS pipeline ...")
            rppg_ok = run_rppg_analysis(
                processed_video, str(RPPG_DIR),
                ground_truth_bvp=gt_bvp_path,
                fps=fps,
                gt_format="ubfc_rppg",
            )
            log(f"  rPPG: {'done' if rppg_ok else 'FAILED'}")
            if rppg_ok:
                s3.success("✅ rPPG")
            else:
                s3.error("❌ rPPG failed")

        if not rppg_ok:
            result["error"] = "rPPG failed"
            return result

        # ── Feature extraction + scoring ──────────────────────────────────
        log(f"  Extracting features ...")
        features     = extract_features(expected_csv, expected_metrics,
                                        subject_id=stem)
        score_result = compute_stress_score(features)

        # ── PDF report ────────────────────────────────────────────────────
        report_path = REPORT_DIR / f"{stem}_stress_report.pdf"
        log(f"  Generating PDF report ...")
        generate_report(
            score_result  = score_result,
            features      = features,
            output_path   = report_path,
            bvp_csv_path  = expected_bvp,
            fps           = fps,
        )

        result["features"]     = features
        result["score_result"] = score_result
        result["report_path"]  = str(report_path) if report_path.exists() else None
        log(f"  ✓ Complete — score: {score_result['score']:.0f}% "
            f"[{score_result['classification']}]")

    except Exception as e:
        import traceback
        result["error"] = str(e)
        log(f"  ERROR: {e}\n{traceback.format_exc()}")
        s1.error("❌ Error")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧬 MOXIE-VASA — Video Stress Analysis")
st.markdown(
    "Automatically extract **heart rate** (rPPG) and **facial action units** "
    "(OpenFace) from video to produce a physiological stress assessment report."
)
st.markdown("---")

# ── Folder input ──────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([4, 1])

with col_input:
    data_root_str = st.text_input(
        "📁 Data root folder",
        value=str(cfg["paths"]["data_root"]),
        placeholder="/path/to/moxie_data",
        help=(
            "Path to your data root folder. Should contain:\n"
            "  • My_video/  — personal video files (.mp4, .avi, etc.)\n"
            "  • ubfc/      — UBFC-rPPG subject folders (subject1/, subject2/, ...)\n"
            "Either or both subfolders will be auto-detected."
        ),
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run Pipeline", type="primary",
                        use_container_width=True)

# ── Source preview ────────────────────────────────────────────────────────────
if data_root_str:
    data_root = Path(data_root_str)
    if data_root.exists():
        sources = discover_sources(data_root)
        if sources:
            total_videos = sum(len(v) for v in sources.values())
            st.success(
                f"✅ Found **{total_videos} video(s)** across "
                f"**{len(sources)} source(s)**"
            )
            for src_label, jobs in sources.items():
                with st.expander(f"📂 {src_label} ({len(jobs)} videos)"):
                    for vp, gt in jobs:
                        gt_tag = " + GT" if gt else ""
                        st.text(f"  • {vp.parent.name}/{vp.name}{gt_tag}")
        else:
            st.warning(
                "No videos found. Expected `My_video/` or `ubfc/` subfolders "
                "inside the data root."
            )
            sources = {}
    else:
        st.error("Folder does not exist — check the path.")
        sources = {}
else:
    sources = {}

# ── Pipeline run ──────────────────────────────────────────────────────────────
if run_btn and sources:
    st.markdown("---")
    st.markdown("## ⚙️ Processing")

    all_results = []
    total_videos = sum(len(v) for v in sources.values())
    overall_progress = st.progress(0, text="Starting...")
    videos_done = 0

    for src_label, jobs in sources.items():
        st.markdown(f"### 📂 {src_label}")

        for video_path, gt_path in jobs:
            vid_label = f"{video_path.parent.name}/{video_path.name}"
            st.markdown(f"**`{vid_label}`**")

            # Three columns for step indicators
            c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
            s1 = c1.empty()
            s2 = c2.empty()
            s3 = c3.empty()
            log_area = c4.empty()

            s1.info("⏳ Preprocessing")
            s2.warning("— OpenFace")
            s3.warning("— rPPG")

            _args = dict(video_path=video_path, gt_path=gt_path, step_containers=(s1, s2, s3), log_area=log_area)
            result = process_one(**_args)
            result["source"] = src_label
            all_results.append(result)

            videos_done += 1
            overall_progress.progress(
                videos_done / total_videos,
                text=f"Completed {videos_done}/{total_videos} videos"
            )

    overall_progress.progress(1.0, text="✅ All videos processed!")
    st.session_state["all_results"] = all_results

# ── Results ───────────────────────────────────────────────────────────────────
results_to_show = st.session_state.get("all_results", [])

if results_to_show:
    st.markdown("---")
    st.markdown("## 📊 Results")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for r in results_to_show:
        if r.get("error") or not r.get("score_result"):
            rows.append({
                "Subject":        r.get("subject_id") or r["video"],
                "Source":         r.get("source", ""),
                "Score (%)":      "—",
                "Classification": f"ERROR: {r.get('error','')}",
                "HR (bpm)":       "—",
                "Confidence":     "—",
            })
        else:
            sr   = r["score_result"]
            feat = r["features"]
            rows.append({
                "Subject":        r["subject_id"],
                "Source":         r.get("source", ""),
                "Score (%)":      f"{sr['score']:.0f}%",
                "Classification": f"{SCORE_EMOJI[sr['classification']]} {sr['classification']}",
                "HR (bpm)":       f"{feat.get('hr_mean',0):.1f}",
                "Confidence":     sr["confidence"],
            })

    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ── Download all results as CSV ───────────────────────────────────────────
    csv_data = df_summary.to_csv(index=False)
    st.download_button(
        "⬇️ Download summary CSV",
        data      = csv_data,
        file_name = "moxie_vasa_results.csv",
        mime      = "text/csv",
    )

    st.markdown("---")
    st.markdown("## 📋 Subject Reports")

    # ── Expandable cards ──────────────────────────────────────────────────────
    for r in results_to_show:
        if r.get("error") or not r.get("score_result"):
            with st.expander(f"❌ {r.get('subject_id') or r['video']}"):
                st.error(f"Error: {r.get('error', 'Unknown')}")
            continue

        sr      = r["score_result"]
        feat    = r["features"]
        score   = sr["score"]
        classi  = sr["classification"]
        colour  = SCORE_COLOUR[classi]
        emoji   = SCORE_EMOJI[classi]

        with st.expander(
            f"{emoji} **{r['subject_id']}** — "
            f"{classi} ({score:.0f}%)",
            expanded=False
        ):
            # ── Score + key metrics ───────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Stress Score",   f"{score:.0f}%")
            m2.metric("HR (rPPG)",      f"{feat.get('hr_mean',0):.1f} bpm")
            m3.metric("AU04 (Brow)",    f"{feat.get('au04_mean',0):.2f}")
            m4.metric("AU14 (Jaw)",     f"{feat.get('au14_mean',0):.2f}")
            m5.metric("Confidence",     sr["confidence"])

            # GT HR if available
            if feat.get("gt_hr"):
                st.info(
                    f"📡 Ground truth HR (pulse oximeter): "
                    f"**{feat['gt_hr']:.1f} bpm** | "
                    f"MAE = **{feat['hr_mae']:.1f} bpm**"
                )

            # ── Triggered indicators ──────────────────────────────────────
            triggered = [i for i in sr["indicators"] if i["triggered"]]
            not_triggered = [i for i in sr["indicators"] if not i["triggered"]]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**🔴 Triggered indicators:**")
                if triggered:
                    for ind in triggered:
                        st.markdown(
                            f"- **{ind['label']}**  \n"
                            f"  Value: `{ind['value']:.3f}` > "
                            f"threshold `{ind['threshold']}`  \n"
                            f"  Contribution: `{ind['contribution']:.1f}%`"
                        )
                else:
                    st.markdown("*None triggered*")

            with col_b:
                st.markdown("**⚪ Not triggered:**")
                for ind in not_triggered:
                    st.markdown(
                        f"- {ind['label']}  \n"
                        f"  Value: `{ind['value']:.3f}` "
                        f"(threshold: `{ind['threshold']}`)"
                    )

            # ── HRV note ──────────────────────────────────────────────────
            hrv = sr.get("hrv_note", {})
            if hrv:
                with st.container():
                    st.caption(
                        f"HRV: RMSSD={hrv.get('rmssd',0):.1f} ms  |  "
                        f"LF/HF={hrv.get('lf_hf_ratio',0):.3f}  |  "
                        f"{hrv.get('reliability_note','')}"
                    )

            # ── PDF download ──────────────────────────────────────────────
            if r.get("report_path") and Path(r["report_path"]).exists():
                with open(r["report_path"], "rb") as f_pdf:
                    st.download_button(
                        label     = "📄 Download PDF Report",
                        data      = f_pdf.read(),
                        file_name = Path(r["report_path"]).name,
                        mime      = "application/pdf",
                        key       = f"pdf_{r['subject_id']}",
                    )

elif not run_btn:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "👆 Enter your data root folder path above and click **Run Pipeline**.  \n\n"
        "The tool will automatically detect `My_video/` and `ubfc/` subfolders "
        "and process all videos found."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What gets extracted")
        st.markdown("""
| Feature | Source | Stress relevance |
|---|---|---|
| Heart rate (bpm) | rPPG / POS | Elevated >85 bpm |
| AU04 — Brow furrow | OpenFace | Concentration/distress |
| AU07 — Lid tighten | OpenFace | Tension/focus |
| AU14 — Jaw tension | OpenFace | Sustained effort |
| AU15 — Lip depress | OpenFace | Negative affect |
| Head motion variance | OpenFace | Restlessness |
        """)

    with col2:
        st.markdown("### Stress score formula")
        st.markdown("""
**Rule-based weighted sum:**

| Indicator | Threshold | Weight |
|---|---|---|
| HR elevated | >85 bpm | 50% |
| AU04 active | >0.25 | 20% |
| AU14 active | >0.25 | 15% |
| AU07 active | >0.15 | 10% |
| AU15 active | >0.05 | 5% |

Score 0–29%: 🟢 LOW STRESS  
Score 30–59%: 🟡 MODERATE STRESS  
Score ≥60%: 🔴 HIGH STRESS
        """)