# Software Requirements Specification (SRS)

**Project:** MOXIE-VASA — Video Analysis for Stress Assessment  
**Version:** 0.5 (Phase 1 — Personal Video Pipeline)  
**Author:** Durvi Bhati  
**Institution:** University of Michigan  
**Last Updated:** March 2026  

---

## 1. Introduction

### 1.1 Purpose

MOXIE-VASA is an automated, non-invasive tool for quantifying human stress levels from video data. It integrates facial behavioural analysis (Action Units via OpenFace) with a custom remote photoplethysmography (rPPG) module using the POS algorithm to extract physiological and facial stress indicators without requiring contact sensors.

The tool is designed for use in psychological and clinical research, with a particular focus on stress induction protocols such as the Trier Social Stress Test (TSST) and PTSD-related assessments.

### 1.2 Scope

MOXIE-VASA serves as the primary video processing module for the broader MOXIE stress assessment framework. The current implementation (Phase 1) establishes a fully working end-to-end pipeline on personal videos.

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | End-to-end pipeline: preprocessing → OpenFace → custom rPPG → metrics output |
| Phase 2 | 🔄 In Progress | Streamlit GUI, PDF report generation, rule-based stress scoring |
| Phase 3 | 📋 Planned | UBFC-Phys ingestion, ML classifier (Random Forest + SHAP), validated stress score |
| Phase 4 | 🔮 Future | Lab dataset integration, OpenPose body pose, full multimodal fusion |

### 1.3 Definitions and Acronyms

| Term | Definition |
|---|---|
| AU | Action Unit — a unit of facial muscle movement per the Facial Action Coding System (FACS) |
| rPPG | Remote Photoplethysmography — contactless HR estimation from skin colour changes in video |
| POS | Plane-Orthogonal-to-Skin — the rPPG algorithm implemented in this tool (Wang et al. 2017) |
| ROI | Region of Interest — facial skin patch used for rPPG signal extraction |
| HRV | Heart Rate Variability — statistical measures of RR interval variation, a stress indicator |
| BVP | Blood Volume Pulse — the physiological ground truth signal from wristband sensors |

### 1.4 What Changed from v0.1

- **pyVHR replaced** by a custom rPPG module built from scratch — avoids dependency conflicts and gives full control over the processing pipeline
- **OpenFace landmarks now drive ROI localization** — ROIs follow the face during head movement rather than using fixed bounding boxes
- **Slurm/cluster deployment deferred** — tool currently runs locally via WSL2 on Windows with Docker for OpenFace
- **Output format expanded** from a single CSV to: per-video BVP waveform CSV, RGB signal CSV, and a metrics JSON containing HR/HRV features
- **Config-driven architecture** — all paths managed via `config.yaml`, no hardcoded paths in source code
- **Three-source data ingestion designed** — personal videos (active), UBFC-Phys (Phase 3), COHFACE (future)

---

## 2. User Requirements

| ID | Requirement | Status |
|---|---|---|
| UR-01 | **Automated Ingestion:** The system processes all videos in a configured folder without manual intervention per video. | ✅ Done |
| UR-02 | **Feature Extraction:** The tool extracts frame-by-frame AU intensities and presence scores for stress-relevant Action Units. | ✅ Done |
| UR-03 | **Physiological Estimation:** The tool estimates HR, SDNN, RMSSD, and LF/HF ratio from the video feed using the POS rPPG algorithm. | ✅ Done |
| UR-04 | **Data Export:** The system produces structured output files (CSV + JSON) per video with facial and physiological features. | ✅ Done |
| UR-05 | **Stress Scoring:** The tool produces a stress score (0–100%) with human-readable explanations of contributing factors. | 🔄 Phase 2 |
| UR-06 | **PDF Report:** The tool generates a downloadable PDF report per subject with stress score, HR timeline, and AU analysis. | 🔄 Phase 2 |
| UR-07 | **GUI Interface:** A Streamlit web interface allows video upload and report download without command-line usage. | 🔄 Phase 2 |
| UR-08 | **ML Classification:** A trained Random Forest classifier produces a validated stress score using UBFC-Phys training data. | 📋 Phase 3 |

---

## 3. Functional Requirements

### 3.1 Input Handling

- The tool accepts `.mp4`, `.avi`, `.mov`, and `.mkv` video formats
- All input paths are resolved from `config.yaml` — no paths are hardcoded in source files
- The system validates that a face is present in at least one sampled frame before processing (Haar cascade check in `preprocessing.py`)
- Videos with no detectable face raise a `RuntimeError` with a descriptive message

### 3.2 Processing Pipeline

#### 3.2.1 Preprocessing (`preprocessing.py`)

- All input videos are transcoded to H.264 MP4 at CRF 15 (near-lossless) to ensure codec compatibility with OpenFace Docker and OpenCV
- Rotation metadata is corrected automatically using `ffprobe`
- A `_safe.mp4` copy is written alongside the original — the original is never modified
- Video FPS is read from file metadata and passed downstream to the rPPG pipeline — never hardcoded
- Face presence check uses OpenCV Haar cascade on 10 evenly-spaced sampled frames

#### 3.2.2 Facial Analysis (`openface/video_engine.py`)

- OpenFace `FeatureExtraction` runs inside the `algebr/openface:latest` Docker container
- Binary path: `/home/openface-build/build/bin/FeatureExtraction`; working directory: `/home/openface-build/build/bin/`
- Extracted features: AU01–AU45 (intensity `_r` and presence `_c`), head pose (`Rx/Ry/Rz`, `Tx/Ty/Tz`), gaze angles, 68 2D/3D facial landmarks
- Output: one CSV per video written to `Pipeline_Output/result_openface/`
- OpenFace landmark coordinates are reused by the rPPG module for ROI localization — face detection runs only once

#### 3.2.3 rPPG Analysis (`rppg/`)

The rPPG module consists of seven sub-modules executed in sequence:

| Module | Responsibility |
|---|---|
| `roi_extractor.py` | Reads OpenFace landmark CSV; derives forehead, left cheek, right cheek, and glabella ROI bounding boxes per frame; scales coordinates from original resolution to processing resolution (640×480) |
| `rgb_signal.py` | Extracts mean R, G, B per ROI per frame; combines four ROIs into one area-weighted signal |
| `filtering.py` | Detrends signal via moving-average subtraction; applies 4th-order zero-phase Butterworth bandpass (0.7–3.0 Hz) using SOS form |
| `pos_algorithm.py` | Implements POS with overlapping Hanning-tapered windows; outputs normalised BVP waveform |
| `hr_estimation.py` | Detects systolic peaks; computes HR (bpm), SDNN, RMSSD, LF power, HF power, LF/HF ratio |
| `evaluation.py` | When ground truth BVP is available (UBFC-Phys), computes MAE and Pearson r vs reference signal — dormant for personal videos |
| `pipeline.py` | Orchestrator: loads frames, calls all sub-modules in order, writes BVP CSV + RGB CSV + metrics JSON |

#### 3.2.4 Feature Extraction and Fusion (Phase 3)

- `feature_extractor.py` will aggregate per-frame OpenFace features into temporal statistics (mean, std, max per AU per condition window)
- `feature_fusion.py` will merge AU features with rPPG HRV metrics into a single feature matrix for ML training and inference

#### 3.2.5 ML Classification (Phase 3)

- Random Forest classifier trained on UBFC-Phys dataset using Leave-One-Subject-Out cross-validation
- SHAP `TreeExplainer` generates per-prediction feature importance for report explainability
- Rule-based threshold scorer used as placeholder in Phase 2 until training data is available

### 3.3 Output Generation

Per video, the pipeline writes the following outputs:

| File | Location | Contents |
|---|---|---|
| `{stem}_safe.mp4` | `moxie_data/personal/` | Preprocessed video (codec-normalised, rotation-corrected) |
| `{stem}.csv` | `result_openface/` | OpenFace features: AUs, pose, gaze, 68 landmarks — one row per frame |
| `{stem}_bvp.csv` | `result_rppg/` | BVP waveform — one value per valid frame |
| `{stem}_rgb.csv` | `result_rppg/` | Per-ROI mean RGB signal — one row per valid frame |
| `{stem}_metrics.json` | `result_rppg/` | HR, SDNN, RMSSD, LF/HF ratio, warnings, video metadata |
| `{stem}_report.pdf` *(Phase 2)* | `result_reports/` | PDF with stress score, HR timeline, SHAP waterfall, AU table |

### 3.4 Configuration Management

- All paths and parameters stored in `config.yaml` at the project root
- `config.yaml` is gitignored — `config.template.yaml` (with placeholder values) is committed instead
- Parameters include: rPPG window/step length, bandpass cutoffs, ROI type, OpenFace AU column list, UBFC-Phys subject subset

---

## 4. Non-Functional Requirements

| ID | Category | Requirement | Status |
|---|---|---|---|
| NFR-01 | Performance | A 2-minute personal video (1920×1080, ~29.7fps) completes OpenFace extraction in under 10 minutes and rPPG in under 2 minutes on a standard laptop | ✅ Met |
| NFR-02 | Privacy | No video data or identifiable information is transmitted to external servers — all processing is local (WSL2) or on the secure U-M cluster | ✅ Met |
| NFR-03 | Reproducibility | All paths and parameters are config-driven — results are reproducible given the same input video and `config.yaml` | ✅ Met |
| NFR-04 | Robustness | The pipeline handles videos where face detection fails on individual frames (up to 30% missing ROI frames) without crashing | ✅ Met |
| NFR-05 | Portability | Runs on Linux/WSL2 — Docker required for OpenFace — Python ≥ 3.10 | ✅ Met |
| NFR-06 | Scalability | Batch processing of 50+ videos via Slurm job arrays on U-M Great Lakes cluster | 🔮 Phase 4 |
| NFR-07 | Explainability | Every stress score output includes a plain-English explanation of the top contributing features | 📋 Phase 3 |

---

## 5. Dependencies

| Dependency | Version | Purpose | Required For |
|---|---|---|---|
| Python | ≥ 3.10 | Runtime | All phases |
| `opencv-python` | ≥ 4.8 | Frame reading, face detection, resize | All phases |
| `numpy` | ≥ 1.24 | Array operations throughout pipeline | All phases |
| `scipy` | ≥ 1.11 | Butterworth filter, Welch PSD, peak detection | All phases |
| `pandas` | ≥ 2.0 | CSV I/O for OpenFace output and feature matrices | All phases |
| `pyyaml` | ≥ 6.0 | `config.yaml` loading | All phases |
| Docker | ≥ 24.0 | OpenFace container execution | All phases |
| `algebr/openface` | latest | Facial AU and landmark extraction | All phases |
| `ffmpeg` | ≥ 6.0 | Video transcoding, rotation correction | All phases |
| `streamlit` | ≥ 1.32 | GUI interface | Phase 2 |
| `fpdf2` | ≥ 2.7 | PDF report generation | Phase 2 |
| `matplotlib` | ≥ 3.8 | HR timeline and signal plots | Phase 2 |
| `scikit-learn` | ≥ 1.4 | Random Forest, LOSO CV, feature selection | Phase 3 |
| `shap` | ≥ 0.45 | ML explainability — TreeExplainer | Phase 3 |
| `xgboost` | ≥ 2.0 | Alternative classifier for comparison | Phase 3 |
