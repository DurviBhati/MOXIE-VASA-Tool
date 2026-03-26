# Work Breakdown Structure (WBS)

**Project:** MOXIE-VASA — Video Analysis for Stress Assessment  
**Author:** Durvi Bhati  
**Last Updated:** March 2026  

---

## Phase 1: Core Pipeline — Personal Video Processing

**Goal:** Build and validate a fully working end-to-end pipeline on personal videos.  
**Status:** ✅ Complete

### Activity 1.1: Environment Setup
- **Task 1.1.1:** Install Docker Desktop and enable WSL2 integration ✅
- **Task 1.1.2:** Pull `algebr/openface:latest` Docker image ✅
- **Task 1.1.3:** Create project directory structure (`src/`, `docs/`, `tests/`) ✅
- **Task 1.1.4:** Set up Python virtual environment (`moxie_env_1` via conda, Python 3.10) ✅
- **Task 1.1.5:** Install core dependencies (`opencv-python`, `numpy`, `scipy`, `pandas`, `pyyaml`) ✅

### Activity 1.2: Configuration and Data Architecture
- **Task 1.2.1:** Design config-driven path system — `config.yaml` + `config.template.yaml` ✅
- **Task 1.2.2:** Implement `load_config()` with `${data_root}` placeholder resolution ✅
- **Task 1.2.3:** Set up `moxie_data/` folder structure outside the repo ✅
- **Task 1.2.4:** Place personal videos (`mine_1.mp4` through `mine_5.mp4`) in `moxie_data/My_video/` ✅
- **Task 1.2.5:** Configure `.gitignore` to exclude `config.yaml`, `Pipeline_Output/`, video files ✅

### Activity 1.3: Preprocessing Module (`preprocessing.py`)
- **Task 1.3.1:** Implement `_make_safe_mp4()` — H.264 CRF 15 transcode via ffmpeg ✅
- **Task 1.3.2:** Implement rotation correction using `ffprobe` metadata ✅
- **Task 1.3.3:** Implement `_extract_metadata()` — FPS, resolution, frame count ✅
- **Task 1.3.4:** Implement `_verify_face_present()` — Haar cascade check on 10 sampled frames ✅
- **Task 1.3.5:** Update `route_and_preprocess()` to return `(safe_path, metadata)` tuple ✅

### Activity 1.4: OpenFace Module (`openface/video_engine.py`)
- **Task 1.4.1:** Investigate Docker image entrypoint — confirmed `/bin/bash`, binary at `/home/openface-build/build/bin/FeatureExtraction` ✅
- **Task 1.4.2:** Build correct Docker command with `--entrypoint` and `-w` flags ✅
- **Task 1.4.3:** Implement live `stdout` streaming via `subprocess.Popen` ✅
- **Task 1.4.4:** Implement output CSV verification (file exists + non-empty) ✅
- **Task 1.4.5:** Add 30-minute timeout guard and `FileNotFoundError` for missing Docker ✅
- **Task 1.4.6:** Verify AU extraction on all 5 personal videos — CSVs confirmed ✅

### Activity 1.5: rPPG Module (`rppg/`)
- **Task 1.5.1:** Implement `roi_extractor.py` — OpenFace landmark → 4 ROI bounding boxes with coordinate scaling ✅
- **Task 1.5.2:** Implement `rgb_signal.py` — BGR→RGB, spatial mean per patch, area-weighted combination ✅
- **Task 1.5.3:** Implement `filtering.py` — moving-average detrend + Butterworth bandpass (SOS form) ✅
- **Task 1.5.4:** Implement `pos_algorithm.py` — POS with Hanning-tapered overlap-add windows ✅
- **Task 1.5.5:** Implement `hr_estimation.py` — peak detection, HR, SDNN, RMSSD, LF/HF ✅
- **Task 1.5.6:** Implement `evaluation.py` — MAE + Pearson r vs BVP ground truth (dormant for personal videos) ✅
- **Task 1.5.7:** Implement `pipeline.py` orchestrator — frame loading, sub-module calls, output saving ✅

### Activity 1.6: Dispatcher Integration
- **Task 1.6.1:** Refactor `dispatcher.py` to read all paths from `load_config()` ✅
- **Task 1.6.2:** Update `process_video()` to unpack `(safe_path, metadata)` from preprocessing ✅
- **Task 1.6.3:** Pass actual FPS from metadata to `run_rppg_analysis()` ✅
- **Task 1.6.4:** Run full pipeline on all 5 personal videos — 2/5 full success, 3/5 partial (mine_1, mine_2 too short for HRV) ✅
- **Task 1.6.5:** Fix OpenFace leading-space column names in `roi_extractor._load_csv()` ✅
- **Task 1.6.6:** Fix landmark coordinate scaling bug (1920×1080 → 640×480) in `roi_extractor._get_landmarks()` ✅
- **Task 1.6.7:** Fix `rgb_signal.py` single-pass extraction to avoid empty signal bug ✅

### Activity 1.7: Design Documentation Update
- **Task 1.7.1:** Update `SRS.md` to reflect v0.5 architecture ✅
- **Task 1.7.2:** Update `DDS.md` with actual module descriptions and dependency diagram ✅
- **Task 1.7.3:** Update `WBS.md` with completed and planned tasks ✅
- **Task 1.7.4:** Update `datasets.md` with actual dataset strategy ✅

---

## Phase 2: Output Layer — GUI, Report, Rule-Based Scoring

**Goal:** Complete the user-facing output layer so the tool produces a downloadable PDF report from a video upload via a Streamlit interface.  
**Status:** 🔄 In Progress

### Activity 2.1: Rule-Based Stress Inference (`stress_inference.py`)
- **Task 2.1.1:** Implement physiological threshold scorer (HR > 85 bpm, RMSSD < 30 ms, AU04 intensity, head motion variance)
- **Task 2.1.2:** Return structured score dict with `score`, `reasons`, `model_version`
- **Task 2.1.3:** Write unit tests for threshold logic edge cases

### Activity 2.2: Plot Utilities (`plot_utils.py`)
- **Task 2.2.1:** Implement HR timeline plot (HR over time, saved as PNG)
- **Task 2.2.2:** Implement AU bar chart (top 5 AUs by mean intensity)
- **Task 2.2.3:** Implement stress gauge visualisation

### Activity 2.3: PDF Report Generator (`report_generator.py`)
- **Task 2.3.1:** Set up `fpdf2` document structure (3–4 pages)
- **Task 2.3.2:** Add subject metadata section
- **Task 2.3.3:** Embed stress score gauge and classification result
- **Task 2.3.4:** Embed HR timeline PNG
- **Task 2.3.5:** Add AU summary table with plain-English AU descriptions
- **Task 2.3.6:** Add methodology note for academic credibility
- **Task 2.3.7:** Add placeholder SHAP section (to be filled in Phase 3)

### Activity 2.4: Streamlit GUI (`app.py`) (Confirm if this is needed)
- **Task 2.4.1:** Build video upload widget with format validation
- **Task 2.4.2:** Add pipeline progress bar with per-step status (preprocessing / OpenFace / rPPG)
- **Task 2.4.3:** Display stress score gauge and HR/HRV summary in results panel
- **Task 2.4.4:** Implement PDF download button
- **Task 2.4.5:** Add sidebar with model info, methodology blurb, and future work note (OpenPose, lab dataset)
- **Task 2.4.6:** Handle edge cases: face not found, video too short for HRV, Docker not running

### Activity 2.5: Tutorial Notebook
- **Task 2.5.1:** Create `tutorials/` folder in repo
- **Task 2.5.2:** Write `moxie_vasa_demo.ipynb` — runs pipeline on one personal video, shows all outputs
- **Task 2.5.3:** Add narrative text explaining each pipeline stage
- **Task 2.5.4:** Plot HR signal and top AU intensities inline
- **Task 2.5.5:** Update `README.md` with link to tutorial and datasets.md

---

## Phase 3: ML Layer — UBFC-Phys Training and Validated Stress Score

**Goal:** Replace the rule-based scorer with a trained Random Forest classifier using UBFC-Phys as training data.  
**Status:** 📋 Planned

### Activity 3.1: Dataset Ingestion (`data_ingestion.py`)
- **Task 3.1.1:** Validate previous logistics for UBFC-Phys dataset.
- **Task 3.1.2:** Confirm download URL pattern.
- **Task 3.1.3:** Implement `iter_ubfc_phys()` download-and-purge loop
- **Task 3.1.4:** Validate BVP CSV format and update `evaluation.py` accordingly
- **Task 3.1.5:** Run pipeline on 15-subject subset (T1 + T3 only)

### Activity 3.2: Feature Engineering
- **Task 3.2.1:** Implement `feature_extractor.py` — AU temporal aggregates (mean, std, max per window)
- **Task 3.2.2:** Implement `feature_fusion.py` — merge AU + rPPG features into master DataFrame
- **Task 3.2.3:** Validate feature matrix — check for NaN, scale, class balance

### Activity 3.3: ML Training (`ml/model_trainer.py`) 
- **Task 3.3.1:** Implement LOSO cross-validation loop
- **Task 3.3.2:** Train decided models....
- **Task 3.3.3:** Tune hyperparameters....
- **Task 3.3.4:** Compare against baseline....
- **Task 3.3.5:** Serialize best model....
- **Task 3.3.6:** Log F1, AUC-ROC, confusion matrix per fold

### Activity 3.4: Explainability (`ml/explainability.py`) (The step needs more thinking and validation)
- **Task 3.4.1:** Implement SHAP... Confirm the plan.
- **Task 3.4.2:** Generate waterfall plot per prediction
- **Task 3.4.3:** Map top SHAP features ....How? will define later 
- **Task 3.4.4:** Replace rule-based reasons in report with SHAP-derived explanations

### Activity 3.5: Quantitative Evaluation
- **Task 3.5.1:** Run rPPG validation against UBFC-Phys BVP — compute MAE and Pearson r
- **Task 3.5.2:** Add model performance tab to Streamlit (F1, AUC per fold, confusion matrix)
- **Task 3.5.3:** Add COHFACE robustness test — verify features don't degrade under poor lighting

---

## Phase 4: Lab Dataset Integration and Full Deployment

**Goal:** Apply the validated pipeline to the Tewari Lab proprietary dataset and deploy for cluster batch processing.  
**Status:** 🔮 Future

### Activity 4.1: Lab Dataset Preprocessing
- **Task 4.1.1:** Obtain access to lab videos once available
- **Task 4.1.2:** Adapt preprocessing for head-to-toe framing (face crop step needed)
- **Task 4.1.3:** Apply lighting correction for yellowish hue
- **Task 4.1.4:** Validate OpenFace performance on lower-quality footage

### Activity 4.2: OpenPose Integration (`openpose/`)
- **Task 4.2.1:** Implement `openpose/pose_engine.py` for body pose extraction
- **Task 4.2.2:** Extract posture features (shoulder tension, body lean, fidgeting)
- **Task 4.2.3:** Add pose features to feature fusion layer

### Activity 4.3: Cluster Deployment
- **Task 4.3.1:** Write Slurm job array script for batch processing on U-M Great Lakes
- **Task 4.3.2:** Test with 10-video batch before full dataset run
- **Task 4.3.3:** Implement result aggregation across Slurm array outputs

---

## Summary

| Phase | Status | Key Deliverable |
|---|---|---|
| Phase 1: Core Pipeline | ✅ Complete | Working end-to-end pipeline on 5 personal videos |
| Phase 2: Output Layer | 🔄 In Progress | Streamlit GUI + PDF report + tutorial notebook |
| Phase 3: ML Layer | 📋 Planned | Trained classifier + SHAP explainability |
| Phase 4: Lab Dataset | 🔮 Future | Full lab dataset processing + cluster deployment |
