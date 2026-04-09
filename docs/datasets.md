# Dataset Description

**Project:** MOXIE-VASA — Video Analysis for Stress Assessment  
**Author:** Durvi Bhati  
**Last Updated:** April 2026  

---

## 1. Target Data — Tewari Lab (Ultimate Goal)

The ultimate goal of MOXIE-VASA is to process proprietary experimental data collected by the Tewari Lab at the University of Michigan.

| Property | Details |
|---|---|
| **Source** | Experimental recordings of subjects undergoing TSST and PTSD stress induction tasks |
| **Format** | Full-body video (head to toe), medium resolution, ~30fps |
| **Lighting** | Indoor, suboptimal — yellowish hue, lower contrast than controlled datasets |
| **Framing** | Head to toe — face crop preprocessing step will be required |
| **Synchronisation** | Paired with other physiological modalities (EDA, ECG) from wearable devices |
| **Validation plan** | Compare rPPG HR output against ECG ground truth to compute MAE and Pearson r |
| **Access** | Pending — not yet available. Pipeline will be adapted when access is granted. |

**Note on OpenPose:** Because these videos are full-body, OpenPose body pose features (shoulder tension, body lean, postural fidgeting) will be added in Phase 4 to complement the facial features extracted by OpenFace.

---

## 2. Development Data — Current Pipeline

### 2.1 Example Dataset for Using the Tool

**Personal videos (`moxie_data/My_video/`)**

These are the primary development and smoke-test videos used to build and validate the Phase 1 pipeline.

| Property | Details |
|---|---|
| **Files** | `mine_1.mp4` through `mine_5.mp4` |
| **Format** | H.264 MP4 |
| **Resolution** | 1920 × 1080 (Full HD) |
| **Frame rate** | ~29.6–29.9 fps (variable, read from metadata at runtime) |
| **Duration** | 34 seconds to ~2.5 minutes per video |
| **Framing** | Head to shoulder — face clearly visible throughout |
| **Lighting** | Good indoor lighting, consistent illumination |
| **Subject** | Single subject wearing glasses — glabella ROI confirmed accessible |
| **Stress labels** | None — used for pipeline validation only, not ML training |

**Why this is a good example dataset:**
These videos are ideal for pipeline smoke-testing because they have consistent lighting, a stationary subject, and a clear frontal face — conditions that minimise confounds when debugging rPPG signal quality and OpenFace landmark accuracy. Running the full pipeline on familiar personal footage makes it immediately obvious when something is wrong (e.g., a flat BVP signal or missing AU columns). They are also small enough to iterate on quickly during development.

---

### 2.2 Real Dataset for Answering a Biological Question Using the Tool

**UBFC-rPPG — University of Bourgogne Franche-Comté Remote PPG**

**Biological question:**
> Can physiological and facial behavioural features extracted automatically from facial video serve as reliable indicators of stress, consistent with established stress physiology literature?

This is a sub-question of the broader MOXIE research program: *"Can multimodal physiological signals — video, ECG, EDA, audio — together characterise human stress levels?"* MOXIE-VASA addresses the video modality specifically.

**URL:** https://drive.google.com/drive/folders/1q4vWuF2GJvKP5xyeX8dxaJ2fmq97-4ai

| Property | Details |
|---|---|
| **Full name** | UBFC-rPPG (University of Bourgogne Franche-Comté Remote PPG) |
| **Subjects** | 49 participants total; 10 used in this analysis |
| **Task** | Mental arithmetic game (cognitive stress induction) |
| **Video format** | 640 × 480 RGB, ~29–30 fps, ~45–70 seconds per subject |
| **Ground truth** | Finger-clip pulse oximeter at 30 Hz (simultaneous recording) |
| **GT file structure** | `ground_truth.txt` — 3 rows: BVP waveform, HR timeseries (bpm), timestamps (s) |
| **Stress labels** | None — used for rPPG validation and feature consistency analysis |
| **Access** | Publicly available via Google Drive (link above) |

**Why this dataset is appropriate:**
The mental arithmetic task induces mild cognitive stress similar to the Tewari Lab experimental protocol, making it the ideal validation dataset before applying MOXIE-VASA to proprietary lab recordings. The dataset provides simultaneous contact-sensor ground truth (pulse oximeter) at the same frame rate as the video, enabling quantitative rPPG validation (MAE in bpm, Pearson r). At 640×480 resolution it matches the processing resolution of our rPPG pipeline exactly, requiring no coordinate scaling.

**Expected results:**
Based on published stress physiology literature, we expect:
- Heart rate elevated above 85 bpm in the majority of subjects during mental arithmetic — consistent with Taelman et al. 2009 (HR rises ~15 bpm under stress)
- Mild facial expression activation (AU04 brow furrow, AU14 jaw tension) — cognitive stress produces subtler facial responses than emotional stress (Fairclough & Houston 2004)
- rPPG HR estimation within 5–12 bpm MAE of the oximeter ground truth — consistent with published POS algorithm benchmarks on this dataset

**Data statistics (10-subject subset used):**

| Metric | Value |
|---|---|
| Video durations | 45–70 seconds per subject |
| Ground truth HR range | 66–113 bpm |
| OpenFace detection rate | 100% of frames (confidence >0.88) across all subjects |
| rPPG mean MAE (vs oximeter) | ~17 bpm (best subject: 8 bpm) |
| Subjects with elevated HR (>85 bpm) | 8/10 |
| Stress score range | 12.9% – 72.1% |

**Tutorial notebook:** [`tutorials/ubfc_rppg_analysis.ipynb`](../tutorials/ubfc_rppg_analysis.ipynb)

---

### 2.3 Future Development — UBFC-Phys

Used for training the ML stress classifier and validating rPPG accuracy against labeled ground truth BVP.

| Property | Details |
|---|---|
| **Full name** | UBFC-Phys: A Multimodal Database for Psychophysiological Studies of Social Stress |
| **Reference** | Meziati Sabour et al., IEEE Transactions on Affective Computing, 2021 |
| **Subjects** | 56 participants (46 female, 10 male) |
| **Stress protocol** | TSST-adjacent: T1 (rest) → T2 (speech task) → T3 (arithmetic task) |
| **Video** | 168 RGB videos at 1024×1024, 35 Hz (EO-23121C camera) |
| **Ground truth** | BVP + EDA from Empatica E4 wristband at 64 Hz |
| **Labels** | T1 = rest (0), T3 = stress (1) — direct stress/rest labels |
| **Access** | IEEE DataPort (open access via S3) — `s3://ieee-dataport/open/49099/sN.zip` |
| **Total size** | ~80–100 GB (full dataset); process-and-purge loop limits peak disk to ~2 GB |
| **Status** | 🔮 Future development — access infrastructure confirmed, implementation pending |

**Why UBFC-Phys is appropriate:**
Uses a TSST-inspired social stress induction protocol directly relevant to the Tewari Lab paradigm. Provides BVP ground truth at 64 Hz enabling quantitative rPPG validation. Widely used benchmark in the rPPG literature — results are comparable to published baselines. Contains explicit stress/rest labels (T1 vs T3) required for training the Phase 3 ML classifier.

**Stress label mapping:**

| Condition | Label | Use |
|---|---|---|
| T1 — Rest task | 0 (no stress) | ML training negative class |
| T2 — Speech task | skipped | Not used |
| T3 — Arithmetic task | 1 (stress) | ML training positive class |

---

### 2.4 Future Development — COHFACE

Used to test pipeline robustness under suboptimal lighting — mimicking the Tewari Lab video conditions.

| Property | Details |
|---|---|
| **Full name** | COHFACE Dataset |
| **Reference** | Heusch, Anjos & Marcel, 2017 |
| **Subjects** | 40 participants |
| **Conditions** | Indoor, suboptimal lighting (similar to lab dataset) |
| **Ground truth** | BVP from Nonin Medical pulse oximeter |
| **Access** | API access (no bulk download needed) |
| **Stress labels** | None — used for rPPG robustness testing only, not stress classification |

**Purpose:** Running the trained rPPG module on COHFACE (without retraining) reveals how much signal quality degrades under poor lighting — directly predicting performance on the Tewari Lab data.

---

## 3. Dataset Strategy Summary

| Dataset | Role | Status |
|---|---|---|
| Personal videos (mine_1–5) | Pipeline smoke test and development | ✅ Complete |
| UBFC-rPPG (10-subject subset) | rPPG validation + rule-based stress scoring | ✅ Complete |
| UBFC-Phys | ML classifier training with stress/rest labels | 🔮 Future development |
| COHFACE | Robustness testing under poor lighting | 🔮 Future development |
| Tewari Lab dataset | Target application — production use | 🔮 Future development |
