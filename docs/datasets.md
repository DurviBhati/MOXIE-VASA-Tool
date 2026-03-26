# Dataset Description

**Project:** MOXIE-VASA — Video Analysis for Stress Assessment  
**Author:** Durvi Bhati  
**Last Updated:** March 2026  

---


## 1. Development Data — Current Pipeline

### 1.1 Example Dataset for Using the Tool

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
| **Size on disk** | Varies; `_safe.mp4` transcoded copies at CRF 15 are stored alongside originals |

**Why this is a good example dataset:**  
These videos are ideal for pipeline smoke-testing because they have consistent lighting, a stationary subject, and a clear frontal face conditions that minimise confounds when debugging rPPG signal quality and OpenFace landmark accuracy. Running the full pipeline on familiar personal footage makes it immediately obvious when something is wrong (e.g., a flat BVP signal or missing AU columns). They are also small enough to iterate on quickly during development.

---

### 1.2 Validation Dataset — UBFC-Phys (Phase 3)

Used for training the ML stress classifier and validating rPPG accuracy against ground truth BVP.

| Property | Details |
|---|---|
| **Full name** | UBFC-Phys: A Multimodal Database for Psychophysiological Studies of Social Stress |
| **Reference** | Meziati Sabour et al., IEEE Transactions on Affective Computing, 2021 |
| **Subjects** | 56 participants (46 female, 10 male) |
| **Stress protocol** | TSST-adjacent: T1 (rest) → T2 (speech task) → T3 (arithmetic task) |
| **Video** | 168 RGB videos at 1024×1024, 35 Hz (EO-23121C camera) |
| **Ground truth** | BVP + EDA from Empatica E4 wristband at 64 Hz |
| **Access** | Open source |
| **Total size** | ~80–100 GB (full dataset) |

**Why UBFC-Phys is appropriate:**  
- Uses a TSST-inspired social stress induction protocol — directly relevant to the lab dataset's stress paradigm  
- Provides BVP ground truth at 64 Hz — enables quantitative rPPG validation (MAE, Pearson r)  
- Widely used benchmark in the rPPG literature — results are comparable to published baselines  
- Multimodal (BVP + EDA) — EDA could be incorporated as an additional validation signal in future work


---

### 1.3 Robustness Test Dataset — COHFACE (Future)

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


---

## Target Data — Tewari Lab (Ultimate Goal)

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
