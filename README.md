# MOXIE-VASA: Video Analysis for Stress Assessment

**A Multimodal, Non-Invasive Pipeline for Automated Stress Quantification from Video.**

---

## What is MOXIE-VASA?

MOXIE-VASA is a specialised bioinformatics pipeline that automates the processing and analysis of video data for stress assessment. It serves as the video processing and feature extraction module for the broader MOXIE project.

Unlike standard tools that analyse a single data stream, VASA integrates **facial behavioural analysis** (Action Units via OpenFace) with **custom remote photoplethysmography** (rPPG using the POS algorithm) to extract both facial and physiological stress indicators from video — no contact sensors required.

---

## The Problem

Current workflows for stress research are fragmented:

1. **Data Silos:** Facial expression data (Action Units) and physiological data (HR, HRV) are processed in separate software, making temporal alignment difficult.
2. **Manual Labor:** Researchers must manually convert proprietary file formats and run tools in sequence without a unified pipeline.
3. **Proprietary Black Boxes:** Commercial tools (e.g., FaceReader, iMotions) are closed-source, Windows-only, and cannot run on HPC clusters like U-M Great Lakes.

---

## The MOXIE-VASA Solution

A single automated pipeline that takes a video file and produces a structured stress analysis:

```
Video input
    → Preprocessing (codec fix, rotation correction, face check)
    → OpenFace (Action Units, head pose, gaze, 68 facial landmarks)
    → Custom rPPG (POS algorithm → HR, SDNN, RMSSD, LF/HF ratio)
    → Stress scoring and PDF report
```

## Inputs and Outputs

### Input

Video files in any of the following formats: `.mp4`, `.avi`, `.mov`, `.mkv`

### Output (per video)

OpenFace features — AUs, pose, gaze, 68 landmarks (one row per frame) 
BVP waveform from rPPG (one value per valid frame) 
HR, SDNN, RMSSD, LF/HF ratio, HRV warnings 
*(Phase 2)*  PDF report with stress score, HR timeline, AU analysis 

---

## Example Data

The pipeline is demonstrated on a set of personal videos (`mine_1.mp4` through `mine_5.mp4`) recorded at 1920×1080, ~29.7 fps, with head-to-shoulder framing and good indoor lighting.

These videos are not committed to the repository (they exceed 20 MB) but reside in `moxie_data/My_video/` outside the repo root. The path is configured via `config.yaml`.

See [`docs/datasets.md`](docs/datasets.md) for full dataset descriptions including resolution, frame counts, pipeline results per video, and the UBFC-Phys validation dataset strategy for Phase 3.

---

## Tutorial

A step-by-step walkthrough of the full pipeline is available as a Jupyter notebook:

**[`tutorials/moxie_vasa_demo.ipynb`](tutorials/moxie_vasa_demo.ipynb)**

The notebook runs the pipeline on `mine_4.mp4` (4,561 frames, ~153 seconds, 29.8 fps) and demonstrates:
- Video preprocessing and metadata extraction
- OpenFace CSV output structure and AU intensity preview
- BVP waveform plot with detected heartbeat peaks
- Instantaneous heart rate over time

---

## Setup

### Prerequisites

- Python ≥ 3.10 (conda environment recommended)
- Docker Desktop with WSL2 integration enabled
- `algebr/openface:latest` Docker image pulled
- `ffmpeg` installed in WSL2

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/moxie_vas_tool.git
cd moxie_vas_tool

# Create environment and install dependencies
conda create -n moxie_env python=3.10
conda activate moxie_env
pip install opencv-python numpy scipy pandas pyyaml matplotlib

# Pull the OpenFace Docker image
docker pull algebr/openface:latest

# Set up your config
cp config.template.yaml config.yaml
# Edit config.yaml — set data_root to your local moxie_data folder
```

### Running the pipeline

```bash
cd src/Moxie

# Process personal videos (default mode)
python dispatcher.py

```

## Citation

If you use MOXIE-VASA in your research, please cite the POS rPPG algorithm this tool is built on:

> Wang, W., ...
