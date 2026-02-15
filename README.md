# MOXIE-VASA: Video Analysis for Stress Assessment

**A Multimodal, Hardware-Agnostic Fusion Engine for Automated Stress Quantification.**

---

## 🧬 What is VASA?
MOXIE-VASA is a specialized bioinformatics pipeline designed to automate the input, processing, and analysis of multimodal human wearable data. It serves as the preprocessing and feature selection (extension to ML/DL and final stress report generation) for the broader MOXIE project.

Unlike standard tools that analyze a single data stream (e.g., just heart rate or just facial expressions), VASA integrates Video-derived Physiological Signals (rPPG) with the complete Video analyses to generate a robust, unified "Stress Report." 

As the development in AI continues, the future in research would be Organization powers AI tool. VASA begins with that thought, It will use a **Private AI Agent** (powered by UMich GPT) to guide researchers through complex decisions and handling multimodal data.

## 🛑 The Problem
Current workflows for stress research are fragmented:
1.  **Data Silos:** Facial expression data (Action Units) and physiological data (ECG) are processed in separate software, making temporal alignment difficult.
2.  **Manual Labor:** Researchers must manually convert proprietary file formats (e.g., .acq, .fit, .csv) before analysis.
3.  **Proprietary Black Boxes:** Commercial tools (e.g., FaceReader, iMotions) are closed-source, Windows-only, and cannot run on High-Performance Computing (HPC) clusters like U-M Great Lakes.

## 💡 The VASA Solution
MOXIE-VASA solves these issues by creating a single Automated Pipeline:
For the sake of the project and bulk data VASA focuses on complete VIDEO and deriving rPPG from the video. 
Later extension to -->
* **Universal Integration:** Automatically detects and routes file types (.mp4, .acq, .csv, .txt) to specific processing workers.

## 📂 Inputs & Outputs
### **Input**
The tool accepts video files:
* **Video:** .mp4 (Face & rPPG extraction).

### **Output (The Report)**
A consolidated Analysis Package for each subject:
1.  **features_merged.csv: Time-aligned dataset containing Facial AUs,  Heart Rate (rPPG + Sensor)** 
2.  **stress_verdict.csv: The Machine Learning prediction of stress levels (Low/Medium/High) over time.**

---
