# Design Document Specification (DDS)
**Project:** MOXIE-VASA  
**Architecture:** Modular Pipeline

## 1. Tool Overview
MOXIE-VASA is designed as a modular pipeline where a central controller ("The Dispatcher") routes data to specialized processing workers. This ensures that the Video processing logic is kept separate from other wearbale data, allowing for easier debugging and updates.

## 2. Module Descriptions

### Module A: The Dispatcher (Controller)
* **Scope:** The entry point of the software.
* **Responsibility:** Scans the input folder, detects file types (`.mp4` vs `.csv`), and decides which processing module to trigger.
* **Inputs:** Raw directory path.
* **Outputs:** Commands to Docker/Python.

### Module B: The Video Engine (OpenFace Wrapper)
* **Scope:** Facial Behavior Analysis.
* **Responsibility:** Takes the Docker container logic. It runs the `FeatureExtraction` binary.
* **Inputs:** Raw Video File (`.mp4`).
* **Outputs:** `facial_features.csv` (Action Units, Gaze, Pose).

### Module C: The Physiological Engine (pyVHR)
* **Scope:** Remote Heart Rate Estimation.
* **Responsibility:** Identifies the ROI (Region of Interest) on the face (forehead/cheeks) and computes the BVP signal using the POS algorithm.
* **Inputs:** Raw Video File (`.mp4`).
* **Outputs:** `heart_rate.csv` (BPM over time).

### Module D: The Fusion Engine
* **Scope:** Data Merging & Validation.
* **Responsibility:** Aligns the timestamps from Module B and Module C. Compares results against the actual clinical data if available.
* **Inputs:** `facial_features.csv`, `heart_rate.csv`.
* **Outputs:** `final_stress_profile.csv`.

## 3. Module Dependency Diagram
This diagram illustrates the data flow between modules.

```mermaid
graph TD
    A[Raw Data Input] -->|File Detected| B(Dispatcher Module)
    
    B -->|Video Found| C[Video Engine: OpenFace]
    B -->|Video Found| D[Physio Engine: pyVHR]
    B -->|Clinal HR signal Found| E[Validation Loader]
    
    C -->|Action Units| F[Fusion Engine]
    D -->|Heart Rate| F
    E -->|True HR| F
    
    F --> G[Final Stress Report]
