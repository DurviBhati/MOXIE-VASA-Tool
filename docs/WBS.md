# Work Breakdown Structure (WBS)

## Phase 1: The "First Basic Output"
**Goal:** Successfully extract Facial Action Units (AUs) from a single test video using OpenFace containerization.

### Activity 1.1: Environment Setup
* **Task 1.1.1:** Install Docker Desktop and enable WSL 2 integration. (In process)
* **Task 1.1.2:** Pull the `algebr/openface` Docker image. (In process)
* **Task 1.1.3:** Create a project directory structure (`src/`, `docs/`, `data/`). (Done)

### Activity 1.2: Data Preparation
* **Task 1.2.1:** Download the UBFC-rPPG sample dataset.
* **Task 1.2.2:** Select one subject video (e.g., `subject1.avi`) and place it in the input folder.
* **Task 1.2.3:** Verify video integrity (ensure it plays and has valid metadata).

### Activity 1.3: Video Processing Implementation
* **Task 1.3.1:** Write the Python "Dispatcher" script (`src/dispatcher.py`) to construct the Docker run command.
* **Task 1.3.2:** Run `FeatureExtraction` tool on the sample video.

### Activity 1.4: Validation
* **Task 1.4.1:** Check if `processed_features.csv` was generated.
* **Task 1.4.2:** Open CSV and inspect column headers (ensure `AU01_r`, `AU04_r` exist).
* **Task 1.4.3:** Generate a simple plot of AU intensity over time to verify data quality.

---

## Phase 2: Future Development (Problem 2B)
**Goal:** Expand the tool to include Heart Rate estimation for Stress Assessment.

### Activity 2.1: Heart Rate Estimation Module
* **Task 2.1.1:** Create a Python virtual environment for `pyVHR`.
* **Task 2.1.2:** Implement the  rPPG algorithm.
* **Task 2.1.3:** Output a time-series CSV of Heart Rate (BPM).
