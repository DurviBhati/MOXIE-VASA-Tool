# Dataset Description & Validation

## 1. Target Data (Real World)
The ultimate goal of MOXIE-VASA is to process proprietary experimental data collected by the Tewari Lab.
* **Source:** Experimental recordings of subjects undergoing stress induction tasks.
* **Format:** Medium-resolution `.mp4` video (30fps) synchronized with other wearable device.
* **Challenge:** Large file sizes and dim lighting conditions.
* **Validation Plan:** Comparing the tool's rPPG output against the authentic ECG signal to calculate accuracy.

## 2. Development Data (Simulation)
To develop and test the software pipeline without risking patient privacy, I will use the **UBFC-rPPG Dataset**. This serves as our "Simulated" environment because it is clean, labeled, and publicly available.

### 2.1 Dataset Structure
* **Name:** UBFC-rPPG (University of Bourgogne Franche-Comté Remote PPG).
* **Size:** 49 videos.
* **Subjects:** Diverse participants sitting stationary (some subject showing minor movements) playing a mathematical game.
* **Inputs:** Uncompressed video frames.
* **Ground Truth:** Synchronized finger-clip pulse oximeter data.

### 2.2 Why this meets requirements?
* **Video Quality:** High quality allows for easier debugging of the rPPG algorithm before trying it on "noisy" lab data.
* **Stress Context:** The dataset specifically captures subjects doing mental math, mimicking a close use case of the MOXIE lab experiment.
* **Ground Truth:** Contains precise heart rate data to validate if the `pyVHR` implementation is working correctly.

### 2.3 Data Statistics (Subset for Testing)
For the initial video analyses, I will my videos and as as scripting progresses, I will use a subset of **5 subjects**:
* **Format:** RGB Video (`.avi`).
* **Duration:** ~2 minutes per subject.
* **Resolution:** 640x480.
* **Frame Rate:** 30fps (matches our minimum requirement).
