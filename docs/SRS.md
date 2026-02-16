# Software Requirements Specifications (SRS)

**Project Name:** MOXIE-VASA(Video Analysis for Stress Assessment)
**Version:** 0.1(Video Phase)
**Author:** Durvi Bhati 

##Introduction

### 1.1 Purpose 
The purpose of the MOXIE-VASA toll is to provide an automates, non invasive method for quantifying human stress levels using video data. By integrating facial behavirol analysis(Action Units) with remote physiological monitoring (rPPG), the tool aims to replace subjective manual coding with objectives.

### 1.2 Scope
The tool acts as the primary video processing module for the larger MOXIE stress assessment framework.
* **Current Phase:** Takes input in the form of video files, extracts facial features via OpenFace, computes the heart rate via pyVHR,
* **Future face ** The tool will later integrate the external wearable data later once the initial pipeline is successful.

## 2. User Requirements 
* **01 Automated Ingestion:** Person uploads a folder of subject videos ('.mp4') to the tool on cluster; the system automatically queues them for processing without manual intervention. 
* **02 Feature Extraction:** The tool extracts frame-by-frame intensities for stress-related Action Units.
* **03 Physiological Estimation:** The tool estimates heart rate from the same video feed using rPPG algorithms, robust to head motions.
* **04 Data Export:** The user recieves a simplified CSV report containing time-aligned facial and heart rate data for stress assessment.

## 3. Functional Requirements
### 3.1 Input Handling
* The tool accepts standard video formats (`.mp4`, `.avi`).
* The tool validates video quality (minimum resolution 1080p, minimum frame rate 30fps).

### 3.2 Processing Pipeline
* **Facial Analysis:** VASA utilizes a containerized version of **OpenFace** to extract presence and intensity of Action Units.
* **Heart Rate Analysis:** VASA will  utilize **pyVHR** (Python Virtual Heart Rate) to extract the  signal from facial ROI (Region of Interest).
* **Fusion (Preliminary):** As the next step VASA will merge the OpenFace CSV and pyVHR CSV based on timestamp.

### 3.3 Output Generation
* The tool will generate a `processed_features.csv` file with columns: `Time`, `HR_bpm`, `AU01`, `AU04`, `AU12`, `Stress_Index`.
* VASA will flag frames where confidence is low (e.g., face not detected).

## 4. Non-Functional Requirements
* **Scalability:** The pipeline must support batch processing of 50+ videos via Slurm job arrays on the U-M Great Lakes cluster.
* **Privacy:** No identifiable video data shall be transmitted to public clouds; all processing remains on the secure cluster environment.
* **Performance:** A 10-minute video should be processed in under reasonable compute time. 
