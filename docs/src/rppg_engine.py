import os
import csv
import sys
print("[RPPGEngine] Waking up and configuring environment...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut off all TensorFlow info/warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force TensorFlow to use CPU only and stop hanging in WSL!
print("[RPPGEngine] Loading heavy libraries...")
import numpy as np
import scipy.signal
##Bypasssing the "cupy" library 
sys.modules['cupy']=np
sys.modules['cusignal']= scipy.signal

from pyVHR.analysis.pipeline import Pipeline

def run_pyvhr_extraction(video_path, output_folder):
    """
    Extracts Heart Rate / BVP using PyVHR.
    """
    video_filename = os.path.basename(video_path)
    print(f"[RPPGEngine] Preparing to process PyVHR for: {video_filename}")

    try:
        print("[RPPGEngine] Initializing PyVHR AI Models...")

        ## Creating the PyVHR Pipeline
        pipe = Pipeline()
        ##Running the analysis
        ##We wil use the default robust settings to find the face and extract the pulse
        print(f"[RPPGEngine] Analyzing micro-color changes (rPPG)...")
        time, BPM, uncertainty = pipe.run_on_video(video_path)

        ##Formating the output path 
        name_only, _ = os.path.splitext(video_filename)
        output_csv = os.path.join(ouput_folder, f"{name_only}_HeartRate.csv")

        ##Save to CSV 
        with open(output_csv, mode='w', newline='') as file:
            writer =csv.writer(file)
            writer.writerow(['Time_sec', 'HeartRate_BPM', 'Uncertainity'])

            #Looping through the data and write each row 
            for t, b, u in zip(time, BPM, uncertainty):
                ##PyVHR will return arrays for the BPM sometimes, making sure we get numbers
                bpm_val = b[0] if isinstance(b, (list, tuple, np.ndarray)) else b
                writer.writerow([t, bpm_val, u])
            
        print("[RPPGEngine] Success! rPPG data extracted and saved to: {output_csv}")
        return True
        
    except Exception as e:
        print(f"[RPPGEngine] Critical Failure: {e}")
        return False