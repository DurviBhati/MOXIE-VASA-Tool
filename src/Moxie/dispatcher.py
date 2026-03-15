import os
import sys
import subprocess

# This forces Python to look in the exact folder where this script lives
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import route_and_preprocess
from openface.video_engine import run_openface_feature_extraction
from rppg.pipeline import run_rppg_analysis
## from openpose .....


def main():
    """
    Main Dispatcher:
    This function identifies the input video.
    checks if output folder exists.
    and finally alls the VideoEngine module.
    """
    
    print("=== MOXIE-VASA Dispatcher v0.3 ===")
    
    #--------------------------------------------------------------------
    # Setting up the Input/output folder path 
    #-------------------------------------------------------------------------
    # Define the input folder path and create the output directory accordingly 
    
    input_folder = "/mnt/c/Users/durvi/University of Michigan Dropbox/Durvi Bhati/moxie_vas_tool/input"
    ##Creating the output folder automatically 
    base_output_folder = os.path.join(input_folder, "Pipeline_Output")    
    # Define module-specific output folders
    openface_output = os.path.join(base_output_folder, "result_openface")
    rppg_output = os.path.join(base_output_folder, "result_rppg")
    openpose_output = os.path.join(base_output_folder, "result_openpose")
   
   # Create all directories safely using a loop
    for folder in [base_output_folder, openface_output, rppg_output, openpose_output]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"Created output directory: {folder}")
    
    #------------------------------------------------------------------------------
    # Looping through every video in the input folder
    #-------------------------------------------------------------------------------
    # Define which video formats
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    # Get the list of files and filter out the ones we don't want
    all_files = os.listdir(input_folder)
    video_files = [
        f for f in all_files 
        if f.lower().endswith(valid_extensions) and "_safe.mp4" not in f
    ]
    
    ## now loop through 
    for filename in video_files:
        input_video = os.path.join(input_folder, filename)

        # Skip non-video files or directories
        if not os.path.isfile(input_video):
            continue
            
        print(f"\n=========================================================")
        print(f" NOW PROCESSING: {filename}")
        print(f"=========================================================")
    
        try:
            # 1. Routing and preprocessing 
            print(">>> Starting Preprocessing...")
            processed_video = route_and_preprocess(input_video)

            # 2. Executing the OpenFace module
            print("\n>>> Starting OpenFace Analysis...")
            openface_success = run_openface_feature_extraction(processed_video, openface_output)

            # 3. Executing the New Native rPPG Analysis
            print("\n>>> Starting Custom rPPG Analysis...")
            rppg_success = run_rppg_analysis(processed_video, rppg_output)
            
            # 4. Executing the OpenPose Analysis 
            # print("\n>>> Starting OpenPose Analysis...")
            # ....................................

            # 5. Reporting success and failure
            if openface_success and rppg_success:
                print(f"\n[SUCCESS] Entire pipeline complete for {filename}!")
            else:
                print(f"\n[WARNING] Pipeline finished with errors for {filename}.")
                print(f"  - OpenFace Success: {openface_success}")
                print(f"  - rPPG Success: {rppg_success}")
    
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Pipeline crashed on {filename}: {e}")

    
if __name__ == "__main__":
    main()