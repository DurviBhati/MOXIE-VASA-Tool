import os
import sys

# --- BULLETPROOF IMPORT PATH ---
# This forces Python to look in the exact folder where this script lives
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import route_and_preprocess
from video_engine import run_openface_feature_extraction
from rppg_engine import run_pyvhr_extraction


def main():
    """
    Main Dispatcher:
    This function identifies the input video.
    checks if output folder exists.
    and finally alls the VideoEngine module.
    """
    
    print("=== MOXIE-VASA Dispatcher v0.3 ===")
    
    #--------------------------------------------------------------------
    # Setting up the Input folder path 
    #-------------------------------------------------------------------------
    # Define the input folder path and create the output directory accordingly 
    
    input_folder = "/mnt/c/Users/durvi/University of Michigan Dropbox/Durvi Bhati/moxie_vas_tool/input"
    ##Creating the output folder automatically 
    output_folder = os.path.join(input_folder, "Results")    
    
    #--------------------------------------------------------------------------------------------------------
    # Validating the existence of the video
    #-------------------------------------------------------------------------------------------------------------
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
     
    #------------------------------------------------------------------------------
    # Looping through every video in the input folder
    #-------------------------------------------------------------------------------
    for filename in os.listdir(input_folder):
        input_video = os.path.join(input_folder, filename)

        ## Skipping other folder or non-video files
        if not os.path.isfile(input_video):
            continue
        print(f"\n=========================================================")
        print(f" NOW PREPROCESSING: {filename}")
        print(f"\n=========================================================")
        print(f"Dispatching job for: {input_video}")
    
        try:
            ##Routing and preprocessing 
            print("=======Starting  preprocessing=====")
            processed_video = route_and_preprocess(input_video)

            ## Executting the OpenFace module
            print("\n==============Starting the OpenFace Analysis===========================")
            openface_success = run_openface_feature_extraction(processed_video, output_folder)

            ## Executing the PyVHR Analysis
            print("\n=============Starting the PyVHR Analysis============================")
            pyvhr_success = run_pyvhr_extraction(processed_video, output_folder)

            ###Reporting the sccuess and failure:
            if openface_success and pyvhr_success:
                print(f"\n SUCCESS: Pipeline complete for {filename}")
            else:
                print(f"\n FAILED: Pipeline error for {filename}")
    
        except Exception as e:
            print(f"\n[Dispatcher] Critical Error on {filename}: {e}")

    
if __name__ == "__main__":
    main()