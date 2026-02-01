import os
import sys
from video_engine import run_openface_feature_extraction


def main():
    """
    Main Dispatcher:
    This function identifies the input video.
    checks if output folder exists.
    and finally alls the VideoEngine module.
    """
    
    print("=== MOXIE-VASA Dispatcher v0.1 ===")
    
    #--------------------------------------------------------------------
    # Fining configerations (only for the test video for now)
    #-------------------------------------------------------------------------
    # Define the input viedo path and create the output directory accordingly 
    
    input_video = f"/mnt/c/Users/durvi/University of Michigan Dropbox/Durvi Bhati/moxie_vas_tool/input/WIN_20260131_19_56_23_Pro.mp4"
    ##Creating the output folder automatically 
    video_dir = os.path.dirname(input_video)
    output_folder = os.path.join(video_dir, "Processed_results")    
    
    #--------------------------------------------------------------------------------------------------------
    # Validating the existence of the video
    #-------------------------------------------------------------------------------------------------------------
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at: {input_video}")
        sys.exit(1)
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    #------------------------------------------------------------------------------
    # Executing the dispatcher 
    #-------------------------------------------------------------------------------

    print(f"Dispatching job for: {input_video}")
    success = run_openface_feature_extraction(input_video, output_folder)
    
    if success:
        print("\n=== Video for processed successfully ===")  
        print(f"Results saved to: {output_folder}")
    else:
        print("\n=== UHH OHHH!! There was some error in processing the video ===")

if __name__ == "__main__":
    main()
