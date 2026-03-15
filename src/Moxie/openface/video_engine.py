import os
import subprocess

def run_openface_feature_extraction(video_path, output_dir):
    """
    This function will run the OpenFace FeatureExtraction tool via Docker on a video.
    This is automatically grabs the correct filename from whatever the dispatcher sends.
    Arguments:
        video_path (str): The full path to the input video file.
        output_dir (str): The full path where results should be saved.
        
    Returns:
        bool: True if successful, False if the Docker command failed.
    """
    
    #---------------------------------------------------------
    # Setting up docker paths
    #---------------------------------------------------------
    # Geting the filename 
    video_filename = os.path.basename(video_path)
    ## Extracting the folder path from the full video path 
    video_folder = os.path.dirname(video_path)
    
    print(f"[VideoEngine] Preparing to process: {video_filename}")
    
    #---------------------------------------------------------
    # Designing the command 
    #---------------------------------------------------------
    # Use os.path.abspath to make sure Docker doesn't get confused by relative paths
    abs_input = os.path.abspath(video_folder)
    abs_output = os.path.abspath(output_dir)
    # This will used to give instruction to dockeer.
    # For the command to run I will have to provide a single string instructions.
    docker_internal_command = (
        f"cd /home/openface-build/build/bin && "
        f"./FeatureExtraction -f '/in/{video_filename}' -out_dir '/out' " 
        f"-aus -pose -gaze -2Dfp -3Dfp -simalign"
    )    
    command = [
        "docker", "run", "--rm", # --rm means "delete container when done" "--platform", 
        "--platform","linux/amd64", #This is done to specify the processing mechanism(though not a concern for my processor but just as a safety net) "-v",    
        "--entrypoint", "/home/openface-build/build/bin/FeatureExtraction", # Full path to the tool
        "-v", f"{abs_input}:/in", # Map the local video folder -> INSIDE /in 
        "-v", f"{abs_output}:/out",  # Map the local folder   --> Ouput/in
        "algebr/openface:latest", # The tool to use
        "-f", f"/in/{video_filename}", 
        "-out_dir", "/out",
        "-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp", "-simalign"
    ]
    
    #---------------------------------------------------------
    # Running the big command
    #---------------------------------------------------------
    try:
        print("[VideoEngine] Spinning up Docker container...")
        
        # subprocess.run executes the command in the terminal
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[VideoEngine] Success! Processing complete.")
            return True
        else:
            # If it failed, print the error from Docker
            print(f"[VideoEngine] Error Code: {result.returncode}")
            print(f"[VideoEngine] Error Logs:\n{result.stderr}")
            print(f"[VideoEngine] Output Logs:\n{result.stdout}")
            return False
            
    except Exception as e:
        print(f"[VideoEngine] Critical Failure: {e}")
        return False
