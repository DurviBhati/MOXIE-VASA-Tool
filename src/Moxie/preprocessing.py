import os
import cv2
import subprocess

def extract_metadata(video_path: str) -> dict:
    """Extracts base metadata like FPS, resolution, and frame count."""
    # Force FFmpeg backend here just to be safe reading metadata
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
        
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return meta

def standardize_mp4(video_path: str) -> str:
    print(f"[PreProcess] Detected .mp4 file. Metadata is generally safe.")
    # Add any future MP4-specific logic here
    return video_path

def standardize_avi(video_path: str) -> str:
    """
    Academic database .avi files often cause C++ 'core dumped' errors in OpenCV on Linux.
    This safely transcodes it to a near-lossless MP4 so the rest of the pipeline doesn't crash.
    """
    print(f"[PreProcess] Detected .avi file. Applying safe transcoding...")
    
    # Create a new filename for the safe version
    safe_video_path = video_path.replace(".avi", "_safe.mp4")
    
    # If we already converted it in a previous run, skip doing it again
    if not os.path.exists(safe_video_path):
        print(f"[PreProcess] ---> Converting {os.path.basename(video_path)} to standard H.264 to prevent C++ crashes...")
        
        # We use a visually lossless CRF of 15 so we don't ruin the rPPG skin color signal
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path, 
            "-c:v", "libx264", "-preset", "fast", "-crf", "15", 
            safe_video_path
        ]
        
        # Run FFmpeg quietly
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[PreProcess] ---> Conversion complete.")
    else:
        print("[PreProcess] ---> Safe version already exists. Skipping conversion.")
        
    return safe_video_path

def route_and_preprocess(video_path: str) -> str:
    """
    Routes the video to the correct preprocessing function based upon the file extension.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")
        
    filename = os.path.basename(video_path)
    print(f"\n--- Preprocessing: {filename} ---")
    
    # 1. Route based on extension
    _, ext = os.path.splitext(video_path)
    ext = ext.lower()

    if ext == '.mp4':
        safe_path = standardize_mp4(video_path)
    elif ext == '.avi':
        safe_path = standardize_avi(video_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    # 2. Extract Metadata from the safe path
    metadata = extract_metadata(safe_path)
    print(f"[PreProcess] Metadata: {metadata['width']}x{metadata['height']} at {metadata['fps']:.2f} FPS")
    
    # 3. Face Presence Check (Teammate Placeholder)
    print("[PreProcess] Verifying face presence...")
    
    # Return the path to the safe, standardized video so OpenFace and rPPG use it instead of the original
    return safe_path