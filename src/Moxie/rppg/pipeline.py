import os
import cv2
from .rgb_signal import RGBSignalExtractor

def run_rppg_analysis(video_path: str, output_dir: str) -> bool:
    print(f"[rPPG] Starting custom analysis on {os.path.basename(video_path)}...")
    
    try:
        print("[rPPG] ---> Safely reading just 30 frames to prevent RAM crashes...")
        cap = cv2.VideoCapture(video_path)
        test_frames = []
        
        # ONLY read the first 30 frames. Do not load the whole video into RAM!
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            test_frames.append(frame)
        cap.release()
        
        print(f"[rPPG] ---> Successfully loaded {len(test_frames)} frames.")
        
        print("[rPPG] ---> Testing MediaPipe Face Mesh & Signal Extraction...")
        extractor = RGBSignalExtractor()
        raw_signals = extractor.extract_signal(test_frames)
        
        print(f"[rPPG] ---> Successfully extracted raw RGB signals! Array shape: {raw_signals.shape}")
        return True 
        
    except Exception as e:
        print(f"[rPPG] CRITICAL ERROR during test run: {e}")
        return False