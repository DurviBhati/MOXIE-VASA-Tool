import cv2
import numpy as np

class VideoReader:
    """Reads a video file and returns its frames and frame rate."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path

    def read_frames(self) -> tuple[np.ndarray, float]:
        """Returns a tuple of (frames_array, fps)."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video at: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR (OpenCV default) to RGB (Required for rPPG)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        return np.array(frames), fps