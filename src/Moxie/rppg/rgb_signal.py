import numpy as np
from .roi_extractor import ROIExtractor

class RGBSignalExtractor:
    """Calculates the spatial average of RGB pixels within the skin ROI across frames."""
    
    def __init__(self):
        self.roi_extractor = ROIExtractor()

    def extract_signal(self, frames: np.ndarray) -> np.ndarray:
        """
        Processes a video array to extract raw RGB traces.
        Returns a NumPy array of shape (num_frames, 3).
        """
        num_frames = len(frames)
        rgb_signals = np.zeros((num_frames, 3))

        for i in range(num_frames):
            frame = frames[i]
            mask = self.roi_extractor.get_skin_mask(frame)
            
            # If face is lost, carry over the last known good signal to prevent spikes
            if np.sum(mask) == 0:
                if i > 0:
                    rgb_signals[i] = rgb_signals[i-1]
                continue

            # Extract just the skin pixels: returns an array of shape (N_pixels, 3)
            skin_pixels = frame[mask == 1]
            
            # Calculate the spatial mean of R, G, B for this frame
            rgb_signals[i] = np.mean(skin_pixels, axis=0)

        return rgb_signals