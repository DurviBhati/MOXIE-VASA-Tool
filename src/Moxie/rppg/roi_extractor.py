import cv2
import numpy as np
import mediapipe as mp

class ROIExtractor:
    """Extracts the skin Region of Interest (ROI) using MediaPipe Face Mesh."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize the FaceMesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Approximate MediaPipe landmark indices for forehead and cheeks (ignoring eyes/mouth)
        self.skin_indices = [
            10, 338, 297, 332, 284,  # Forehead
            234, 93, 132, 58, 172,   # Left cheek area
            454, 323, 361, 288, 397  # Right cheek area
        ]

    def get_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """Returns a binary mask of the skin ROI for a given frame."""
        h, w, _ = frame.shape
        results = self.face_mesh.process(frame)
        mask = np.zeros((h, w), dtype=np.uint8)

        # If no face is found, return the empty mask
        if not results.multi_face_landmarks:
            return mask

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Map normalized landmarks to pixel coordinates
        roi_points = []
        for idx in self.skin_indices:
            point = landmarks[idx]
            x, y = int(point.x * w), int(point.y * h)
            roi_points.append((x, y))
            
        # Create a boundary (convex hull) around the skin points and fill it
        hull = cv2.convexHull(np.array(roi_points))
        cv2.fillConvexPoly(mask, hull, 1)
        
        return mask