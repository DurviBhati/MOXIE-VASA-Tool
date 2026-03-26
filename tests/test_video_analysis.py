import os
import sys
import unittest
import tempfile
import pandas as pd

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src", "moxie")) 
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.Moxie.openface.video_engine import run_openface_feature_extraction

class TestVideoAnalysisReal(unittest.TestCase):
    
    def setUp(self):
        """Creates a clean output folder for each test."""
        # ignore_cleanup_errors=True stops the test from crashing if Docker locks the files
        self.test_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.output_folder = self.test_dir.name
        
        # Point dynamically to the input folder
        self.input_folder = "/mnt/c/Users/durvi/University of Michigan Dropbox/Durvi Bhati/moxie_vas_tool/input"

    def tearDown(self):
        """Cleans up the temporary output folder after the test is done."""
        self.test_dir.cleanup()

    # ---------------------------------------------------------
    # Handling Missing Video
    # ---------------------------------------------------------
    def test_edge_case_missing_file(self):
        """Tests that the OpenFace engine safely catches a missing file error."""
        bad_path = os.path.join(self.output_folder, "ghost_video.mp4")
        result = run_openface_feature_extraction(bad_path, self.output_folder)
        self.assertFalse(result, "Engine should catch the missing file error and safely return False")

    # ---------------------------------------------------------
    # Handling Corrupted Video 
    # ---------------------------------------------------------
    def test_edge_case_corrupt_video(self):
        """Tests that OpenFace survives if a user uploads a fake/corrupted video."""
        corrupt_path = os.path.join(self.output_folder, "corrupt.mp4")
        with open(corrupt_path, "w") as f:
            f.write("This is a text file pretending to be a video.")
            
        result = run_openface_feature_extraction(corrupt_path, self.output_folder)
        self.assertFalse(result, "Engine should catch the video decoding crash and safely return False")

    # ---------------------------------------------------------
    # Folder Processing
    # ---------------------------------------------------------
    def test_dynamic_input_folder(self):
        """Dynamically finds all videos in the input folder and runs OpenFace on them."""
        if not os.path.exists(self.input_folder):
            self.skipTest(f"Input folder not found at {self.input_folder}")

        video_files = [f for f in os.listdir(self.input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            self.skipTest(f"No video files found inside {self.input_folder}")

        for video_name in video_files:
            with self.subTest(video=video_name):
                video_path = os.path.join(self.input_folder, video_name)
                
                #Run OpenFace Analysis
                result = run_openface_feature_extraction(video_path, self.output_folder)
                self.assertTrue(result, f"Video analysis failed on {video_name}")
                
                #Verification
                self._verify_openface_csv()

    # ---------------------------------------------------------
    # OpenFace CSV Verification
    # ---------------------------------------------------------
    def _verify_openface_csv(self):
        """Ensures OpenFace actually extracted facial data and wrote it to a CSV."""
        csv_files = [f for f in os.listdir(self.output_folder) if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0, "No CSV file was created by OpenFace")
        
        csv_path = os.path.join(self.output_folder, csv_files[-1])
        df = pd.read_csv(csv_path)
        
        self.assertGreater(len(df), 0, "The generated OpenFace CSV is completely empty!")
        
       
        # Checking for few meningful columns to make sure the output is good
        columns_lower = [col.lower().strip() for col in df.columns]
        is_valid_openface = any(expected in columns_lower for expected in ['frame', 'confidence', 'face_id'])
        
        self.assertTrue(is_valid_openface, "The CSV does not contain standard OpenFace columns (frame, confidence)!")

if __name__ == '__main__':
    unittest.main()