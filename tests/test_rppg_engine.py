import os
import sys
import unittest
import tempfile
import pandas as pd

# ============== PATH SETUP =================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Pointing to the new src directory structure
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src", "moxie")) 
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the Python pipeline
from rppg.pipeline import run_rppg_analysis

class TestNativeRPPGEngine(unittest.TestCase):
    
    def setUp(self):
        """Creates a clean output folder for each test."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_folder = self.test_dir.name
        
        # Pointing to the input folder which has the videos
        self.input_folder = os.path.abspath(os.path.join(current_dir, "..", "input"))

    def tearDown(self):
        """Cleans up the temporary output folder after the test is done."""
        self.test_dir.cleanup()

    #---------------------------------------------------------
    # Handling Missing Video 
    #---------------------------------------------------------
    def test_edge_case_missing_file(self):
        """Tests that the engine safely catches a missing file error."""
        bad_path = os.path.join(self.output_folder, "this_video_does_not_exist.mp4")
        
        # Expecting the new pipeline to handle this gracefully or raise a specific error
        result = run_rppg_analysis(bad_path, self.output_folder)
        self.assertFalse(result, "Engine should catch the missing file error and safely return False")

    #---------------------------------------------------------
    # Corrupted Video check
    #---------------------------------------------------------
    def test_edge_case_corrupt_video(self):
        """Tests that the engine survives if a user uploads a fake/corrupted video."""
        corrupt_path = os.path.join(self.output_folder, "corrupt.mp4")
        with open(corrupt_path, "w") as f:
            f.write("This is a text file pretending to be a video.")   
            
        result = run_rppg_analysis(corrupt_path, self.output_folder)
        self.assertFalse(result, "Engine should catch the video decoding crash and safely return False")

    
    #---------------------------------------------------------
    # Happy Path Folder processing 
    #---------------------------------------------------------
    def test_happy_path_video_extraction(self):
        """Dynamically finds valid videos and tests the native pipeline on them."""
        if not os.path.exists(self.input_folder):
            self.skipTest(f"Could not find video at {self.input_folder}")
            
        # Grab all video files in the input folder
        video_files = [f for f in os.listdir(self.input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            self.skipTest(f"No video files found inside {self.input_folder}")

        # Dynamically test every video found in the folder
        for video_name in video_files:
            with self.subTest(video=video_name):
                video_path = os.path.join(self.input_folder, video_name)
                
                # Run the NEW Native rPPG Pipeline
                result = run_rppg_analysis(video_path, self.output_folder)
                self.assertTrue(result, f"Native rPPG extraction failed on {video_name}")
                
                # Verification
                self._verify_csv_output()

    #---------------------------------------------------------
    # CSV Verification
    #---------------------------------------------------------
    def _verify_csv_output(self):
        """Helper method to ensure the pipeline wrote the HR data to a CSV."""
        csv_files = [f for f in os.listdir(self.output_folder) if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0, "No CSV file was created by the new engine")
        
        csv_path = os.path.join(self.output_folder, csv_files[0])
        df = pd.read_csv(csv_path)
        
        self.assertGreater(len(df), 0, "The generated CSV is completely empty!")
        
        # Checking for whatever columns our new pipeline will output
        has_bpm_column = any("bpm" in col.lower() for col in df.columns)
        self.assertTrue(has_bpm_column, "The CSV does not contain a BPM column!")

if __name__ == '__main__':
    unittest.main()