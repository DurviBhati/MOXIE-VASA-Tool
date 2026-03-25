import os
import sys
import tempfile
import unittest

import pandas as pd  # type: ignore

from Moxie.openface.validator import NoFaceDetectedError, validate_openface_output

# Path Setup to import from src/Moxie
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

class TestValidateOpenfaceOutput(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def _write_csv(self, filename, df):
        path = os.path.join(self.test_dir.name, filename)
        df.to_csv(path, index=False)
        return path

    # Test 1: All-zero confidence (black/corrupted video)
    def test_zero_confidence_raises_error(self):
        """A CSV full of zero confidence values should raise NoFaceDetectedError."""
        df = pd.DataFrame({
            "frame": [1, 2, 3],
            "face_id": [0, 0, 0],
            "confidence": [0.0, 0.0, 0.0],
        })
        csv_path = self._write_csv("zeros.csv", df)
        with self.assertRaises(NoFaceDetectedError):
            validate_openface_output(csv_path)

    # Test 2: Good data passes validation
    def test_good_confidence_passes(self):
        """A CSV with high confidence values should pass without raising."""
        df = pd.DataFrame({
            "frame": [1, 2, 3],
            "face_id": [0, 0, 0],
            "confidence": [0.92, 0.87, 0.95],
        })
        csv_path = self._write_csv("good.csv", df)
        # Should not raise
        validate_openface_output(csv_path)

    # Test 3: File does not exist
    def test_missing_file_raises_error(self):
        """A path pointing to a nonexistent file should raise FileNotFoundError."""
        fake_path = os.path.join(self.test_dir.name, "ghost.csv")
        with self.assertRaises(FileNotFoundError):
            validate_openface_output(fake_path)

    # Test 4: Empty file (0 bytes)
    def test_empty_file_raises_error(self):
        """A 0-byte file should raise FileNotFoundError."""
        empty_path = os.path.join(self.test_dir.name, "empty.csv")
        open(empty_path, "w").close()
        with self.assertRaises(FileNotFoundError):
            validate_openface_output(empty_path)


if __name__ == "__main__":
    unittest.main()
