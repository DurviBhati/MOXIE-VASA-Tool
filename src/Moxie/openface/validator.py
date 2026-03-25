import os

import pandas as pd  # type: ignore


class NoFaceDetectedError(Exception):
    """Raised when an OpenFace output CSV contains no valid face detections."""
    pass


def validate_openface_output(csv_path: str) -> None:
    """
    Validates an OpenFace output CSV to ensure it contains real face detections.

    Arguments:
        csv_path (str): Full path to the OpenFace output CSV file.

    Raises:
        FileNotFoundError: If the file does not exist or is 0 bytes.
        NoFaceDetectedError: If the CSV is empty or average confidence is below 0.4.
    """
    # Check 1: File existence and non-zero size
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Validator] OpenFace output not found: {csv_path}")

    if os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"[Validator] OpenFace output is 0 bytes: {csv_path}")

    # Check 2: Non-empty CSV with valid face confidence
    df = pd.read_csv(csv_path)

    if df.empty:
        raise NoFaceDetectedError(
            f"[Validator] CSV is empty — OpenFace found no frames: {csv_path}"
        )

    # Normalize column names to handle leading/trailing whitespace
    df.columns = [col.strip() for col in df.columns]

    if "confidence" not in df.columns:
        raise NoFaceDetectedError(
            f"[Validator] 'confidence' column missing from CSV: {csv_path}"
        )

    avg_confidence = df["confidence"].mean()

    if avg_confidence < 0.4:
        raise NoFaceDetectedError(
            f"[Validator] Average confidence {avg_confidence:.3f} is below threshold 0.4"
            f"— video may be black, corrupted, or missing a face: {csv_path}"
        )
