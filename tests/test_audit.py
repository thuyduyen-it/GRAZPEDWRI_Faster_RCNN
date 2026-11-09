"""Unit tests for dataset audit and validation functions (Phase 1)."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src.data.audit import (
    check_patient_leakage,
    check_split_overlap,
    profile_image_properties,
    validate_annotation_file,
    validate_yolo_line,
)


class TestDatasetAudit(unittest.TestCase):
    """Test suite for audit validation functions."""

    def test_validate_yolo_line_valid(self):
        """Verify parsing a valid YOLO line."""
        is_valid, msg, box = validate_yolo_line("0 0.5 0.5 0.2 0.2")
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
        self.assertIsNotNone(box)
        self.assertEqual(box["class_id"], 0)
        self.assertAlmostEqual(box["x_center"], 0.5)

    def test_validate_yolo_line_malformed(self):
        """Verify rejection of malformed lines with missing fields."""
        is_valid, msg, box = validate_yolo_line("0 0.5 0.5")
        self.assertFalse(is_valid)
        self.assertIn("Expected 5 fields", msg)
        self.assertIsNone(box)

    def test_validate_yolo_line_invalid_coords(self):
        """Verify rejection of coordinates out of [0, 1] range."""
        is_valid, msg, box = validate_yolo_line("0 1.5 0.5 0.2 0.2")
        self.assertFalse(is_valid)
        self.assertIn("out of normalized range", msg)

    def test_validate_yolo_line_non_positive_dim(self):
        """Verify rejection of zero or negative width/height."""
        is_valid, msg, box = validate_yolo_line("0 0.5 0.5 0.0 0.2")
        self.assertFalse(is_valid)
        self.assertIn("Non-positive box dimensions", msg)

    def test_validate_annotation_file_empty(self):
        """Verify empty annotation file (true negative) is considered valid with 0 boxes."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp_path = Path(tmp.name)

        try:
            res = validate_annotation_file(tmp_path)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["num_boxes"], 0)
            self.assertEqual(len(res["errors"]), 0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_annotation_file_positive(self):
        """Verify annotation file with valid bounding boxes."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp.write("0 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.2 0.2\n")
            tmp_path = Path(tmp.name)

        try:
            res = validate_annotation_file(tmp_path)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["num_boxes"], 2)
            self.assertEqual(res["classes"].get(0), 2)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_check_split_overlap_clean(self):
        """Verify split overlap returns clean when sets are disjoint."""
        train = ["img1.png", "img2.png", "img3.png"]
        val = ["img4.png", "img5.png"]
        test = ["img6.png", "img7.png"]

        res = check_split_overlap(train, val, test)
        self.assertTrue(res["is_clean"])
        self.assertEqual(res["total_overlap"], 0)

    def test_check_split_overlap_detected(self):
        """Verify split overlap flags leaked images across sets."""
        train = ["img1.png", "img2.png", "img3.png"]
        val = ["img2.png", "img4.png"]  # Leaked img2.png
        test = ["img5.png"]

        res = check_split_overlap(train, val, test)
        self.assertFalse(res["is_clean"])
        self.assertEqual(res["overlap_train_val"], 1)
        self.assertEqual(res["total_overlap"], 1)

    def test_check_patient_leakage_clean(self):
        """Verify patient leakage check passes when all patients are in distinct splits."""
        df = pd.DataFrame([
            {"patient_id": "0001", "split": "train"},
            {"patient_id": "0002", "split": "train"},
            {"patient_id": "0003", "split": "val"},
            {"patient_id": "0004", "split": "test"},
        ])
        res = check_patient_leakage(df)
        self.assertTrue(res["is_leak_free"])
        self.assertEqual(res["leakage_count"], 0)

    def test_check_patient_leakage_detected(self):
        """Verify patient leakage check detects patient occurring in multiple splits."""
        df = pd.DataFrame([
            {"patient_id": "0001", "split": "train"},
            {"patient_id": "0001", "split": "val"},  # Leaked patient
            {"patient_id": "0002", "split": "test"},
        ])
        res = check_patient_leakage(df)
        self.assertFalse(res["is_leak_free"])
        self.assertEqual(res["leakage_count"], 1)
        self.assertIn("0001", res["leaking_patient_ids"])

    def test_profile_image_properties(self):
        """Verify image metadata profiling (16-bit uint16)."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".png") as tmp:
            tmp_path = Path(tmp.name)
            # Create dummy 16-bit image
            img_16u = np.full((100, 80), 30000, dtype=np.uint16)
            cv2.imwrite(str(tmp_path), img_16u)

        try:
            prof = profile_image_properties(tmp_path)
            self.assertEqual(prof["width"], 80)
            self.assertEqual(prof["height"], 100)
            self.assertEqual(prof["bit_depth"], 16)
            self.assertEqual(prof["dtype"], "uint16")
            self.assertEqual(prof["channels"], 1)
            self.assertAlmostEqual(prof["min_intensity"], 30000.0)
            self.assertAlmostEqual(prof["max_intensity"], 30000.0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
