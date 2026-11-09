"""Unit tests for annotation parsing, conversion, and validation."""

import tempfile
import unittest
from pathlib import Path
import torch

from src.data.annotation import is_valid_box, parse_yolo_annotation_file, yolo_to_xyxy


class TestAnnotations(unittest.TestCase):
    """Test suite for YOLO annotation parsing and bounding box operations."""

    def test_yolo_to_xyxy_conversion(self):
        """Verify conversion of YOLO normalized (xc, yc, w, h) to absolute pixel [xmin, ymin, xmax, ymax]."""
        img_w, img_h = 1000, 800
        # Center at (500, 400), width=200, height=100
        yolo_box = (0.5, 0.5, 0.2, 0.125)
        xyxy = yolo_to_xyxy(yolo_box, img_w, img_h)

        self.assertAlmostEqual(xyxy[0], 400.0, delta=1e-2)
        self.assertAlmostEqual(xyxy[1], 350.0, delta=1e-2)
        self.assertAlmostEqual(xyxy[2], 600.0, delta=1e-2)
        self.assertAlmostEqual(xyxy[3], 450.0, delta=1e-2)

    def test_yolo_to_xyxy_clamping(self):
        """Verify that bounding boxes near edges are clamped to image boundaries."""
        img_w, img_h = 100, 100
        # Box protruding outside left boundary
        yolo_box = (0.05, 0.5, 0.2, 0.2)
        xyxy = yolo_to_xyxy(yolo_box, img_w, img_h, clamp=True)

        self.assertGreaterEqual(xyxy[0], 0.0)
        self.assertLessEqual(xyxy[2], 100.0)

    def test_is_valid_box(self):
        """Verify bounding box validation logic."""
        self.assertTrue(is_valid_box([10, 10, 50, 50]))
        # Zero width
        self.assertFalse(is_valid_box([10, 10, 10, 50]))
        # Inverted coordinates
        self.assertFalse(is_valid_box([50, 50, 10, 10]))

    def test_empty_annotation_negative_sample(self):
        """Verify that empty annotation files return empty target tensors (negative sample)."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp_path = Path(tmp.name)

        try:
            target = parse_yolo_annotation_file(tmp_path, img_width=500, img_height=500)
            self.assertIsInstance(target["boxes"], torch.Tensor)
            self.assertEqual(target["boxes"].shape, (0, 4))
            self.assertEqual(target["labels"].shape, (0,))
            self.assertEqual(target["area"].shape, (0,))
            self.assertEqual(target["iscrowd"].shape, (0,))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_parse_yolo_annotation_positive(self):
        """Verify parsing a valid YOLO annotation line and class ID mapping (0 -> 1)."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp.write("0 0.5 0.5 0.2 0.2\n")
            tmp_path = Path(tmp.name)

        try:
            target = parse_yolo_annotation_file(
                tmp_path, img_width=1000, img_height=1000, yolo_class_offset=1
            )
            self.assertEqual(target["boxes"].shape, (1, 4))
            # YOLO class 0 mapped to Faster R-CNN label 1 (Fracture)
            self.assertEqual(target["labels"].tolist(), [1])
            self.assertEqual(target["boxes"][0].tolist(), [400.0, 400.0, 600.0, 600.0])
            self.assertAlmostEqual(target["area"].item(), 40000.0, delta=1.0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
