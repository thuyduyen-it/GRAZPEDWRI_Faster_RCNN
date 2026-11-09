"""Unit tests for FracturePredictor inference engine (Phase 7)."""

from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import torch

from src.inference.predictor import FracturePredictor


class TestPredictor(unittest.TestCase):
    """Test suite for Phase 7 inference API, bit-depth handling, and visual overlays."""

    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

        self.config = {
            "model": {
                "architecture": "fasterrcnn_resnet50_fpn",
                "pretrained": False,
                "trainable_backbone_layers": 3,
            }
        }
        self.predictor = FracturePredictor(config=self.config, confidence_threshold=0.5)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_predict_uint8_grayscale(self):
        """Verify inference on 2D uint8 grayscale image."""
        img_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = self.predictor.predict(img_gray)

        self.assertIn("has_fracture", result)
        self.assertIn("num_fractures", result)
        self.assertIn("boxes", result)
        self.assertIn("scores", result)
        self.assertIn("latency_ms", result)
        self.assertEqual(result["image_size"], (100, 100))
        self.assertGreater(result["latency_ms"], 0.0)

    def test_predict_uint16_radiograph(self):
        """Verify inference on native 16-bit uint16 image array."""
        img_16bit = np.random.randint(0, 65535, (120, 120), dtype=np.uint16)
        result = self.predictor.predict(img_16bit)

        self.assertEqual(result["image_size"], (120, 120))
        self.assertIsInstance(result["has_fracture"], bool)
        self.assertIn("max_confidence", result)

    def test_predict_from_file_path(self):
        """Verify prediction from an image file saved to disk."""
        img_sample = np.random.randint(0, 255, (80, 80), dtype=np.uint8)
        file_path = self.tmp_dir / "sample_xray.png"
        cv2.imwrite(str(file_path), img_sample)

        result = self.predictor.predict(file_path)
        self.assertEqual(result["image_size"], (80, 80))

    def test_predict_batch(self):
        """Verify batch prediction across multiple image inputs."""
        img1 = np.zeros((60, 60), dtype=np.uint8)
        img2 = np.ones((70, 70), dtype=np.uint8) * 128
        results = self.predictor.predict_batch([img1, img2])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["image_size"], (60, 60))
        self.assertEqual(results[1]["image_size"], (70, 70))

    def test_predict_and_visualize(self):
        """Verify annotated visual overlay with clinical status banner."""
        img = np.random.randint(0, 255, (90, 90), dtype=np.uint8)
        vis_img, result = self.predictor.predict_and_visualize(img)

        self.assertIsInstance(vis_img, np.ndarray)
        self.assertEqual(vis_img.ndim, 3)
        self.assertEqual(vis_img.shape[2], 3)
        # Visual output includes header banner (+36px height)
        self.assertGreater(vis_img.shape[0], 90)
        self.assertEqual(vis_img.shape[1], 90)


if __name__ == "__main__":
    unittest.main()
