"""Unit tests for medical image preprocessing, normalization, and transforms (Phase 2)."""

import unittest
import numpy as np
import torch

from src.data.transforms import (
    MedicalCLAHE,
    MedicalCompose,
    RandomBrightnessContrast,
    RandomHorizontalFlip,
    ToTensor,
    get_transforms,
)


class TestTransforms(unittest.TestCase):
    """Test suite for image transformations and bounding box consistency."""

    def test_to_tensor_16bit_uint16(self):
        """Verify that 16-bit uint16 images are normalized by 65535.0 to [0.0, 1.0]."""
        transform = ToTensor()
        # Create 1-channel 16-bit image with known pixel values
        img_16u = np.array([[0, 32767], [65535, 16383]], dtype=np.uint16)
        tensor, _ = transform(img_16u, None)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        # Should be expanded to 3 channels for backbone
        self.assertEqual(tensor.shape, (3, 2, 2))
        self.assertAlmostEqual(tensor[0, 0, 0].item(), 0.0, delta=1e-4)
        self.assertAlmostEqual(tensor[0, 0, 1].item(), 32767.0 / 65535.0, delta=1e-4)
        self.assertAlmostEqual(tensor[0, 1, 0].item(), 1.0, delta=1e-4)
        # Verify all 3 channels are identical copies
        self.assertTrue(torch.allclose(tensor[0], tensor[1]))
        self.assertTrue(torch.allclose(tensor[0], tensor[2]))

    def test_to_tensor_8bit_uint8(self):
        """Verify that 8-bit uint8 images are normalized by 255.0 to [0.0, 1.0]."""
        transform = ToTensor()
        img_8u = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        tensor, _ = transform(img_8u, None)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape, (3, 2, 2))
        self.assertAlmostEqual(tensor[0, 0, 0].item(), 0.0, delta=1e-4)
        self.assertAlmostEqual(tensor[0, 1, 0].item(), 1.0, delta=1e-4)

    def test_random_horizontal_flip_positive_boxes(self):
        """Verify exact coordinate reversal during horizontal flip."""
        flip = RandomHorizontalFlip(prob=1.0)  # Always flip
        width, height = 200, 150
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Original box: [x_min=20, y_min=30, x_max=60, y_max=80]
        original_box = torch.tensor([[20.0, 30.0, 60.0, 80.0]], dtype=torch.float32)
        target = {
            "boxes": original_box.clone(),
            "labels": torch.tensor([1], dtype=torch.int64),
            "area": torch.tensor([2000.0], dtype=torch.float32),
        }

        flipped_img, flipped_target = flip(img, target)

        # Expected new coordinates:
        # new_xmin = width - old_xmax = 200 - 60 = 140
        # new_xmax = width - old_xmin = 200 - 20 = 180
        # y coordinates remain identical
        expected_box = [140.0, 30.0, 180.0, 80.0]
        self.assertAlmostEqual(flipped_target["boxes"][0, 0].item(), expected_box[0], delta=1e-3)
        self.assertAlmostEqual(flipped_target["boxes"][0, 1].item(), expected_box[1], delta=1e-3)
        self.assertAlmostEqual(flipped_target["boxes"][0, 2].item(), expected_box[2], delta=1e-3)
        self.assertAlmostEqual(flipped_target["boxes"][0, 3].item(), expected_box[3], delta=1e-3)

        # Area should remain identical
        expected_area = (180.0 - 140.0) * (80.0 - 30.0)
        self.assertAlmostEqual(flipped_target["area"][0].item(), expected_area, delta=1e-3)

    def test_random_horizontal_flip_negative_boxes(self):
        """Verify that horizontal flip handles true negative samples (0 boxes) without error."""
        flip = RandomHorizontalFlip(prob=1.0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        flipped_img, flipped_target = flip(img, target)
        self.assertEqual(flipped_target["boxes"].shape, (0, 4))
        self.assertEqual(flipped_target["labels"].shape, (0,))

    def test_medical_clahe_uint16(self):
        """Verify CLAHE on native 16-bit uint16 image preserves dtype and range."""
        clahe_transform = MedicalCLAHE(clip_limit=2.0, prob=1.0)
        img_16u = np.linspace(0, 65535, 400, dtype=np.uint16).reshape(20, 20)
        enhanced, _ = clahe_transform(img_16u, None)

        self.assertEqual(enhanced.dtype, np.uint16)
        self.assertEqual(enhanced.shape, (20, 20))
        self.assertGreaterEqual(enhanced.min(), 0)
        self.assertLessEqual(enhanced.max(), 65535)

    def test_random_brightness_contrast_bounds(self):
        """Verify brightness/contrast keeps values inside valid range."""
        adjust = RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, prob=1.0)
        img_16u = np.full((50, 50), 30000, dtype=np.uint16)
        adjusted, _ = adjust(img_16u, None)

        self.assertEqual(adjusted.dtype, np.uint16)
        self.assertGreaterEqual(adjusted.min(), 0)
        self.assertLessEqual(adjusted.max(), 65535)

    def test_get_transforms_pipeline(self):
        """Verify get_transforms factory returns valid MedicalCompose instances."""
        train_transforms = get_transforms(is_train=True)
        val_transforms = get_transforms(is_train=False)

        self.assertIsInstance(train_transforms, MedicalCompose)
        self.assertIsInstance(val_transforms, MedicalCompose)

        # Apply to a dummy image
        img = np.zeros((100, 100), dtype=np.uint8)
        tensor_train, _ = train_transforms(img, None)
        tensor_val, _ = val_transforms(img, None)

        self.assertEqual(tensor_train.shape, (3, 100, 100))
        self.assertEqual(tensor_val.shape, (3, 100, 100))


if __name__ == "__main__":
    unittest.main()
