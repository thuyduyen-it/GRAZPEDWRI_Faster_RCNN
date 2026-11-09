"""Unit tests for GRAZPEDWRIDataset and DataLoader pipeline."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GRAZPEDWRIDataset, collate_fn


class TestDataset(unittest.TestCase):
    """Test suite for dataset indexing and batch collation."""

    def setUp(self):
        """Create a temporary dataset with 1 positive sample and 1 negative sample."""
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir_obj.name)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.splits_dir = self.root / "splits"

        self.images_dir.mkdir()
        self.labels_dir.mkdir()
        self.splits_dir.mkdir()

        # Create dummy positive image and annotation
        pos_img = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(self.images_dir / "sample_pos.png"), pos_img)
        with open(self.labels_dir / "sample_pos.txt", "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Create dummy negative image (empty annotation)
        neg_img = np.zeros((400, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(self.images_dir / "sample_neg.png"), neg_img)
        with open(self.labels_dir / "sample_neg.txt", "w", encoding="utf-8") as f:
            pass  # Empty file

        # Create split file containing both
        self.split_file = self.splits_dir / "test_split.txt"
        with open(self.split_file, "w", encoding="utf-8") as f:
            f.write("/content/dataset/images/sample_pos.png\n")
            f.write("sample_neg.png\n")

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_dataset_positive_and_negative_loading(self):
        """Verify loading positive sample (with boxes) and negative sample (without boxes)."""
        dataset = GRAZPEDWRIDataset(
            split_file=self.split_file,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            is_train=False,
        )

        self.assertEqual(len(dataset), 2)

        # Positive sample
        img_pos, target_pos = dataset[0]
        self.assertIsInstance(img_pos, torch.Tensor)
        self.assertEqual(img_pos.shape, (3, 400, 300))
        self.assertEqual(target_pos["boxes"].shape, (1, 4))
        self.assertEqual(target_pos["labels"].tolist(), [1])

        # Negative sample
        img_neg, target_neg = dataset[1]
        self.assertIsInstance(img_neg, torch.Tensor)
        self.assertEqual(img_neg.shape, (3, 400, 300))
        self.assertEqual(target_neg["boxes"].shape, (0, 4))
        self.assertEqual(target_neg["labels"].shape, (0,))

    def test_dataloader_collate_fn(self):
        """Verify DataLoader batching with custom collate_fn."""
        dataset = GRAZPEDWRIDataset(
            split_file=self.split_file,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            is_train=False,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        images, targets = next(iter(loader))

        self.assertEqual(len(images), 2)
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["boxes"].shape, (1, 4))
        self.assertEqual(targets[1]["boxes"].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
