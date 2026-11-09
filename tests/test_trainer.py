"""Unit tests for Faster R-CNN Trainer engine and checkpoint recovery (Phase 4)."""

import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.engine.trainer import Trainer
from src.models.faster_rcnn import build_faster_rcnn_resnet50


class MockDataset(Dataset):
    """Small mock dataset for testing training loop without external files."""

    def __init__(self, size: int = 2):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        img = torch.rand(3, 50, 50)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        return img, target


def mock_collate(batch):
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)


class TestTrainer(unittest.TestCase):
    """Test suite for training engine, validation, and checkpoint recovery."""

    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

        self.model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        self.dataset = MockDataset(size=2)
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False, collate_fn=mock_collate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)

        self.config = {
            "training": {
                "epochs": 1,
                "mixed_precision": False,
                "clip_grad_norm": 5.0,
                "val_interval": 1,
                "save_interval": 1,
            },
            "paths": {
                "checkpoints_dir": str(self.tmp_dir / "checkpoints"),
                "logs_dir": str(self.tmp_dir / "logs"),
                "output_dir": str(self.tmp_dir / "outputs"),
            },
        }

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_trainer_initialization(self):
        """Verify Trainer object initialization and configuration attributes."""
        trainer = Trainer(
            model=self.model,
            train_loader=self.loader,
            val_loader=self.loader,
            optimizer=self.optimizer,
            config=self.config,
        )
        self.assertEqual(trainer.epochs, 1)
        self.assertEqual(trainer.start_epoch, 1)
        self.assertFalse(trainer.use_amp)

    def test_train_one_epoch(self):
        """Verify execution of one training epoch and loss computation."""
        trainer = Trainer(
            model=self.model,
            train_loader=self.loader,
            val_loader=None,
            optimizer=self.optimizer,
            config=self.config,
        )
        losses = trainer.train_one_epoch(epoch=1)

        self.assertIn("loss_total", losses)
        self.assertIn("loss_classifier", losses)
        self.assertIn("loss_box_reg", losses)
        self.assertIn("loss_objectness", losses)
        self.assertIn("loss_rpn_box_reg", losses)
        self.assertGreater(losses["loss_total"], 0.0)

    def test_checkpoint_saving_and_resumption(self):
        """Verify that checkpoint can be saved and accurately resumed."""
        trainer = Trainer(
            model=self.model,
            train_loader=self.loader,
            val_loader=None,
            optimizer=self.optimizer,
            config=self.config,
        )

        # Run 1 epoch training
        trainer.train()

        last_ckpt = Path(self.config["paths"]["checkpoints_dir"]) / "last_model.pth"
        self.assertTrue(last_ckpt.exists(), "last_model.pth was not created!")

        # Initialize fresh model and resume from checkpoint
        new_model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.005)

        resumed_trainer = Trainer(
            model=new_model,
            train_loader=self.loader,
            val_loader=None,
            optimizer=new_optimizer,
            config=self.config,
        )
        resumed_trainer.resume_from_checkpoint(last_ckpt)

        # start_epoch should be 2 after completing epoch 1
        self.assertEqual(resumed_trainer.start_epoch, 2)


if __name__ == "__main__":
    unittest.main()
