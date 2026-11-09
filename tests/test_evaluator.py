"""Unit tests for Evaluation engine and metric computation (Phase 5)."""

import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.engine.evaluator import Evaluator
from src.models.faster_rcnn import build_faster_rcnn_resnet50
from src.utils.metrics import (
    DetectionEvaluator,
    compute_average_precision,
    compute_precision_recall_f1,
    evaluate_threshold_sweep,
)


class MockEvalDataset(Dataset):
    """Small mock dataset for testing evaluation loop without external files."""

    def __init__(self, size: int = 2):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        img = torch.rand(3, 50, 50)
        if idx % 2 == 0:
            # Positive sample with 1 fracture bounding box
            target = {
                "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        else:
            # Negative sample (no fracture bounding box)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        return img, target


def mock_collate(batch):
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)


class TestEvaluator(unittest.TestCase):
    """Test suite for Phase 5 detection evaluation and clinical metrics."""

    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

        self.model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        self.dataset = MockEvalDataset(size=2)
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False, collate_fn=mock_collate)

        self.config = {
            "evaluation": {
                "confidence_threshold": 0.1,
                "iou_thresholds": [0.5, 0.75],
            }
        }

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_precision_recall_f1_calculation(self):
        """Verify precision, recall, and f1 formula implementation."""
        res = compute_precision_recall_f1(tp=8, fp=2, fn=2)
        self.assertAlmostEqual(res["precision"], 0.8, places=4)
        self.assertAlmostEqual(res["recall"], 0.8, places=4)
        self.assertAlmostEqual(res["f1"], 0.8, places=4)

        # Zero edge case
        zero_res = compute_precision_recall_f1(tp=0, fp=0, fn=0)
        self.assertAlmostEqual(zero_res["precision"], 0.0, places=4)

    def test_average_precision_perfect_match(self):
        """Verify AP equals 1.0 when predictions match ground truth boxes perfectly."""
        preds = [{
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
            "scores": torch.tensor([0.95], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }]
        targets = [{
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }]
        ap = compute_average_precision(preds, targets, target_class=1, iou_threshold=0.5)
        self.assertAlmostEqual(ap, 1.0, places=3)

    def test_average_precision_negative_sample_fp(self):
        """Verify that predictions on negative images are penalized as false positives."""
        preds = [{
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
            "scores": torch.tensor([0.90], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }]
        targets = [{
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }]
        # Total GT is 0, so AP should be 0.0
        ap = compute_average_precision(preds, targets, target_class=1, iou_threshold=0.5)
        self.assertEqual(ap, 0.0)

    def test_threshold_sweep(self):
        """Verify threshold sweep table generation and optimal operating point selection."""
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
                "scores": torch.tensor([0.85], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor([[15.0, 15.0, 35.0, 35.0]], dtype=torch.float32),
                "scores": torch.tensor([0.35], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            },
        ]
        sweep = evaluate_threshold_sweep(preds, targets, thresholds=[0.3, 0.5, 0.8])
        self.assertIn("sweep_table", sweep)
        self.assertIn("optimal_threshold", sweep)
        self.assertIn("optimal_metrics", sweep)
        self.assertEqual(len(sweep["sweep_table"]), 3)

    def test_evaluator_end_to_end(self):
        """Verify Evaluator class executes forward pass, latency measurement, and metrics export."""
        evaluator = Evaluator(
            model=self.model,
            data_loader=self.loader,
            config=self.config,
        )
        diag_dir = self.tmp_dir / "diagnostics"
        metrics = evaluator.evaluate(
            save_diagnostics=True,
            diagnostics_dir=str(diag_dir),
            max_diagnostics=2,
        )

        self.assertIn("map_50", metrics)
        self.assertIn("map_50_95", metrics)
        self.assertIn("latency", metrics)
        self.assertIn("avg_per_image_ms", metrics["latency"])
        self.assertIn("optimal_threshold", metrics)
        self.assertIn("optimal_metrics", metrics)
        self.assertEqual(metrics["total_images"], 2)


if __name__ == "__main__":
    unittest.main()
