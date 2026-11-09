"""Unit tests for Faster R-CNN model building, head modification, loss computation, and gradient flow (Phase 3)."""

import unittest
import torch

from src.models.faster_rcnn import (
    build_faster_rcnn_mobilenet,
    build_faster_rcnn_resnet50,
    build_faster_rcnn_resnet101,
    build_model,
    get_model_summary,
)


class TestModel(unittest.TestCase):
    """Test suite for Faster R-CNN model instantiation, inference, and backward pass."""

    def test_model_build_num_classes(self):
        """Verify that FastRCNNPredictor is configured with num_classes = 2 (Background + Fracture)."""
        model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        # Regression outputs 2 classes * 4 coordinates = 8 features
        self.assertEqual(model.roi_heads.box_predictor.bbox_pred.out_features, 8)

    def test_model_build_from_config(self):
        """Verify factory build_model with configuration dictionary."""
        config = {
            "model": {
                "architecture": "fasterrcnn_resnet50_fpn",
                "pretrained": False,
                "trainable_backbone_layers": 3,
            },
            "data": {
                "classes": {
                    "num_classes": 2,
                }
            },
        }
        model = build_model(config)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)

    def test_model_summary(self):
        """Verify get_model_summary calculates correct parameter metrics."""
        model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        summary = get_model_summary(model)

        self.assertIn("total_parameters", summary)
        self.assertIn("trainable_parameters", summary)
        self.assertIn("frozen_parameters", summary)
        self.assertIn("model_size_mb", summary)
        self.assertEqual(summary["num_classes"], 2)
        self.assertGreater(summary["total_parameters"], 30_000_000)
        self.assertGreater(summary["model_size_mb"], 100.0)

    def test_model_eval_forward_pass(self):
        """Verify forward pass in eval mode outputs detection predictions."""
        model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        model.eval()

        images = [torch.rand(3, 200, 200)]
        with torch.no_grad():
            detections = model(images)

        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 1)
        self.assertIn("boxes", detections[0])
        self.assertIn("labels", detections[0])
        self.assertIn("scores", detections[0])

    def test_model_training_forward_pass(self):
        """Verify forward pass in training mode computes expected loss dictionary."""
        model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        model.train()

        # Create dummy images: 2 images of size 200x200
        images = [torch.rand(3, 200, 200), torch.rand(3, 200, 200)]

        # Targets: 1 positive image with 1 box, 1 negative image with 0 boxes
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            },
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            },
        ]

        loss_dict = model(images, targets)

        self.assertIsInstance(loss_dict, dict)
        expected_loss_keys = {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
        self.assertTrue(expected_loss_keys.issubset(set(loss_dict.keys())))

        total_loss = sum(l for l in loss_dict.values())
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

    def test_model_backward_gradient_flow(self):
        """Verify backward pass computes non-zero gradients on head parameters."""
        model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False)
        model.train()

        images = [torch.rand(3, 150, 150)]
        targets = [
            {
                "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

        loss_dict = model(images, targets)
        total_loss = sum(l for l in loss_dict.values())

        model.zero_grad()
        total_loss.backward()

        # Verify classification head weights received gradients
        cls_grad = model.roi_heads.box_predictor.cls_score.weight.grad
        self.assertIsNotNone(cls_grad)
        self.assertGreater(torch.sum(torch.abs(cls_grad)).item(), 0.0)

        # Verify bbox regression head weights received gradients
        bbox_grad = model.roi_heads.box_predictor.bbox_pred.weight.grad
        self.assertIsNotNone(bbox_grad)
        self.assertGreater(torch.sum(torch.abs(bbox_grad)).item(), 0.0)

    def test_model_alternative_backbones(self):
        """Verify instantiation of ResNet-101 and MobileNetV3 backbones (EXP-03)."""
        # ResNet101
        m_r101 = build_faster_rcnn_resnet101(num_classes=2, pretrained=False)
        self.assertEqual(m_r101.roi_heads.box_predictor.cls_score.out_features, 2)

        # MobileNetV3
        m_mob = build_faster_rcnn_mobilenet(num_classes=2, pretrained=False)
        self.assertEqual(m_mob.roi_heads.box_predictor.cls_score.out_features, 2)


if __name__ == "__main__":
    unittest.main()
