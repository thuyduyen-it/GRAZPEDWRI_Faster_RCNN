"""Unit tests for Controlled Experiment Runner and Tracker (Phase 6)."""

import copy
from pathlib import Path
import tempfile
import unittest
import yaml

from scripts.run_experiment import build_experiment_config, deep_merge
from src.utils.experiment_tracker import ExperimentTracker


class TestExperiments(unittest.TestCase):
    """Test suite for Phase 6 experimental strategy, configuration merging, and metric tracking."""

    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

        self.base_cfg = {
            "experiment": {"name": "baseline", "seed": 42},
            "model": {
                "architecture": "fasterrcnn_resnet50_fpn",
                "trainable_backbone_layers": 3,
                "min_size": 800,
                "max_size": 1333,
            },
            "training": {"epochs": 20, "batch_size": 2},
            "optimizer": {"type": "SGD", "lr": 0.005, "weight_decay": 0.0005},
            "lr_scheduler": {"type": "StepLR", "step_size": 6, "gamma": 0.1},
            "transforms": {"augmentation": {"horizontal_flip_prob": 0.5}},
            "paths": {
                "checkpoints_dir": "checkpoints",
                "logs_dir": "outputs/logs",
                "output_dir": "outputs",
            },
        }

        self.exp_meta = {
            "experiments": {
                "exp_01_baseline": {
                    "name": "Baseline ResNet-50-FPN",
                },
                "exp_02_augmentation": {
                    "name": "Medical Image Domain Augmentation",
                    "transforms": {
                        "augmentation_strategy": "medical_conservative",
                        "clahe": {"enabled": True, "clip_limit": 2.0},
                    },
                },
                "exp_03_backbone": {
                    "name": "Backbone Comparison",
                    "variants": {
                        "resnet101_fpn": {
                            "architecture": "fasterrcnn_resnet101_fpn",
                            "trainable_backbone_layers": 3,
                        },
                        "mobilenet_v3_large_fpn": {
                            "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
                            "trainable_backbone_layers": 3,
                        },
                    },
                },
                "exp_04_image_resolution": {
                    "name": "Resolution Comparison",
                    "variants": {
                        "res_600": {"min_size": 600, "max_size": 1000},
                        "res_1024": {"min_size": 1024, "max_size": 1600},
                    },
                },
                "exp_05_hyperparameters": {
                    "name": "Optimizer Tuning",
                    "variants": {
                        "adamw_lr_0001": {
                            "optimizer": "AdamW",
                            "lr": 0.0001,
                            "weight_decay": 0.01,
                        },
                    },
                },
            }
        }

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_deep_merge(self):
        """Verify recursive dictionary merging preserves untouched keys."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        update = {"b": {"d": 4, "e": 5}, "f": 6}
        merged = deep_merge(base, update)

        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["b"]["c"], 2)
        self.assertEqual(merged["b"]["d"], 4)
        self.assertEqual(merged["b"]["e"], 5)
        self.assertEqual(merged["f"], 6)

    def test_build_experiment_config_baseline(self):
        """Verify baseline experiment configuration merging."""
        cfg, variant = build_experiment_config(
            exp_id="exp_01_baseline",
            variant_name=None,
            base_cfg=self.base_cfg,
            exp_meta=self.exp_meta,
        )
        self.assertEqual(variant, "default")
        self.assertEqual(cfg["model"]["architecture"], "fasterrcnn_resnet50_fpn")
        self.assertIn("exp_01_baseline", cfg["paths"]["checkpoints_dir"])

    def test_build_experiment_config_backbone_variant(self):
        """Verify backbone ablation configuration (EXP-03)."""
        cfg, variant = build_experiment_config(
            exp_id="exp_03_backbone",
            variant_name="mobilenet_v3_large_fpn",
            base_cfg=self.base_cfg,
            exp_meta=self.exp_meta,
        )
        self.assertEqual(variant, "mobilenet_v3_large_fpn")
        self.assertEqual(cfg["model"]["architecture"], "fasterrcnn_mobilenet_v3_large_fpn")
        self.assertIn("mobilenet_v3_large_fpn", cfg["paths"]["checkpoints_dir"])

    def test_build_experiment_config_resolution_variant(self):
        """Verify resolution ablation configuration (EXP-04)."""
        cfg, variant = build_experiment_config(
            exp_id="exp_04_image_resolution",
            variant_name="res_600",
            base_cfg=self.base_cfg,
            exp_meta=self.exp_meta,
        )
        self.assertEqual(variant, "res_600")
        self.assertEqual(cfg["model"]["min_size"], 600)
        self.assertEqual(cfg["model"]["max_size"], 1000)

    def test_build_experiment_config_hyperparameters_variant(self):
        """Verify optimizer ablation configuration (EXP-05)."""
        cfg, variant = build_experiment_config(
            exp_id="exp_05_hyperparameters",
            variant_name="adamw_lr_0001",
            base_cfg=self.base_cfg,
            exp_meta=self.exp_meta,
        )
        self.assertEqual(variant, "adamw_lr_0001")
        self.assertEqual(cfg["optimizer"]["type"], "AdamW")
        self.assertEqual(cfg["optimizer"]["lr"], 0.0001)
        self.assertEqual(cfg["lr_scheduler"]["type"], "CosineAnnealingLR")

    def test_experiment_tracker(self):
        """Verify ExperimentTracker CSV logging, update, and markdown table generation."""
        csv_path = self.tmp_dir / "test_summary.csv"
        tracker = ExperimentTracker(csv_path=str(csv_path))

        record_1 = {
            "exp_id": "exp_01_baseline",
            "variant": "default",
            "architecture": "fasterrcnn_resnet50_fpn",
            "parameters_m": 41.35,
            "image_resolution": "800x1333",
            "augmentation": "Horizontal Flip",
            "optimizer": "SGD (0.005)",
            "val_map_50": 0.7240,
            "val_map_50_95": 0.4510,
            "val_f1": 0.7650,
            "val_precision": 0.7810,
            "val_recall": 0.7500,
            "latency_ms": 42.5,
            "status": "COMPLETED",
        }
        tracker.log_result(record_1)

        summary = tracker.get_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["exp_id"], "exp_01_baseline")
        self.assertEqual(summary[0]["val_map_50"], "0.724")

        # Test updating record
        record_1_updated = copy.deepcopy(record_1)
        record_1_updated["val_map_50"] = 0.7350
        tracker.log_result(record_1_updated)

        summary_after = tracker.get_summary()
        self.assertEqual(len(summary_after), 1)
        self.assertEqual(summary_after[0]["val_map_50"], "0.735")

        # Test markdown table formatting
        md = tracker.format_markdown_table()
        self.assertIn("exp_01_baseline", md)
        self.assertIn("41.35", md)


if __name__ == "__main__":
    unittest.main()
