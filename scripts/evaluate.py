"""Evaluation script for trained Faster R-CNN models on GRAZPEDWRI-DX (Phase 5).

Computes standardized detection metrics:
- mAP@0.5 and mAP@0.5:0.95 (COCO standard)
- Precision, Recall, and F1 across confidence threshold sweeps
- Optimal operating threshold identification on Validation set
- Clinical False Positive analysis on True Negative radiographs
- Inference latency benchmarking (ms/image, FPS)
- Diagnostic visual overlays comparing Ground Truth vs Predictions
"""

import argparse
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GRAZPEDWRIDataset, collate_fn
from src.engine.evaluator import Evaluator
from src.models.faster_rcnn import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on GRAZPEDWRI-DX (Phase 5)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split (val for threshold calibration, test for final benchmark)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override dataset root path (or set GRAZPEDWRI_DATA_ROOT)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional confidence threshold filter (default from config or sweep)",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Sweep confidence thresholds to calibrate optimal clinical operating point",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Export visual diagnostic image overlays (GT vs Prediction)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/metrics",
        help="Directory to save evaluation metric JSON artifacts",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default="outputs/predictions/eval_diagnostics",
        help="Directory to save diagnostic visual overlays",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(name="EvaluateScript")
    logger.info("============================================================")
    logger.info("FASTER R-CNN MODEL EVALUATION & BENCHMARK (PHASE 5)")
    logger.info("============================================================")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Target Split:  {args.split}")
    logger.info(f"Checkpoint:    {args.checkpoint}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation Compute Device: {device} (CUDA: {torch.cuda.is_available()})")

    # Resolve dataset paths
    data_cfg = config.get("data", {})
    env_root = os.environ.get("GRAZPEDWRI_DATA_ROOT")
    dataset_root = Path(args.data_root or env_root or data_cfg.get("root_dir", "/content/dataset"))

    images_dir = dataset_root / data_cfg.get("images_dir", "images")
    labels_dir = dataset_root / data_cfg.get("labels_dir", "labels")

    split_file = Path(data_cfg.get("metadata", {}).get(f"{args.split}_split", f"data/splits/{args.split}.txt"))

    fallback_img_dir = Path("data/sample_dataset/images")
    fallback_lbl_dir = Path("data/sample_dataset/labels")
    fallback_split = Path("data/sample_dataset/sample_split.txt")

    if not images_dir.exists() and fallback_img_dir.exists():
        logger.warning(
            f"Primary image directory not found at: {images_dir}. "
            f"Falling back to local sample dataset ({fallback_img_dir}) for evaluation execution."
        )
        split_file = fallback_split
        images_dir = fallback_img_dir
        labels_dir = fallback_lbl_dir

    if not split_file.exists():
        logger.error(f"Cannot initialize evaluation: Split file {split_file} missing!")
        return 1

    dataset = GRAZPEDWRIDataset(
        split_file=split_file,
        images_dir=images_dir,
        labels_dir=labels_dir,
        is_train=False,
        fallback_images_dir=fallback_img_dir,
        fallback_labels_dir=fallback_lbl_dir,
    )
    logger.info(f"Evaluation dataset loaded: {len(dataset)} images from {split_file}")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Build and load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return 1

    # In local offline mode without GPU, instantiate model without COCO download
    if not torch.cuda.is_available():
        temp_cfg = config.copy()
        temp_cfg["model"] = temp_cfg.get("model", {}).copy()
        temp_cfg["model"]["pretrained"] = False
        model = build_model(temp_cfg).to(device)
    else:
        model = build_model(config).to(device)

    load_checkpoint(checkpoint_path, model, device=device)
    logger.info("Checkpoint state successfully loaded into model.")

    evaluator = Evaluator(
        model=model,
        data_loader=loader,
        device=device,
        config=config,
    )

    metrics = evaluator.evaluate(
        confidence_threshold=args.confidence_threshold,
        optimize_threshold=args.optimize_threshold,
        save_diagnostics=args.save_diagnostics,
        diagnostics_dir=args.diagnostics_dir,
        max_diagnostics=10,
    )

    # Save metrics JSON artifact
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{args.split}_metrics.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation report JSON successfully written to: {out_file}")

    # Print clinical evaluation table
    print("\n" + "=" * 80)
    print(f"CLINICAL EVALUATION BENCHMARK ({args.split.upper()} SET)")
    print("=" * 80)
    print(f"Total Evaluated Images:  {metrics['total_images']}")
    print(f"mAP@0.50 (IoU=0.50):     {metrics['map_50']:.4f}")
    print(f"mAP@0.50:0.95:           {metrics['map_50_95']:.4f}")
    print(f"Inference Latency:       {metrics['latency']['avg_per_image_ms']:.2f} ms/image ({metrics['latency']['fps']:.1f} FPS)")
    print("-" * 80)
    opt = metrics.get("optimal_metrics", {})
    print(f"Calibrated Threshold:    {metrics.get('optimal_threshold', 0.5)}")
    print(f"Precision:               {opt.get('precision', 0.0):.4f}")
    print(f"Recall (Sensitivity):    {opt.get('recall', 0.0):.4f}")
    print(f"F1-Score:                {opt.get('f1', 0.0):.4f}")
    print(f"True Positives (TP):     {opt.get('tp', 0)}")
    print(f"False Positives (FP):    {opt.get('fp', 0)} (On Negative Samples: {opt.get('fp_negative_images', 0)})")
    print(f"False Negatives (FN):    {opt.get('fn', 0)}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
