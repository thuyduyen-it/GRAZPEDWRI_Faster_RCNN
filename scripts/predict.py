"""Inference CLI script for pediatric wrist fracture detection (Phase 7).

Predicts fracture bounding boxes on individual radiographs or batch directories.
Exports annotated visual diagnostic images, structured JSON, and CSV summaries.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import List
import cv2
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import FracturePredictor
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Pediatric Wrist Fractures on X-rays (Phase 7)")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single X-ray image file (.png, .jpg, .bmp)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory containing X-ray images for batch prediction",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence score detection threshold [0.0 - 1.0]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save visualized predictions and reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(name="PredictScript")
    logger.info("============================================================")
    logger.info("FASTER R-CNN FRACTURE INFERENCE ENGINE (PHASE 7)")
    logger.info("============================================================")

    if args.image is None and args.input_dir is None:
        # If neither is provided, check fallback sample images
        fallback_dir = Path("data/sample_dataset/images")
        if fallback_dir.exists():
            logger.warning(f"No --image or --input-dir provided. Using sample directory: {fallback_dir}")
            args.input_dir = str(fallback_dir)
        else:
            logger.error("Please provide either --image or --input-dir.")
            return 1

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.warning(
            f"Checkpoint not found at: {checkpoint_path}. "
            "Running with freshly initialized weights (useful for pipeline verification)."
        )
        checkpoint_to_load = None
    else:
        checkpoint_to_load = str(checkpoint_path)

    predictor = FracturePredictor(
        checkpoint_path=checkpoint_to_load,
        config=config,
        confidence_threshold=args.threshold,
    )

    # Collect images
    image_paths: List[Path] = []
    if args.image:
        p = Path(args.image)
        if not p.exists():
            logger.error(f"Image not found: {p}")
            return 1
        image_paths.append(p)
    elif args.input_dir:
        dir_p = Path(args.input_dir)
        if not dir_p.exists():
            logger.error(f"Input directory not found: {dir_p}")
            return 1
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_paths.extend(dir_p.glob(ext))

    image_paths.sort()
    logger.info(f"Loaded {len(image_paths)} image(s) for fracture inference.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    csv_rows = []

    for img_p in image_paths:
        vis_img, result = predictor.predict_and_visualize(img_p)

        save_img_path = output_dir / f"pred_{img_p.stem}.png"
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(str(save_img_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        # Omit large numpy array from JSON record
        record = {k: v for k, v in result.items() if k != "image_rgb"}
        record["image_path"] = str(img_p)
        record["output_visualization"] = str(save_img_path)
        all_results[img_p.name] = record

        csv_rows.append({
            "image_name": img_p.name,
            "has_fracture": result["has_fracture"],
            "num_fractures": result["num_fractures"],
            "max_confidence": result["max_confidence"],
            "boxes": json.dumps(result["boxes"]),
            "latency_ms": result["latency_ms"],
        })

        status_str = f"FRACTURE ({result['num_fractures']})" if result["has_fracture"] else "NORMAL"
        logger.info(
            f"[{status_str}] {img_p.name} | Latency: {result['latency_ms']:.1f}ms | "
            f"Boxes: {result['boxes']} (Conf: {result['scores']})"
        )

    # Save JSON report
    json_path = output_dir / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV report
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "has_fracture", "num_fractures", "max_confidence", "boxes", "latency_ms"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info(f"Visual predictions saved to: {output_dir}")
    logger.info(f"Structured JSON exported to: {json_path}")
    logger.info(f"Summary CSV exported to:     {csv_path}")

    # Summary Stats
    total_imgs = len(image_paths)
    fractures_detected = sum(1 for r in all_results.values() if r["has_fracture"])
    avg_lat = sum(r["latency_ms"] for r in all_results.values()) / max(1, total_imgs)

    print("\n" + "=" * 70)
    print("INFERENCE RUN SUMMARY")
    print("=" * 70)
    print(f"Total Radiographs Processed: {total_imgs}")
    print(f"Fractures Detected:          {fractures_detected} ({fractures_detected/max(1, total_imgs)*100:.1f}%)")
    print(f"Normal / Negative Cases:     {total_imgs - fractures_detected}")
    print(f"Average Inference Latency:   {avg_lat:.2f} ms / image")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
