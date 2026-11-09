"""Dataset preparation, verification, and DataLoader sanity script (Phase 2).

Tests instantiation of GRAZPEDWRIDataset, verifies YOLO-to-Faster-RCNN conversion,
checks true negative empty target handling, validates DataLoader batching,
and generates visual sample overlays with ground-truth bounding boxes.
"""

import argparse
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GRAZPEDWRIDataset, collate_fn
from src.utils.logging import setup_logger
from src.utils.visualization import draw_bounding_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Dataset Loader & Preprocessing Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split to verify",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader test",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=5,
        help="Number of individual samples to inspect",
    )
    parser.add_argument(
        "--export-samples",
        action="store_true",
        default=True,
        help="Export visual inspection images to outputs/predictions/dataset_samples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("PrepareDatasetCLI")
    logger.info("============================================================")
    logger.info("STARTING PHASE 2: DATA PREPARATION & LOADER VERIFICATION")
    logger.info("============================================================")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    env_root = os.environ.get("GRAZPEDWRI_DATA_ROOT")
    dataset_root = Path(env_root or data_cfg.get("root_dir", "/content/dataset"))

    images_dir = dataset_root / data_cfg.get("images_dir", "images")
    labels_dir = dataset_root / data_cfg.get("labels_dir", "labels")

    split_key = f"{args.split}_split"
    split_file = Path(data_cfg.get("metadata", {}).get(split_key, f"data/splits/{args.split}.txt"))

    fallback_img_dir = Path("data/sample_dataset/images")
    fallback_lbl_dir = Path("data/sample_dataset/labels")

    logger.info("Resolving Dataset Paths:")
    logger.info(f"  * Split File: {split_file} (exists: {split_file.exists()})")
    logger.info(f"  * Primary Images: {images_dir} (exists: {images_dir.exists()})")
    logger.info(f"  * Primary Labels: {labels_dir} (exists: {labels_dir.exists()})")
    logger.info(f"  * Fallback Images: {fallback_img_dir} (exists: {fallback_img_dir.exists()})")

    # If full split points to external images that don't exist locally, create a quick sample split for testing
    active_split_file = split_file
    if not images_dir.exists() and fallback_img_dir.exists():
        logger.info("Primary image root not mounted locally. Using sample dataset for loader testing.")
        sample_split_file = Path("data/sample_dataset/sample_split.txt")
        sample_files = sorted(list(fallback_img_dir.glob("*.png")))
        with open(sample_split_file, "w", encoding="utf-8") as f:
            for s in sample_files:
                f.write(f"{s.name}\n")
        active_split_file = sample_split_file
        images_dir = fallback_img_dir
        labels_dir = fallback_lbl_dir

    dataset = GRAZPEDWRIDataset(
        split_file=active_split_file,
        images_dir=images_dir,
        labels_dir=labels_dir,
        is_train=(args.split == "train"),
        fallback_images_dir=fallback_img_dir,
        fallback_labels_dir=fallback_lbl_dir,
    )

    total_samples = len(dataset)
    logger.info(f"Dataset successfully initialized. Total samples: {total_samples}")

    # Inspect individual samples
    num_to_check = min(args.verify_samples, total_samples)
    logger.info(f"Inspecting first {num_to_check} samples individually...")

    out_viz_dir = Path("outputs/predictions/dataset_samples")
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    positive_count = 0
    negative_count = 0

    for i in range(num_to_check):
        img_tensor, target = dataset[i]

        # Verify tensor constraints
        assert isinstance(img_tensor, torch.Tensor), f"Sample {i}: Expected torch.Tensor"
        assert img_tensor.dtype == torch.float32, f"Sample {i}: Expected float32, got {img_tensor.dtype}"
        assert img_tensor.ndim == 3 and img_tensor.shape[0] == 3, f"Sample {i}: Expected shape [3, H, W], got {img_tensor.shape}"
        assert 0.0 <= img_tensor.min() and img_tensor.max() <= 1.0, f"Sample {i}: Values outside [0, 1] range"

        # Verify target fields
        required_keys = {"boxes", "labels", "area", "iscrowd", "image_id", "orig_size"}
        missing_keys = required_keys - set(target.keys())
        assert not missing_keys, f"Sample {i}: Missing target keys {missing_keys}"

        num_boxes = len(target["boxes"])
        if num_boxes > 0:
            positive_count += 1
            # Verify Faster R-CNN label offset (Fracture = 1)
            assert (target["labels"] == 1).all(), f"Sample {i}: Labels must be 1 (Fracture)"
            assert target["boxes"].shape[1] == 4, f"Sample {i}: Bounding boxes must have 4 coordinates"
        else:
            negative_count += 1
            assert target["boxes"].shape == (0, 4), f"Sample {i}: Negative boxes must have shape [0, 4]"
            assert target["labels"].shape == (0,), f"Sample {i}: Negative labels must have shape [0]"

        logger.info(
            f"  [Sample {i}] File: {target.get('filename', f'idx_{i}')} | "
            f"Shape: {list(img_tensor.shape)} | Range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}] | "
            f"Boxes: {num_boxes} | Type: {'Positive (Fracture)' if num_boxes > 0 else 'Negative (Healthy)'}"
        )

        # Export sample visualization if requested
        if args.export_samples:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            boxes = target["boxes"].tolist()
            labels = ["Fracture"] * len(boxes) if boxes else []
            if boxes:
                annotated = draw_bounding_boxes(img_np, boxes, labels=labels, color=(0, 255, 0), thickness=2)
            else:
                annotated = img_np.copy()
                cv2.putText(
                    annotated,
                    "Negative Sample (No Fracture)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 200, 0),
                    2,
                )
            out_file = out_viz_dir / f"loader_verified_{args.split}_{i}.png"
            cv2.imwrite(str(out_file), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Test DataLoader batching and collate_fn
    logger.info("Testing PyTorch DataLoader batching & collate_fn...")
    batch_size = min(args.batch_size, total_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    start_time = time.time()
    batch_images, batch_targets = next(iter(loader))
    elapsed = time.time() - start_time

    assert len(batch_images) == batch_size, f"Expected {batch_size} images, got {len(batch_images)}"
    assert len(batch_targets) == batch_size, f"Expected {batch_size} targets, got {len(batch_targets)}"

    total_batch_boxes = sum(len(t["boxes"]) for t in batch_targets)
    logger.info(f"Successfully assembled batch of {batch_size} items in {elapsed*1000:.1f}ms:")
    logger.info(f"  * Batch Size: {len(batch_images)}")
    logger.info(f"  * Total Fracture Bounding Boxes in Batch: {total_batch_boxes}")
    logger.info(f"  * Image 0 Tensor: {batch_images[0].shape}, dtype={batch_images[0].dtype}")

    logger.info("============================================================")
    logger.info("PHASE 2 DATASET LOADER VERIFICATION: ALL CHECKS PASSED [OK]")
    logger.info("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
