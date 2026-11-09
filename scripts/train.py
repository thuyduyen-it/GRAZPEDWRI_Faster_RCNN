"""Main training entrypoint for Faster R-CNN on GRAZPEDWRI-DX (Phase 4).

Supports configuration-driven training, automatic mixed precision,
learning rate scheduling, validation checkpoints, early stopping,
TensorBoard logging, and checkpoint resumption.
"""

import argparse
import os
from pathlib import Path
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GRAZPEDWRIDataset, collate_fn
from src.engine.trainer import Trainer
from src.models.faster_rcnn import build_model
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on GRAZPEDWRI-DX (Phase 4)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override dataset root path (or set GRAZPEDWRI_DATA_ROOT)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 quick epoch to verify training pipeline end-to-end",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(name="TrainScript")
    logger.info("============================================================")
    logger.info("FASTER R-CNN TRAINING ENGINE (PHASE 4)")
    logger.info("============================================================")
    logger.info(f"Loading configuration from: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.dry_run:
        config.setdefault("training", {})["epochs"] = 1
        logger.info("DRY-RUN MODE ACTIVATED: Setting epochs=1 for pipeline verification.")

    # Random seed for reproducibility
    seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(seed)
    logger.info(f"Experimental seed set to: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # Resolve dataset paths
    data_cfg = config.get("data", {})
    env_root = os.environ.get("GRAZPEDWRI_DATA_ROOT")
    dataset_root = Path(args.data_root or env_root or data_cfg.get("root_dir", "/content/dataset"))
    images_dir = dataset_root / data_cfg.get("images_dir", "images")
    labels_dir = dataset_root / data_cfg.get("labels_dir", "labels")

    train_split = Path(data_cfg.get("metadata", {}).get("train_split", "data/splits/train.txt"))
    val_split = Path(data_cfg.get("metadata", {}).get("val_split", "data/splits/val.txt"))

    fallback_img_dir = Path("data/sample_dataset/images")
    fallback_lbl_dir = Path("data/sample_dataset/labels")
    fallback_split = Path("data/sample_dataset/sample_split.txt")

    # If full dataset is external and not yet mounted locally (e.g. running dry-run in VS Code)
    if not images_dir.exists() and fallback_img_dir.exists():
        logger.warning(
            f"Primary image directory not found at: {images_dir}. "
            "Falling back to local sample dataset (data/sample_dataset) for pipeline execution."
        )
        train_split = fallback_split
        val_split = fallback_split
        images_dir = fallback_img_dir
        labels_dir = fallback_lbl_dir

    if not train_split.exists():
        logger.error(f"Cannot initialize training: Split file {train_split} missing!")
        return 1

    # Instantiate Datasets
    train_dataset = GRAZPEDWRIDataset(
        split_file=train_split,
        images_dir=images_dir,
        labels_dir=labels_dir,
        is_train=True,
        fallback_images_dir=fallback_img_dir,
        fallback_labels_dir=fallback_lbl_dir,
    )

    val_dataset = GRAZPEDWRIDataset(
        split_file=val_split,
        images_dir=images_dir,
        labels_dir=labels_dir,
        is_train=False,
        fallback_images_dir=fallback_img_dir,
        fallback_labels_dir=fallback_lbl_dir,
    )
    logger.info(f"Datasets initialized: Train={len(train_dataset)}, Val={len(val_dataset)}")

    loader_cfg = data_cfg.get("loader", {})
    batch_size = 2 if args.dry_run else loader_cfg.get("batch_size", 2)
    num_workers = 0 if os.name == "nt" else loader_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Build model (disable COCO download attempt if running dry-run on offline/sandboxed environment)
    if args.dry_run or not torch.cuda.is_available():
        temp_cfg = config.copy()
        temp_cfg["model"] = temp_cfg.get("model", {}).copy()
        temp_cfg["model"]["pretrained"] = False
        model = build_model(temp_cfg).to(device)
    else:
        model = build_model(config).to(device)

    # Optimizer
    opt_cfg = config.get("optimizer", {})
    opt_type = opt_cfg.get("type", "SGD").upper()
    lr = opt_cfg.get("lr", 0.005)
    momentum = opt_cfg.get("momentum", 0.9)
    weight_decay = opt_cfg.get("weight_decay", 0.0005)

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_type == "ADAMW":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "ADAM":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # LR Scheduler
    sched_cfg = config.get("lr_scheduler", {})
    sched_type = sched_cfg.get("type", "StepLR").lower()
    if sched_type in ["cosineannealinglr", "cosine"]:
        total_epochs = config.get("training", {}).get("epochs", 20)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    else:
        step_size = sched_cfg.get("step_size", 6)
        gamma = sched_cfg.get("gamma", 0.1)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Trainer instantiation
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        config=config,
    )

    # Checkpoint resumption if requested
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    results = trainer.train()
    logger.info(f"Training finished with best metric: {results.get('best_metric', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
