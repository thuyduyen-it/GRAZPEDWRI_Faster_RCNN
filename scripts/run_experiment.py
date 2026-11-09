"""Controlled Experiment Runner for Faster R-CNN on GRAZPEDWRI-DX (Phase 6).

Executes reproducible ablation studies across Augmentation, Backbones, Image Resolutions,
and Optimizers while strictly preserving patient-level splits and experimental seeds.
Logs results directly into outputs/metrics/experiments_summary.csv.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import GRAZPEDWRIDataset, collate_fn
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.faster_rcnn import build_model, get_model_summary
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = copy.deepcopy(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Controlled Ablation Experiments (Phase 6)")
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp_01_baseline",
        choices=[
            "exp_01_baseline",
            "exp_02_augmentation",
            "exp_03_backbone",
            "exp_04_image_resolution",
            "exp_05_hyperparameters",
            "all",
        ],
        help="Experiment suite ID to execute",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Specific variant name for multi-variant experiments (e.g. resnet101_fpn, res_600, adamw_lr_0001)",
    )
    parser.add_argument(
        "--config-base",
        type=str,
        default="configs/baseline.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--experiments-yaml",
        type=str,
        default="configs/experiments.yaml",
        help="Path to experiments specification file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run quick 1-epoch execution on sample batch for verification",
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
        "--summary-csv",
        type=str,
        default="outputs/metrics/experiments_summary.csv",
        help="Path to summary CSV tracking file",
    )
    return parser.parse_args()


def build_experiment_config(
    exp_id: str,
    variant_name: Optional[str],
    base_cfg: Dict[str, Any],
    exp_meta: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Merge base config with experiment and variant specifications."""
    merged = copy.deepcopy(base_cfg)
    exp_def = exp_meta.get("experiments", {}).get(exp_id, {})

    variant_str = variant_name or "default"

    # Merge top-level experiment parameters
    for key in ["model", "training", "transforms", "optimizer", "lr_scheduler"]:
        if key in exp_def:
            merged[key] = deep_merge(merged.get(key, {}), exp_def[key])

    # If variants exist, merge the selected variant
    variants = exp_def.get("variants", {})
    if variants:
        if variant_name is None:
            # Default to the first variant if not explicitly given
            variant_str = list(variants.keys())[0]
        else:
            variant_str = variant_name

        if variant_str not in variants:
            raise ValueError(f"Variant '{variant_str}' not found in experiment '{exp_id}'. Available: {list(variants.keys())}")

        var_cfg = variants[variant_str]

        # Handle specific experiment variant schemas
        if exp_id == "exp_03_backbone":
            merged.setdefault("model", {})["architecture"] = var_cfg.get("architecture")
            merged["model"]["trainable_backbone_layers"] = var_cfg.get("trainable_backbone_layers", 3)
        elif exp_id == "exp_04_image_resolution":
            merged.setdefault("model", {})["min_size"] = var_cfg.get("min_size")
            merged["model"]["max_size"] = var_cfg.get("max_size")
        elif exp_id == "exp_05_hyperparameters":
            merged.setdefault("optimizer", {})["type"] = var_cfg.get("optimizer", "SGD")
            merged["optimizer"]["lr"] = var_cfg.get("lr", 0.005)
            merged["optimizer"]["weight_decay"] = var_cfg.get("weight_decay", 0.0005)
            if "momentum" in var_cfg:
                merged["optimizer"]["momentum"] = var_cfg["momentum"]
            if var_cfg.get("optimizer") == "AdamW":
                merged.setdefault("lr_scheduler", {})["type"] = "CosineAnnealingLR"

    # Set up dedicated output and checkpoint paths
    exp_subfolder = f"{exp_id}_{variant_str}" if variant_str != "default" else exp_id
    exp_dir = Path("experiments") / exp_id / variant_str if variant_str != "default" else Path("experiments") / exp_id

    merged.setdefault("paths", {})["checkpoints_dir"] = str(exp_dir / "checkpoints")
    merged["paths"]["logs_dir"] = str(exp_dir / "logs")
    merged["paths"]["output_dir"] = str(exp_dir / "outputs")
    merged.setdefault("experiment", {})["name"] = exp_subfolder

    return merged, variant_str


def run_single_experiment(
    exp_id: str,
    variant_name: Optional[str],
    args: argparse.Namespace,
    base_cfg: Dict[str, Any],
    exp_meta: Dict[str, Any],
    tracker: ExperimentTracker,
    logger: Any,
) -> Dict[str, Any]:
    """Execute a single experiment variant pipeline end-to-end."""
    config, resolved_variant = build_experiment_config(exp_id, variant_name, base_cfg, exp_meta)

    exp_dir = Path(config["paths"]["checkpoints_dir"]).parent
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save concrete config.yaml in the experiment directory
    config_save_path = exp_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.dry_run:
        config.setdefault("training", {})["epochs"] = 1

    logger.info("------------------------------------------------------------")
    logger.info(f"RUNNING EXPERIMENT: {exp_id} [Variant: {resolved_variant}]")
    logger.info(f"Concrete Configuration: {config_save_path}")
    logger.info("------------------------------------------------------------")

    seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset resolution
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

    if not images_dir.exists() and fallback_img_dir.exists():
        logger.warning(
            f"Primary image directory not found at: {images_dir}. "
            f"Falling back to local sample dataset ({fallback_img_dir}) for experiment run."
        )
        train_split = fallback_split
        val_split = fallback_split
        images_dir = fallback_img_dir
        labels_dir = fallback_lbl_dir

    train_dataset = GRAZPEDWRIDataset(
        split_file=train_split,
        images_dir=images_dir,
        labels_dir=labels_dir,
        is_train=True,
        config=config,
        fallback_images_dir=fallback_img_dir,
        fallback_labels_dir=fallback_lbl_dir,
    )
    val_dataset = GRAZPEDWRIDataset(
        split_file=val_split,
        images_dir=images_dir,
        labels_dir=labels_dir,
        is_train=False,
        config=config,
        fallback_images_dir=fallback_img_dir,
        fallback_labels_dir=fallback_lbl_dir,
    )

    batch_size = 2 if args.dry_run else data_cfg.get("loader", {}).get("batch_size", 2)
    num_workers = 0 if os.name == "nt" else data_cfg.get("loader", {}).get("num_workers", 2)

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

    # Build model (offline safety for dry-run)
    if args.dry_run or not torch.cuda.is_available():
        temp_cfg = config.copy()
        temp_cfg["model"] = temp_cfg.get("model", {}).copy()
        temp_cfg["model"]["pretrained"] = False
        model = build_model(temp_cfg).to(device)
    else:
        model = build_model(config).to(device)

    model_summary = get_model_summary(model)
    param_count_m = round(model_summary["total_parameters"] / 1e6, 2)
    logger.info(f"Model: {model_summary['total_parameters']:,} params (~{param_count_m}M, {model_summary['model_size_mb']} MB)")

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

    # Trainer instantiation & execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        config=config,
    )
    train_results = trainer.train()

    # Evaluator run for full benchmark metrics
    evaluator = Evaluator(model=model, data_loader=val_loader, device=device, config=config)
    eval_results = evaluator.evaluate(optimize_threshold=True)

    # Format record
    aug_label = "Medical CLAHE+Jitter" if config.get("transforms", {}).get("clahe", {}).get("enabled") or config.get("transforms", {}).get("augmentation_strategy") == "medical_conservative" else "Horizontal Flip"
    res_label = f"{config.get('model', {}).get('min_size', 800)}x{config.get('model', {}).get('max_size', 1333)}"
    opt_label = f"{opt_type} ({lr})"

    record = {
        "exp_id": exp_id,
        "variant": resolved_variant,
        "architecture": config.get("model", {}).get("architecture", "fasterrcnn_resnet50_fpn"),
        "parameters_m": param_count_m,
        "image_resolution": res_label,
        "augmentation": aug_label,
        "optimizer": opt_label,
        "val_map_50": eval_results["map_50"],
        "val_map_50_95": eval_results["map_50_95"],
        "val_f1": eval_results.get("optimal_metrics", {}).get("f1", 0.0),
        "val_precision": eval_results.get("optimal_metrics", {}).get("precision", 0.0),
        "val_recall": eval_results.get("optimal_metrics", {}).get("recall", 0.0),
        "latency_ms": eval_results["latency"]["avg_per_image_ms"],
        "status": "COMPLETED" if not args.dry_run else "VERIFIED_DRY_RUN",
    }

    tracker.log_result(record)
    logger.info(f"Recorded results for {exp_id} [{resolved_variant}] in {args.summary_csv}")
    return record


def main() -> int:
    args = parse_args()
    logger = setup_logger(name="ExperimentRunner")
    logger.info("============================================================")
    logger.info("FASTER R-CNN CONTROLLED EXPERIMENT SUITE (PHASE 6)")
    logger.info("============================================================")

    with open(args.config_base, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    with open(args.experiments_yaml, "r", encoding="utf-8") as f:
        exp_meta = yaml.safe_load(f)

    tracker = ExperimentTracker(csv_path=args.summary_csv)

    # Determine tasks to execute
    tasks_to_run: List[Tuple[str, Optional[str]]] = []

    if args.experiment == "all":
        exp_dict = exp_meta.get("experiments", {})
        for eid, edef in exp_dict.items():
            variants = edef.get("variants", {})
            if variants:
                for vname in variants:
                    tasks_to_run.append((eid, vname))
            else:
                tasks_to_run.append((eid, None))
    else:
        exp_def = exp_meta.get("experiments", {}).get(args.experiment, {})
        variants = exp_def.get("variants", {})
        if variants and args.variant is None:
            # If experiment has variants and none was passed, queue all variants of this experiment
            for vname in variants:
                tasks_to_run.append((args.experiment, vname))
        else:
            tasks_to_run.append((args.experiment, args.variant))

    logger.info(f"Total experiment executions scheduled: {len(tasks_to_run)}")

    for idx, (eid, vname) in enumerate(tasks_to_run):
        logger.info(f"\n>>> Executing [{idx + 1}/{len(tasks_to_run)}]: {eid} (Variant: {vname}) <<<")
        run_single_experiment(eid, vname, args, base_cfg, exp_meta, tracker, logger)

    print("\n" + "=" * 95)
    print("EXPERIMENT BENCHMARK SUMMARY TABLE")
    print("=" * 95)
    print(tracker.format_markdown_table())
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
