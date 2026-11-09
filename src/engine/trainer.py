"""Training engine for Faster R-CNN on pediatric wrist X-rays (Phase 4).

Handles multi-task loss optimization, Automatic Mixed Precision (AMP),
gradient clipping, validation metrics evaluation, TensorBoard logging,
CSV loss history tracking, checkpoint persistence, and fault-tolerant resumption.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.engine.evaluator import Evaluator
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import setup_logger

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class Trainer:
    """Orchestrates Faster R-CNN model training, validation, logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.logger = setup_logger(name="Trainer")
        self.training_cfg = self.config.get("training", {})
        self.epochs = self.training_cfg.get("epochs", 20)
        self.start_epoch = 1

        # Automatic Mixed Precision
        self.use_amp = self.training_cfg.get("mixed_precision", True) and (self.device.type == "cuda")
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Output and Checkpoint Paths
        paths_cfg = self.config.get("paths", {})
        self.checkpoints_dir = Path(paths_cfg.get("checkpoints_dir", "checkpoints"))
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = Path(paths_cfg.get("logs_dir", "outputs/logs"))
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.history_dir = Path(paths_cfg.get("output_dir", "outputs")) / "loss_curves"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv_path = self.history_dir / "training_history.csv"

        # Initialize TensorBoard Writer
        self.writer = None
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.logs_dir))
            self.logger.info(f"TensorBoard logging enabled: {self.logs_dir}")

        # Validation Evaluator
        self.evaluator = None
        if self.val_loader is not None and len(self.val_loader) > 0:
            self.evaluator = Evaluator(
                model=self.model,
                data_loader=self.val_loader,
                device=self.device,
                config=self.config,
            )

        # Checkpoint Tracking & Early Stopping
        self.best_metric = -1.0
        self.early_stopping_cfg = self.training_cfg.get("early_stopping", {})
        self.early_stopping_enabled = self.early_stopping_cfg.get("enabled", False)
        self.patience = self.early_stopping_cfg.get("patience", 5)
        self.patience_counter = 0

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Restore training state from a saved checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint .pth file.
        """
        chk_path = Path(checkpoint_path)
        self.logger.info(f"Resuming training from checkpoint: {chk_path}")
        checkpoint = load_checkpoint(
            filepath=chk_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device,
        )

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric = checkpoint.get("best_metric", -1.0)
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.info(
            f"Successfully resumed at epoch {self.start_epoch} (Best metric: {self.best_metric:.4f})"
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch over train_loader."""
        self.model.train()
        running_losses: Dict[str, float] = {
            "loss_total": 0.0,
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
        }
        total_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.epochs}] Training")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            if self.optimizer is not None:
                self.optimizer.zero_grad()

            autocast_ctx = (
                torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp)
                if hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )
            with autocast_ctx:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if self.optimizer is not None:
                self.scaler.scale(losses).backward()

                clip_norm = self.training_cfg.get("clip_grad_norm", 10.0)
                if clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            running_losses["loss_total"] += losses.item()
            for k in loss_dict:
                if k in running_losses:
                    running_losses[k] += loss_dict[k].item()

            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        avg_losses = {k: v / max(1, total_batches) for k, v in running_losses.items()}
        return avg_losses

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop and calculate object detection metrics."""
        if self.evaluator is None:
            return {"val_map_50": 0.0, "val_map_50_95": 0.0}

        self.logger.info(f"Epoch [{epoch}/{self.epochs}] Running Validation Evaluation...")
        metrics = self.evaluator.evaluate()
        return {
            "val_map_50": metrics.get("map_50", 0.0),
            "val_map_50_95": metrics.get("map_50_95", 0.0),
        }

    def _log_metrics_to_tensorboard(
        self, epoch: int, train_losses: Dict[str, float], val_metrics: Dict[str, float], lr: float
    ) -> None:
        """Log scalars to TensorBoard."""
        if self.writer is None:
            return

        for k, v in train_losses.items():
            self.writer.add_scalar(f"Loss/{k}", v, epoch)

        for k, v in val_metrics.items():
            self.writer.add_scalar(f"Metrics/{k}", v, epoch)

        self.writer.add_scalar("Learning_Rate", lr, epoch)

    def _record_history_csv(
        self, epoch: int, train_losses: Dict[str, float], val_metrics: Dict[str, float], lr: float
    ) -> None:
        """Append epoch summary row to training_history.csv."""
        file_exists = self.history_csv_path.exists()
        with open(self.history_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "epoch",
                    "lr",
                    "loss_total",
                    "loss_classifier",
                    "loss_box_reg",
                    "loss_objectness",
                    "loss_rpn_box_reg",
                    "val_map_50",
                    "val_map_50_95",
                ])
            writer.writerow([
                epoch,
                f"{lr:.6f}",
                f"{train_losses.get('loss_total', 0.0):.4f}",
                f"{train_losses.get('loss_classifier', 0.0):.4f}",
                f"{train_losses.get('loss_box_reg', 0.0):.4f}",
                f"{train_losses.get('loss_objectness', 0.0):.4f}",
                f"{train_losses.get('loss_rpn_box_reg', 0.0):.4f}",
                f"{val_metrics.get('val_map_50', 0.0):.4f}",
                f"{val_metrics.get('val_map_50_95', 0.0):.4f}",
            ])

    def train(self) -> Dict[str, Any]:
        """Execute full training pipeline across configured epochs."""
        self.logger.info(
            f"Starting Faster R-CNN training from epoch {self.start_epoch} to {self.epochs} on {self.device} "
            f"(Mixed Precision: {self.use_amp})"
        )
        history = []
        val_interval = self.training_cfg.get("val_interval", 1)
        save_interval = self.training_cfg.get("save_interval", 5)

        for epoch in range(self.start_epoch, self.epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0

            # 1. Train one epoch
            train_losses = self.train_one_epoch(epoch)
            self.logger.info(
                f"Epoch [{epoch}/{self.epochs}] - Train Loss: {train_losses['loss_total']:.4f} "
                f"(Cls: {train_losses['loss_classifier']:.4f}, Box: {train_losses['loss_box_reg']:.4f}, "
                f"RPN-Obj: {train_losses['loss_objectness']:.4f}, RPN-Box: {train_losses['loss_rpn_box_reg']:.4f}) | "
                f"LR: {current_lr:.6f}"
            )

            # 2. Validation step
            val_metrics = {}
            if epoch % val_interval == 0:
                val_metrics = self.validate(epoch)
                self.logger.info(
                    f"Epoch [{epoch}/{self.epochs}] - Validation mAP@0.5: {val_metrics['val_map_50']:.4f}, "
                    f"mAP@0.5:0.95: {val_metrics['val_map_50_95']:.4f}"
                )

            # 3. Learning rate step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 4. Log to TensorBoard and CSV
            self._log_metrics_to_tensorboard(epoch, train_losses, val_metrics, current_lr)
            self._record_history_csv(epoch, train_losses, val_metrics, current_lr)

            # 5. Check if best model
            current_map50 = val_metrics.get("val_map_50", 0.0)
            is_best = current_map50 > self.best_metric
            if is_best:
                self.best_metric = current_map50
                self.patience_counter = 0
                self.logger.info(f"==> New best validation mAP@0.5: {self.best_metric:.4f}! Saving best_model.pth")
            else:
                self.patience_counter += 1

            # 6. Save checkpoints (last, periodic, best)
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "best_metric": self.best_metric,
                "config": self.config,
                "train_losses": train_losses,
                "val_metrics": val_metrics,
            }

            # Save last_model.pth
            save_checkpoint(checkpoint_state, self.checkpoints_dir, filename="last_model.pth", is_best=is_best)

            # Automatic Backup to Google Drive (if running in Colab with mounted Drive or configured path)
            drive_backup_dir = self.config.get("paths", {}).get("drive_backup_dir", None)
            if drive_backup_dir is None and Path("/content/drive/MyDrive").exists():
                drive_backup_dir = "/content/drive/MyDrive/GRAZPEDWRI_Dataset/checkpoints"

            if drive_backup_dir:
                try:
                    import shutil
                    drive_dir = Path(drive_backup_dir)
                    drive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.checkpoints_dir / "last_model.pth", drive_dir / "last_model.pth")
                    if is_best and (self.checkpoints_dir / "best_model.pth").exists():
                        shutil.copy2(self.checkpoints_dir / "best_model.pth", drive_dir / "best_model.pth")
                        self.logger.info(f"==> Synced best_model.pth to Google Drive: {drive_dir / 'best_model.pth'}")
                except Exception as e:
                    self.logger.warning(f"Could not backup checkpoint to Google Drive: {e}")

            # Optional: trigger direct browser download on Colab when best model is updated
            if is_best and self.training_cfg.get("auto_download_colab", False):
                try:
                    from google.colab import files
                    self.logger.info("==> Triggering Colab direct download for best_model.pth...")
                    files.download(str(self.checkpoints_dir / "best_model.pth"))
                except Exception as e:
                    self.logger.warning(f"Could not trigger Colab browser download: {e}")

            # Save periodic checkpoint
            if epoch % save_interval == 0:
                save_checkpoint(checkpoint_state, self.checkpoints_dir, filename=f"checkpoint_epoch_{epoch:02d}.pth")

            history.append({
                "epoch": epoch,
                "train_losses": train_losses,
                "val_metrics": val_metrics,
                "lr": current_lr,
            })

            # 7. Early Stopping Check
            if self.early_stopping_enabled and self.patience_counter >= self.patience:
                self.logger.warning(
                    f"Early stopping triggered at epoch {epoch} (No improvement for {self.patience} epochs). "
                    f"Best mAP@0.5: {self.best_metric:.4f}"
                )
                break

        if self.writer is not None:
            self.writer.close()

        self.logger.info(
            f"Training completed successfully. Best Validation mAP@0.5: {self.best_metric:.4f}. "
            f"Checkpoints saved to: {self.checkpoints_dir}"
        )
        return {"history": history, "best_metric": self.best_metric}
