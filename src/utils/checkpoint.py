"""Model checkpoint persistence, saving, and restoration module."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> Path:
    """Save training state checkpoint to disk.

    Args:
        state: State dictionary containing model, optimizer, scheduler, epoch, and metrics.
        checkpoint_dir: Directory where checkpoints should be stored.
        filename: Checkpoint filename.
        is_best: If True, also copies checkpoint as 'best_model.pth'.

    Returns:
        Path to the saved checkpoint file.
    """
    save_dir = Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)

    return filepath


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model weights and optional optimizer/scheduler states.

    Args:
        filepath: Path to saved checkpoint file.
        model: PyTorch model into which weights will be loaded.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional learning rate scheduler to restore state.
        device: Target torch device.

    Returns:
        Dictionary containing checkpoint metadata (epoch, best_metric, etc.).
    """
    checkpoint_path = Path(filepath)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device or torch.device("cpu"))

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
