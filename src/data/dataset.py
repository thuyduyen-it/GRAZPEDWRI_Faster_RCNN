"""PyTorch Dataset implementation for GRAZPEDWRI-DX (Phase 2).

Handles image loading, label translation, dynamic split path resolution,
and negative samples (healthy pediatric wrists without fracture).
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.annotation import parse_yolo_annotation_file
from src.data.transforms import get_transforms


class GRAZPEDWRIDataset(Dataset):
    """Pediatric Wrist X-Ray Bounding Box Dataset."""

    def __init__(
        self,
        split_file: Union[str, Path],
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path],
        transforms: Optional[Callable] = None,
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        yolo_class_offset: int = 1,
        fallback_images_dir: Optional[Union[str, Path]] = None,
        fallback_labels_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize dataset.

        Args:
            split_file: Path to split text file (e.g. train.txt, val.txt, test.txt).
            images_dir: Path to directory containing image files.
            labels_dir: Path to directory containing YOLO annotation text files.
            transforms: Callable transform pipeline.
            is_train: Whether this instance is for training.
            config: Optional config dictionary for transform construction.
            yolo_class_offset: Added to YOLO class ID (0 -> 1 for Faster R-CNN).
            fallback_images_dir: Optional fallback directory for sample images when full dataset is external.
            fallback_labels_dir: Optional fallback directory for sample labels.
        """
        self.split_file = Path(split_file)
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms or get_transforms(is_train=is_train, config=config)
        self.is_train = is_train
        self.yolo_class_offset = yolo_class_offset

        self.fallback_images_dir = Path(fallback_images_dir) if fallback_images_dir else None
        self.fallback_labels_dir = Path(fallback_labels_dir) if fallback_labels_dir else None

        self.samples: List[Tuple[str, Path, Path]] = []
        self._load_split()

    def _load_split(self) -> None:
        """Parse split file and resolve image and label paths dynamically."""
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found at: {self.split_file}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            # Safely resolve image filename by stripping any external/hardcoded directories
            filename = os.path.basename(line)
            stem = os.path.splitext(filename)[0]

            img_path = self.images_dir / filename
            label_path = self.labels_dir / f"{stem}.txt"

            # Check fallback if primary does not exist
            if not img_path.exists() and self.fallback_images_dir and (self.fallback_images_dir / filename).exists():
                img_path = self.fallback_images_dir / filename
                if self.fallback_labels_dir:
                    label_path = self.fallback_labels_dir / f"{stem}.txt"

            self.samples.append((filename, img_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        filename, img_path, label_path = self.samples[idx]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        # Read image using OpenCV (unchanged bit depth: handles native 8-bit or 16-bit)
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to read image at: {img_path}")

        # If OpenCV loaded BGR 3-channel, convert to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        # Parse YOLO annotation
        target = parse_yolo_annotation_file(
            file_path=label_path,
            img_width=width,
            img_height=height,
            yolo_class_offset=self.yolo_class_offset,
        )

        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["orig_size"] = torch.tensor([height, width], dtype=torch.int64)
        target["filename"] = filename

        # Apply transforms synchronously to image and boxes
        if self.transforms is not None:
            image_tensor, target = self.transforms(image, target)
        else:
            # Fallback tensor conversion
            if image.ndim == 2:
                image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
            denom = 65535.0 if image.dtype == np.uint16 else 255.0
            image_tensor = torch.from_numpy(
                (image.astype(np.float32) / denom).transpose((2, 0, 1))
            ).contiguous()

        return image_tensor, target


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]]
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, Any], ...]]:
    """Torchvision Faster R-CNN compatible collate function.

    Handles variable-length bounding boxes per sample, including negative samples
    with 0 bounding boxes.
    """
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)
