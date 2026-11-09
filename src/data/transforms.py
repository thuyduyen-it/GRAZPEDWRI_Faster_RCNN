"""Image transforms and data augmentation module for pediatric X-rays (Phase 2).

Provides conservative, clinically safe augmentation and normalization
for training and validation pipelines while maintaining bounding box consistency.

NOTE: Torchvision Faster R-CNN internally normalizes inputs using ImageNet
mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
Therefore, the input tensors to the model must be in range [0.0, 1.0] with shape [3, H, W].
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch


class MedicalCompose:
    """Composite transform ensuring synchronous image and bounding box transformations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class MedicalCLAHE:
    """Contrast-Limited Adaptive Histogram Equalization for bone trabecular enhancement.

    Operates natively on 16-bit uint16 or 8-bit uint8 grayscale images.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), prob: float = 0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.prob = prob
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        if np.random.rand() < self.prob:
            if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                # Grayscale
                base = image if image.ndim == 2 else image[:, :, 0]
                enhanced = self.clahe.apply(base)
                if image.ndim == 3:
                    image = np.expand_dims(enhanced, axis=2)
                else:
                    image = enhanced
            elif image.ndim == 3 and image.shape[2] == 3:
                # 3-channel: if uint8, convert to LAB and apply to L channel
                if image.dtype == np.uint8:
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    # For uint16 3-channel, apply to each channel
                    chans = [self.clahe.apply(image[:, :, i]) for i in range(3)]
                    image = np.stack(chans, axis=2)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip with exact bounding box coordinate transformation."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        if np.random.rand() < self.prob:
            image = np.fliplr(image).copy()
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                width = image.shape[1]
                boxes = target["boxes"].clone()
                # Flip x_min and x_max:
                # new_xmin = width - old_xmax
                # new_xmax = width - old_xmin
                x_min = width - boxes[:, 2]
                x_max = width - boxes[:, 0]
                boxes[:, 0] = x_min
                boxes[:, 2] = x_max

                # Clamp to [0, width]
                boxes[:, 0] = torch.clamp(boxes[:, 0], min=0.0, max=float(width))
                boxes[:, 2] = torch.clamp(boxes[:, 2], min=0.0, max=float(width))

                target["boxes"] = boxes
                if "area" in target:
                    target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return image, target


class RandomBrightnessContrast:
    """Conservative brightness and contrast adjustment for radiographic images."""

    def __init__(self, brightness_limit: float = 0.1, contrast_limit: float = 0.1, prob: float = 0.3):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        if np.random.rand() < self.prob:
            alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = np.random.uniform(-self.brightness_limit, self.brightness_limit)

            max_val = 65535.0 if image.dtype == np.uint16 else 255.0
            img_float = image.astype(np.float32)
            adjusted = img_float * alpha + beta * max_val
            adjusted = np.clip(adjusted, 0.0, max_val).astype(image.dtype)
            image = adjusted

        return image, target


class ToTensor:
    """Convert numpy array image (H, W) or (H, W, C) to float32 Tensor [3, H, W] in [0.0, 1.0].

    Correctly normalizes 16-bit uint16 images (dividing by 65535.0) and
    8-bit uint8 images (dividing by 255.0) without losing dynamic range.
    Expands 1-channel grayscale to 3 channels for pre-trained backbone compatibility.
    """

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)

        # Handle bit-depth normalization safely
        if image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)
            if img_float.max() > 1.0:
                img_float = img_float / 255.0

        # Expand 1-channel grayscale to 3-channel RGB for ResNet backbones
        if img_float.shape[2] == 1:
            img_float = np.repeat(img_float, 3, axis=2)

        # Convert (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_float.transpose((2, 0, 1))).contiguous()
        return tensor, target


def get_transforms(is_train: bool = True, config: Optional[Dict[str, Any]] = None) -> MedicalCompose:
    """Build transform pipeline based on configuration.

    Args:
        is_train: If True, applies medical-safe data augmentation; otherwise validation normalization only.
        config: Optional configuration dictionary.

    Returns:
        MedicalCompose transform object.
    """
    transforms = []
    if is_train:
        t_cfg = config.get("transforms", {}) if config else {}
        aug_cfg = t_cfg.get("augmentation", {})

        # Flip probability
        flip_prob = aug_cfg.get("horizontal_flip_prob", t_cfg.get("horizontal_flip", 0.5))
        if flip_prob > 0:
            transforms.append(RandomHorizontalFlip(prob=flip_prob))

        # Medical CLAHE
        clahe_cfg = t_cfg.get("clahe", {})
        clahe_enabled = aug_cfg.get("clahe_enabled", False) or clahe_cfg.get("enabled", False)
        if clahe_enabled or t_cfg.get("augmentation_strategy") == "medical_conservative":
            clip_limit = clahe_cfg.get("clip_limit", 2.0)
            transforms.append(MedicalCLAHE(clip_limit=clip_limit, prob=0.5))

        # Brightness & Contrast
        bc_cfg = t_cfg.get("random_brightness_contrast", {})
        bc_enabled = aug_cfg.get("brightness_contrast_enabled", False) or bool(bc_cfg)
        if bc_enabled:
            b_lim = bc_cfg.get("brightness_limit", 0.1)
            c_lim = bc_cfg.get("contrast_limit", 0.1)
            transforms.append(RandomBrightnessContrast(brightness_limit=b_lim, contrast_limit=c_lim, prob=0.3))

    # Standard tensor conversion (handles 8-bit, 16-bit, and 1 -> 3 channel expansion into [0, 1])
    transforms.append(ToTensor())
    return MedicalCompose(transforms)
