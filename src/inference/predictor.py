"""Inference predictor module for pediatric wrist fracture detection (Phase 7).

Provides a high-level, production-ready inference API supporting:
- Input as file path, PIL Image, or NumPy array (8-bit uint8 or 16-bit uint16)
- Automatic bit-depth normalization and channel expansion
- Batch prediction with latency benchmarking
- Clinical visual annotation overlays with diagnostic banners
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

from src.models.faster_rcnn import build_faster_rcnn_resnet50, build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import draw_bounding_boxes


class FracturePredictor:
    """End-to-end inference engine for running fracture detection on pediatric X-rays."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """Initialize the fracture predictor.

        Args:
            checkpoint_path: Path to trained model weights (.pth).
            config: Optional configuration dictionary.
            confidence_threshold: Minimum detection confidence score [0.0, 1.0].
            device: Compute device (CPU or CUDA).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.config = config or {}

        if self.config:
            if not torch.cuda.is_available():
                temp_cfg = self.config.copy()
                temp_cfg["model"] = temp_cfg.get("model", {}).copy()
                temp_cfg["model"]["pretrained"] = False
                self.model = build_model(temp_cfg).to(self.device)
            else:
                self.model = build_model(self.config).to(self.device)
        else:
            self.model = build_faster_rcnn_resnet50(num_classes=2, pretrained=False).to(self.device)

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            load_checkpoint(checkpoint_path, self.model, device=self.device)

        self.model.eval()

    def _preprocess_input(
        self,
        image_input: Any,
    ) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """Preprocess raw input into model tensor and displayable uint8 RGB image."""
        # Check if PIL Image
        if hasattr(image_input, "convert"):
            img_np = np.array(image_input)
        elif isinstance(image_input, (str, Path)):
            img_np = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
            if img_np is None:
                raise IOError(f"Could not read image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img_np = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_shape = (img_np.shape[0], img_np.shape[1])  # (H, W)

        # Normalize bit-depth
        is_16bit = img_np.dtype == np.uint16
        if is_16bit:
            img_float = img_np.astype(np.float32) / 65535.0
            # Scaled uint8 image for visual display
            img_disp = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            img_float = img_np.astype(np.float32) / 255.0
            img_disp = img_np.copy()
        else:
            img_float = img_np.astype(np.float32)
            if img_float.max() > 1.0:
                img_float /= 255.0
            img_disp = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

        # Ensure 3-channel RGB for both tensor and display
        if img_float.ndim == 2:
            img_float = np.repeat(np.expand_dims(img_float, axis=2), 3, axis=2)
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
        elif img_float.ndim == 3 and img_float.shape[2] == 1:
            img_float = np.repeat(img_float, 3, axis=2)
            img_disp = cv2.cvtColor(img_disp[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif img_float.ndim == 3 and img_float.shape[2] == 3:
            # If loaded from OpenCV, convert BGR to RGB
            if isinstance(image_input, (str, Path)):
                img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
                img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(img_float.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
        return tensor, img_disp, orig_shape

    @torch.no_grad()
    def predict(
        self,
        image_input: Any,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict fracture bounding boxes on a single X-ray image.

        Args:
            image_input: Image file path, PIL Image, or NumPy array.
            confidence_threshold: Optional threshold override.

        Returns:
            Dictionary with structured prediction results.
        """
        score_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        tensor, img_disp, (height, width) = self._preprocess_input(image_input)

        start_t = time.perf_counter()
        outputs = self.model(tensor)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        # Filter by threshold and target class (1: Fracture)
        mask = (scores >= score_thresh) & (labels == 1)
        filtered_boxes = [[round(float(coord), 2) for coord in box] for box in boxes[mask]]
        filtered_scores = [round(float(s), 4) for s in scores[mask]]
        filtered_labels = [int(lbl) for lbl in labels[mask]]
        class_names = ["Fracture"] * len(filtered_boxes)

        has_fracture = len(filtered_boxes) > 0

        return {
            "has_fracture": has_fracture,
            "num_fractures": len(filtered_boxes),
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "labels": filtered_labels,
            "class_names": class_names,
            "max_confidence": max(filtered_scores) if filtered_scores else 0.0,
            "image_size": (height, width),
            "latency_ms": round(latency_ms, 2),
            "image_rgb": img_disp,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        image_inputs: List[Any],
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Predict fractures across a list of image inputs."""
        return [self.predict(img, confidence_threshold=confidence_threshold) for img in image_inputs]

    def predict_and_visualize(
        self,
        image_input: Any,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run prediction and render an annotated visual diagnostic image.

        Returns:
            Tuple of (annotated RGB image, results dictionary).
        """
        results = self.predict(image_input, confidence_threshold=confidence_threshold)
        image_rgb = results["image_rgb"].copy()

        # Draw bounding boxes (Red for fracture)
        annotated = draw_bounding_boxes(
            image=image_rgb,
            boxes=results["boxes"],
            labels=results["class_names"],
            scores=results["scores"],
            color=(255, 0, 0),  # Red
            thickness=2,
            font_scale=0.6,
        )

        # Draw clinical header banner
        if results["has_fracture"]:
            banner_text = f"DIAGNOSIS: FRACTURE DETECTED ({results['num_fractures']} site(s), max conf: {results['max_confidence']:.2f})"
            banner_bg = (220, 20, 60)  # Crimson
        else:
            banner_text = "DIAGNOSIS: NO FRACTURE DETECTED (Normal / Negative)"
            banner_bg = (34, 139, 34)  # Forest Green

        # Draw header banner
        banner_h = 36
        banner = np.full((banner_h, annotated.shape[1], 3), banner_bg, dtype=np.uint8)
        cv2.putText(
            banner,
            banner_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        final_vis = np.vstack([banner, annotated])
        return final_vis, results
