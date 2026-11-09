"""Visualization utilities for bounding box detections on radiographic X-rays."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Union[np.ndarray, List[List[float]]],
    labels: Optional[Union[np.ndarray, List[int], List[str]]] = None,
    scores: Optional[Union[np.ndarray, List[float]]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and optional labels/scores on an image.

    Args:
        image: HxWxC RGB uint8 numpy array.
        boxes: List or array of [x_min, y_min, x_max, y_max].
        labels: Optional class names or integer IDs.
        scores: Optional detection confidence scores [0, 1].
        color: RGB tuple for box outline.
        thickness: Box line thickness in pixels.
        font_scale: Font scale for text annotation.

    Returns:
        Annotated image as RGB uint8 numpy array.
    """
    annotated = image.copy()
    if annotated.dtype != np.uint8:
        if annotated.max() <= 1.0:
            annotated = (annotated * 255).astype(np.uint8)
        else:
            annotated = annotated.astype(np.uint8)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, thickness)

        caption_parts = []
        if labels is not None and i < len(labels):
            caption_parts.append(str(labels[i]))
        if scores is not None and i < len(scores):
            caption_parts.append(f"{scores[i]:.2f}")

        if caption_parts:
            caption = ": ".join(caption_parts)
            (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(
                annotated,
                (x_min, max(0, y_min - h - 4)),
                (x_min + w + 4, max(0, y_min)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                caption,
                (x_min + 2, max(0, y_min - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated


def save_prediction_overlay(
    image: np.ndarray,
    output_path: Union[str, Path],
    boxes: Union[np.ndarray, List[List[float]]],
    labels: Optional[Union[np.ndarray, List[str]]] = None,
    scores: Optional[Union[np.ndarray, List[float]]] = None,
) -> Path:
    """Draw boxes and save annotated RGB image to disk."""
    annotated = draw_bounding_boxes(image, boxes, labels, scores)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Convert RGB -> BGR for OpenCV imwrite
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out), bgr)
    return out
