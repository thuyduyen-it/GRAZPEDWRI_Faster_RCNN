"""Annotation loading, translation, and validation module.

Handles conversion from YOLO normalized annotations (x_center, y_center, width, height)
to Faster R-CNN absolute pixel coordinates (x_min, y_min, x_max, y_max), class ID mapping
(YOLO 0 -> Faster R-CNN 1), empty annotation files (negative samples), and bounding box checks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch


def yolo_to_xyxy(
    box: Union[List[float], Tuple[float, ...]],
    img_width: int,
    img_height: int,
    clamp: bool = True
) -> List[float]:
    """Convert a single YOLO normalized bounding box to absolute Pascal VOC [x_min, y_min, x_max, y_max].

    Args:
        box: Tuple or list of (x_center, y_center, width, height) normalized in [0, 1].
        img_width: Width of image in pixels.
        img_height: Height of image in pixels.
        clamp: Whether to clamp output coordinates to [0, width] and [0, height].

    Returns:
        List of [x_min, y_min, x_max, y_max] in pixel coordinates.

    Raises:
        ValueError: If box dimensions or image dimensions are invalid.
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image dimensions: width={img_width}, height={img_height}")

    x_c, y_c, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box width/height: w={w}, h={h}")

    x_min = (x_c - w / 2.0) * img_width
    y_min = (y_c - h / 2.0) * img_height
    x_max = (x_c + w / 2.0) * img_width
    y_max = (y_c + h / 2.0) * img_height

    if clamp:
        x_min = max(0.0, min(float(img_width), x_min))
        y_min = max(0.0, min(float(img_height), y_min))
        x_max = max(0.0, min(float(img_width), x_max))
        y_max = max(0.0, min(float(img_height), y_max))

    return [x_min, y_min, x_max, y_max]


def is_valid_box(box: List[float], min_dimension: float = 1.0) -> bool:
    """Validate that bounding box has positive area and width/height >= min_dimension.

    Args:
        box: [x_min, y_min, x_max, y_max]
        min_dimension: Minimum acceptable width and height in pixels.

    Returns:
        True if valid box, False otherwise.
    """
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return width >= min_dimension and height >= min_dimension


def parse_yolo_annotation_file(
    file_path: Union[str, Path],
    img_width: int,
    img_height: int,
    yolo_class_offset: int = 1
) -> Dict[str, torch.Tensor]:
    """Parse a YOLO annotation text file and return Torchvision target dictionary.

    Args:
        file_path: Path to YOLO annotation text file.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        yolo_class_offset: Offset added to YOLO class ID (default: 1, mapping 0 -> 1).

    Returns:
        Dictionary containing:
            - 'boxes': FloatTensor of shape [N, 4] in (x_min, y_min, x_max, y_max)
            - 'labels': Int64Tensor of shape [N]
            - 'area': FloatTensor of shape [N]
            - 'iscrowd': Int64Tensor of shape [N]
    """
    path = Path(file_path)
    boxes: List[List[float]] = []
    labels: List[int] = []
    areas: List[float] = []

    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(
                        f"Malformed YOLO annotation at {path}:{line_idx} -> '{line}'"
                    )

                class_id = int(parts[0])
                # We expect class_id == 0 for fracture; mapping adds offset (e.g. 0 -> 1)
                mapped_label = class_id + yolo_class_offset

                x_c, y_c, w, h = map(float, parts[1:5])
                xyxy = yolo_to_xyxy((x_c, y_c, w, h), img_width, img_height, clamp=True)

                if is_valid_box(xyxy):
                    boxes.append(xyxy)
                    labels.append(mapped_label)
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    areas.append(area)

    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)
    else:
        # Negative sample (no fractures or empty annotation file)
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
        area_tensor = torch.zeros((0,), dtype=torch.float32)
        iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "area": area_tensor,
        "iscrowd": iscrowd_tensor,
    }
