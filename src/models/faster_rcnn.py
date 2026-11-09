"""Faster R-CNN model definition and factory builders using Torchvision (Phase 3).

Configures ResNet-50-FPN, ResNet-101-FPN, and MobileNetV3-Large-FPN backbones with
custom classification and bounding box regression heads for pediatric wrist fracture
detection (2 classes: Background=0 and Fracture=1).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    rpn_anchor_generator: Optional[AnchorGenerator] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """Build Faster R-CNN model with ResNet-50-FPN backbone.

    Args:
        num_classes: Total classes including background (e.g. 2: 0=background, 1=fracture).
        pretrained: If True, load pre-trained COCO weights.
        trainable_backbone_layers: Number of trainable stages from top (0-5).
        box_score_thresh: Minimum score threshold during evaluation.
        box_nms_thresh: Non-maximum suppression IoU threshold.
        box_detections_per_img: Maximum detections retained per image.
        rpn_anchor_generator: Optional custom AnchorGenerator for RPN.

    Returns:
        Configured FasterRCNN PyTorch model instance.
    """
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_backbone = None
    else:
        weights = None
        weights_backbone = None

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        rpn_anchor_generator=rpn_anchor_generator,
        **kwargs,
    )

    # Replace FastRCNNPredictor head with 2 classes (Background + Fracture)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def build_faster_rcnn_resnet101(
    num_classes: int = 2,
    pretrained: bool = False,
    trainable_backbone_layers: int = 3,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    rpn_anchor_generator: Optional[AnchorGenerator] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """Build Faster R-CNN model with ResNet-101-FPN backbone (EXP-03)."""
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=None,
        trainable_layers=trainable_backbone_layers,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        rpn_anchor_generator=rpn_anchor_generator,
        **kwargs,
    )
    return model


def build_faster_rcnn_mobilenet(
    num_classes: int = 2,
    pretrained: bool = False,
    trainable_backbone_layers: int = 3,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
    **kwargs: Any,
) -> FasterRCNN:
    """Build Faster R-CNN with MobileNetV3-Large-FPN backbone (EXP-03 lightweight)."""
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        **kwargs,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter counts, trainable vs frozen parameters, and model size.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with parameter statistics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Size in Megabytes (assuming float32: 4 bytes per parameter)
    param_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)

    # Classification and bbox regression head details
    num_classes = None
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "box_predictor"):
        num_classes = model.roi_heads.box_predictor.cls_score.out_features

    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(frozen_params),
        "trainable_ratio": round(float(trainable_params) / max(1, total_params), 4),
        "model_size_mb": round(float(param_size_mb), 2),
        "num_classes": num_classes,
    }


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to build detection model from configuration dictionary.

    Args:
        config: Loaded YAML configuration dictionary.

    Returns:
        Instantiated PyTorch detection model.
    """
    model_cfg = config.get("model", {})
    num_classes = config.get("data", {}).get("classes", {}).get("num_classes", 2)
    architecture = model_cfg.get("architecture", "fasterrcnn_resnet50_fpn")
    pretrained = model_cfg.get("pretrained", True)
    trainable_layers = model_cfg.get("trainable_backbone_layers", 3)

    score_thresh = model_cfg.get("box_score_thresh", 0.05)
    nms_thresh = model_cfg.get("box_nms_thresh", 0.5)
    detections_per_img = model_cfg.get("box_detections_per_img", 100)

    # Optional custom anchor generator
    anchor_sizes = model_cfg.get("anchor_sizes", None)
    aspect_ratios = model_cfg.get("aspect_ratios", None)
    anchor_gen = None
    if anchor_sizes and aspect_ratios:
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    min_size = model_cfg.get("min_size", 800)
    max_size = model_cfg.get("max_size", 1333)

    if architecture == "fasterrcnn_resnet50_fpn":
        return build_faster_rcnn_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=detections_per_img,
            rpn_anchor_generator=anchor_gen,
            min_size=min_size,
            max_size=max_size,
        )
    elif architecture in ["fasterrcnn_resnet101_fpn", "resnet101_fpn"]:
        return build_faster_rcnn_resnet101(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=detections_per_img,
            rpn_anchor_generator=anchor_gen,
            min_size=min_size,
            max_size=max_size,
        )
    elif architecture in ["fasterrcnn_mobilenet_v3_large_fpn", "mobilenet_v3_large_fpn"]:
        return build_faster_rcnn_mobilenet(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=detections_per_img,
            min_size=min_size,
            max_size=max_size,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
