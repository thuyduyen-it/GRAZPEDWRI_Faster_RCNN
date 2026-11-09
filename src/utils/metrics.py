"""Object detection evaluation metrics module for pediatric wrist fractures (Phase 5).

Provides native and TorchMetrics-compatible Mean Average Precision (mAP@0.5, mAP@0.5:0.95),
per-threshold Precision, Recall, and F1 calculations, and confidence threshold optimization
specifically designed for clinical pediatric radiology evaluation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torchvision.ops import box_iou


class DetectionEvaluator:
    """Detection metric calculator supporting COCO mAP and native fallback."""

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        """Initialize evaluator.

        Args:
            iou_thresholds: List of IoU thresholds for mAP computation.
        """
        self.iou_thresholds = iou_thresholds or [
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        ]
        self._predictions: List[Dict[str, torch.Tensor]] = []
        self._targets: List[Dict[str, torch.Tensor]] = []

        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            self.map_metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                iou_thresholds=self.iou_thresholds,
            )
            self._has_torchmetrics = True
        except ImportError:
            self._has_torchmetrics = False
            self.map_metric = None

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Update metrics state with predictions and ground truths for a batch."""
        for p, t in zip(preds, targets):
            self._predictions.append({
                "boxes": p["boxes"].detach().cpu(),
                "scores": p["scores"].detach().cpu(),
                "labels": p["labels"].detach().cpu(),
            })
            self._targets.append({
                "boxes": t["boxes"].detach().cpu(),
                "labels": t["labels"].detach().cpu(),
            })

        if self._has_torchmetrics and self.map_metric is not None:
            self.map_metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Compute mAP@0.5 and mAP@0.5:0.95."""
        if self._has_torchmetrics and self.map_metric is not None:
            metrics = self.map_metric.compute()
            map_50 = float(metrics.get("map_50", 0.0).cpu().item())
            map_50_95 = float(metrics.get("map", 0.0).cpu().item())
            return {
                "map_50": map_50,
                "map_50_95": map_50_95,
            }

        # Standalone native implementation if torchmetrics is unavailable
        return self._compute_native_coco_map()

    def _compute_native_coco_map(self) -> Dict[str, float]:
        """Compute mAP using native box IoU matching across IoU thresholds."""
        if not self._predictions or not self._targets:
            return {"map_50": 0.0, "map_50_95": 0.0}

        ap_per_iou = []
        for iou_thresh in self.iou_thresholds:
            ap = compute_average_precision(
                self._predictions,
                self._targets,
                target_class=1,
                iou_threshold=iou_thresh,
            )
            ap_per_iou.append(ap)

        map_50 = ap_per_iou[0] if len(ap_per_iou) > 0 else 0.0
        map_50_95 = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0

        return {
            "map_50": float(map_50),
            "map_50_95": float(map_50_95),
        }

    def reset(self) -> None:
        """Reset metric state."""
        self._predictions.clear()
        self._targets.clear()
        if self._has_torchmetrics and self.map_metric is not None:
            self.map_metric.reset()


def compute_average_precision(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    target_class: int = 1,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision (AP) for a specific class using 11-point/all-points VOC interpolation.

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'.
        targets: List of dicts with 'boxes', 'labels'.
        target_class: Target label ID (1 for fracture).
        iou_threshold: Minimum IoU overlap for True Positive.

    Returns:
        Average Precision value between 0.0 and 1.0.
    """
    total_gt = sum(
        int((t["labels"] == target_class).sum().item())
        for t in targets
    )
    if total_gt == 0:
        return 0.0

    all_scores: List[float] = []
    all_matches: List[int] = []

    for pred, target in zip(predictions, targets):
        pred_mask = pred["labels"] == target_class
        p_boxes = pred["boxes"][pred_mask]
        p_scores = pred["scores"][pred_mask]

        gt_mask = target["labels"] == target_class
        gt_boxes = target["boxes"][gt_mask]

        num_preds = len(p_boxes)
        if num_preds == 0:
            continue

        for s in p_scores:
            all_scores.append(float(s.item()))

        if len(gt_boxes) == 0:
            # All predictions on negative sample are false positives
            all_matches.extend([0] * num_preds)
            continue

        ious = box_iou(p_boxes, gt_boxes)  # Shape: (num_preds, num_gts)
        gt_matched = [False] * len(gt_boxes)

        # Sort predictions in this image by score descending for matching
        sort_idx = torch.argsort(p_scores, descending=True)
        img_matches = [0] * num_preds

        for idx in sort_idx:
            best_iou, best_gt = torch.max(ious[idx], dim=0)
            best_iou_val = float(best_iou.item())
            best_gt_idx = int(best_gt.item())

            if best_iou_val >= iou_threshold and not gt_matched[best_gt_idx]:
                img_matches[idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                img_matches[idx] = 0

        all_matches.extend(img_matches)

    if not all_scores:
        return 0.0

    # Sort all detections globally by score descending
    sorted_indices = np.argsort(-np.array(all_scores))
    sorted_matches = np.array(all_matches)[sorted_indices]

    tp_cumsum = np.cumsum(sorted_matches)
    fp_cumsum = np.cumsum(1 - sorted_matches)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # 11-point interpolation or VOC 2010+ continuous integral
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int, eps: float = 1e-7
) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1-score from counts.

    Args:
        tp: True positives count.
        fp: False positives count.
        fn: False negatives count.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary with precision, recall, and f1_score.
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_threshold_sweep(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.5,
    target_class: int = 1,
) -> Dict[str, Any]:
    """Evaluate performance across a range of confidence thresholds to find optimal operating point.

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'.
        targets: List of dicts with 'boxes', 'labels'.
        thresholds: List of confidence thresholds to evaluate (default 0.1 to 0.9).
        iou_threshold: Minimum IoU overlap for True Positive.
        target_class: Target label ID (1 for fracture).

    Returns:
        Dictionary containing sweep table, optimal threshold, and best metrics.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sweep_results = []
    best_f1 = -1.0
    optimal_thresh = 0.5
    optimal_metrics: Dict[str, float] = {}

    for score_thresh in thresholds:
        tp = 0
        fp = 0
        fn = 0
        fp_on_negative_images = 0

        for pred, target in zip(predictions, targets):
            p_mask = (pred["labels"] == target_class) & (pred["scores"] >= score_thresh)
            p_boxes = pred["boxes"][p_mask]

            t_mask = target["labels"] == target_class
            t_boxes = target["boxes"][t_mask]

            is_negative_image = len(t_boxes) == 0

            if len(p_boxes) == 0:
                fn += len(t_boxes)
                continue

            if is_negative_image:
                fp += len(p_boxes)
                fp_on_negative_images += len(p_boxes)
                continue

            # Both predictions and ground truths exist
            ious = box_iou(p_boxes, t_boxes)
            matched_gt = set()

            for p_idx in range(len(p_boxes)):
                best_iou, best_gt = torch.max(ious[p_idx], dim=0)
                best_iou_val = float(best_iou.item())
                best_gt_idx = int(best_gt.item())

                if best_iou_val >= iou_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += (len(t_boxes) - len(matched_gt))

        metrics = compute_precision_recall_f1(tp, fp, fn)
        entry = {
            "threshold": round(score_thresh, 2),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "fp_negative_images": int(fp_on_negative_images),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
        }
        sweep_results.append(entry)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            optimal_thresh = score_thresh
            optimal_metrics = entry

    return {
        "sweep_table": sweep_results,
        "optimal_threshold": optimal_thresh,
        "optimal_metrics": optimal_metrics,
    }
