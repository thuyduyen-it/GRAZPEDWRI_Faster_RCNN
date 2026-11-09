"""Evaluation engine for object detection on validation and test sets (Phase 5).

Evaluates trained Faster R-CNN models without weight updates, tracking mAP@0.5,
mAP@0.5:0.95, Precision, Recall, F1, inference latency, and diagnostic false positives
specifically categorized on true negative images.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logging import setup_logger
from src.utils.metrics import DetectionEvaluator, evaluate_threshold_sweep


class Evaluator:
    """Evaluates detection models using standardized COCO metrics and clinical thresholds."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.logger = setup_logger(name="Evaluator")

        iou_thresholds = self.config.get("evaluation", {}).get(
            "iou_thresholds",
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        )
        self.metric_evaluator = DetectionEvaluator(iou_thresholds=iou_thresholds)

    @torch.no_grad()
    def evaluate(
        self,
        confidence_threshold: Optional[float] = None,
        optimize_threshold: bool = False,
        save_diagnostics: bool = False,
        diagnostics_dir: Optional[str] = "outputs/predictions/eval_diagnostics",
        max_diagnostics: int = 10,
    ) -> Dict[str, Any]:
        """Run full evaluation pass over data_loader.

        Args:
            confidence_threshold: Optional confidence threshold to filter detections.
            optimize_threshold: Whether to sweep thresholds to identify optimal operating point.
            save_diagnostics: Whether to export visual overlay comparisons.
            diagnostics_dir: Directory to save diagnostic visual overlays.
            max_diagnostics: Maximum number of visual diagnostic samples to export.

        Returns:
            Dictionary containing mAP, latency, threshold sweep, and operational metrics.
        """
        self.model.eval()
        self.metric_evaluator.reset()

        score_thresh = confidence_threshold or self.config.get(
            "evaluation", {}
        ).get("confidence_threshold", 0.05)

        raw_predictions: List[Dict[str, torch.Tensor]] = []
        raw_targets: List[Dict[str, torch.Tensor]] = []
        raw_images_for_viz: List[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = []

        total_inference_time = 0.0
        total_images = 0

        self.logger.info(f"Starting evaluation pass on {len(self.data_loader.dataset)} images...")

        for images, targets in tqdm(self.data_loader, desc="Evaluating Detection"):
            batch_images = [img.to(self.device) for img in images]

            # Measure inference time
            start_t = time.perf_counter()
            outputs = self.model(batch_images)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_duration = time.perf_counter() - start_t

            total_inference_time += inference_duration
            total_images += len(images)

            # Format predictions and targets
            batch_preds: List[Dict[str, torch.Tensor]] = []
            for out in outputs:
                boxes = out["boxes"].cpu()
                scores = out["scores"].cpu()
                labels = out["labels"].cpu()

                mask = scores >= score_thresh
                pred_dict = {
                    "boxes": boxes[mask],
                    "scores": scores[mask],
                    "labels": labels[mask],
                }
                batch_preds.append(pred_dict)
                raw_predictions.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })

            batch_targets: List[Dict[str, torch.Tensor]] = [
                {
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
                for t in targets
            ]
            raw_targets.extend(batch_targets)

            self.metric_evaluator.update(batch_preds, batch_targets)

            # Cache diagnostic samples if requested
            if save_diagnostics and len(raw_images_for_viz) < max_diagnostics:
                for img_tensor, p_dict, t_dict in zip(images, batch_preds, batch_targets):
                    if len(raw_images_for_viz) >= max_diagnostics:
                        break
                    # Convert tensor back to uint8 image for visualization
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                    raw_images_for_viz.append((img_np, p_dict, t_dict))

        # Compute standard COCO detection metrics
        coco_metrics = self.metric_evaluator.compute()
        avg_latency_ms = (total_inference_time / max(1, total_images)) * 1000.0
        fps = total_images / max(1e-5, total_inference_time)

        results: Dict[str, Any] = {
            "map_50": round(coco_metrics.get("map_50", 0.0), 4),
            "map_50_95": round(coco_metrics.get("map_50_95", 0.0), 4),
            "total_images": total_images,
            "latency": {
                "avg_per_image_ms": round(avg_latency_ms, 2),
                "fps": round(fps, 2),
                "total_time_seconds": round(total_inference_time, 2),
            },
        }

        # Threshold optimization / sweep
        sweep_data = evaluate_threshold_sweep(
            raw_predictions,
            raw_targets,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            iou_threshold=0.5,
            target_class=1,
        )
        results["sweep_analysis"] = sweep_data["sweep_table"]
        results["optimal_threshold"] = sweep_data["optimal_threshold"]
        results["optimal_metrics"] = sweep_data["optimal_metrics"]

        # Default operating point metrics (e.g. at threshold 0.5)
        default_op = [row for row in sweep_data["sweep_table"] if row["threshold"] == 0.5]
        if default_op:
            results["metrics_at_0_5"] = default_op[0]

        self.logger.info(
            f"Evaluation Summary - mAP@0.5: {results['map_50']:.4f} | "
            f"mAP@0.5:0.95: {results['map_50_95']:.4f} | "
            f"Latency: {avg_latency_ms:.2f} ms/img ({fps:.1f} FPS)"
        )
        self.logger.info(
            f"Optimal Operating Point: Threshold={results['optimal_threshold']} "
            f"(F1={results['optimal_metrics']['f1']:.4f}, "
            f"Precision={results['optimal_metrics']['precision']:.4f}, "
            f"Recall={results['optimal_metrics']['recall']:.4f}, "
            f"FP on Negatives={results['optimal_metrics']['fp_negative_images']})"
        )

        # Export diagnostic overlays
        if save_diagnostics and diagnostics_dir:
            self._save_diagnostic_overlays(
                raw_images_for_viz,
                Path(diagnostics_dir),
                optimal_threshold=results["optimal_threshold"],
            )

        return results

    def _save_diagnostic_overlays(
        self,
        samples: List[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]],
        out_dir: Path,
        optimal_threshold: float,
    ) -> None:
        """Export visual diagnostic images overlaying Ground Truth vs Predictions."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Exporting {len(samples)} diagnostic visual overlays to {out_dir}...")

        for idx, (img_rgb, p_dict, t_dict) in enumerate(samples):
            # Convert RGB to BGR for OpenCV
            vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Draw Ground Truth boxes in GREEN
            gt_boxes = t_dict.get("boxes", torch.tensor([]))
            for box in gt_boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis_img,
                    "GT Fracture",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Draw Predicted boxes in RED (filtered by optimal threshold)
            pred_boxes = p_dict.get("boxes", torch.tensor([]))
            pred_scores = p_dict.get("scores", torch.tensor([]))

            for box, score in zip(pred_boxes, pred_scores):
                if score < optimal_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                score_val = float(score.item())
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    vis_img,
                    f"Pred: {score_val:.2f}",
                    (x1, min(vis_img.shape[0] - 5, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Add sample category label
            category = "Positive Sample" if len(gt_boxes) > 0 else "True Negative Sample"
            cv2.putText(
                vis_img,
                f"Sample {idx+1}: {category}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_file = out_dir / f"eval_sample_{idx+1:02d}.png"
            cv2.imwrite(str(out_file), vis_img)
