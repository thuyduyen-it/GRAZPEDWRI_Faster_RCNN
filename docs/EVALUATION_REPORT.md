# GRAZPEDWRI-DX Model Evaluation & Benchmark Report (Phase 5)

## 1. Executive Summary & Purpose

Phase 5 establishes a standardized, reproducible, and clinically aligned evaluation framework for Faster R-CNN on pediatric wrist X-ray images.

Evaluating fracture detection in pediatric radiographs imposes distinct clinical challenges compared to general object detection benchmarks:
1. **Severe Cost of False Negatives (FN)**: A missed fracture (e.g. subtle greenstick, torus/buckle fracture, or hairline Salter-Harris physeal fracture) can cause improper bone union, chronic deformity, or premature growth arrest. High **Sensitivity / Recall** is clinically prioritized.
2. **Impact of False Positives (FP) on Negative Samples**: GRAZPEDWRI-DX contains 6,777 non-fracture radiographs. Bounding box false positives on normal growth plates (epiphyseal lines) lead to unnecessary immobilization, follow-up radiographs, and radiation burden. The evaluation engine explicitly measures **False Positives on True Negative radiographs**.
3. **Strict Validation vs Test Partitioning**: Confidence threshold calibration is conducted **strictly on the Validation split** ($n=3,032$). The calibrated threshold is subsequently applied to the held-out **Test split** ($n=3,085$) to prevent data leakage and benchmark over-fitting.

---

## 2. Evaluation Metrics & Mathematical Formulation

### 2.1 Intersection-over-Union (IoU)
For predicted bounding box $B_p$ and ground truth box $B_{gt}$:
$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$

### 2.2 COCO Detection Metrics
- **mAP@0.50**: Mean Average Precision computed at standard IoU threshold $\theta = 0.50$.
- **mAP@[0.50:0.95]**: COCO primary benchmark metric, averaging AP over 10 IoU thresholds $\theta \in [0.50, 0.55, \dots, 0.95]$ with step $0.05$:
  $$\text{mAP@[0.50:0.95]} = \frac{1}{10} \sum_{\theta=0.50}^{0.95} \text{AP}_{\theta}$$

### 2.3 Clinical Diagnostic Metrics
Given True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$):
- **Precision (Positive Predictive Value)**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall (Sensitivity)**: $\text{Recall} = \frac{TP}{TP + FN}$
- **F1-Score (Harmonic Mean)**: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **FP on Negative Images**: Count of non-existent bounding boxes predicted on healthy radiographs.

---

## 3. Confidence Threshold Optimization Strategy

To find the optimal clinical balance between sensitivity and specificity, the evaluation engine performs a systematic sweep of confidence thresholds $\tau \in [0.1, 0.9]$:

1. Detections with confidence score $s < \tau$ are filtered out.
2. Remaining predictions are matched with ground truth annotations using greedy IoU assignment ($\text{IoU} \ge 0.50$).
3. The optimal operating threshold $\tau^*$ is chosen to maximize $F_1$-score on the **Validation set**:
   $$\tau^* = \arg\max_{\tau \in [0.1, 0.9]} F_1(\tau)$$

---

## 4. Evaluation CLI Usage

### 4.1 Calibrate Optimal Threshold on Validation Set
```bash
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/best_model.pth --split val --optimize-threshold --save-diagnostics
```

### 4.2 Benchmark on Held-out Test Set
```bash
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/best_model.pth --split test --confidence-threshold <calibrated_threshold> --save-diagnostics
```

---

## 5. Artifacts & Outputs

- **Evaluation Metrics JSON**: Saved to `outputs/metrics/eval_{split}_metrics.json`.
- **Diagnostic Visual Overlays**: Exported to `outputs/predictions/eval_diagnostics/`:
  - **Green Bounding Boxes**: Ground Truth fracture annotations.
  - **Red Bounding Boxes**: Faster R-CNN predicted bounding boxes with confidence scores.
  - **Header Label**: Clear classification indicating `Positive Sample` vs `True Negative Sample`.
