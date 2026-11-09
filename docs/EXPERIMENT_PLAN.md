# Faster R-CNN Experimental Strategy & Benchmark Plan

This document outlines the rigorous, controlled experimental suite designed to optimize Faster R-CNN for pediatric wrist fracture detection on the **GRAZPEDWRI-DX** dataset.

---

## 1. Experimental Integrity & Methodology

To ensure scientific validity and clinical credibility:

1. **Patient-Level Split Invariance**: Every experiment operates on the exact same patient-level train (14,210 images), validation (3,032 images), and test (3,085 images) sets. No cross-split leakage is permitted.
2. **Strict Test Set Isolation**: The test set is locked and unexamined during all exploratory runs and hyperparameter tuning. Operating confidence thresholds are calibrated exclusively on the validation set. The test set is evaluated once per finalized configuration.
3. **Fixed Random Seeds**: All stochastic processes (weight initialization, data shuffling, augmentation sampling) use fixed random seed `42`.
4. **Consistent Evaluation Protocol**: Evaluation uses TorchMetrics detection metrics ($\text{mAP@0.5}$, $\text{mAP@0.5:0.95}$, Precision, Recall, and F1-score) across both positive and true negative images.

---

## 2. Planned Experiments

### EXP-01: Baseline Faster R-CNN (ResNet-50-FPN)
- **Objective**: Establish benchmark detection performance using standard, widely validated object detection settings.
- **Hypothesis**: Pre-trained ResNet-50-FPN provides a strong feature representation across multi-scale features via FPN pyramids.
- **Specification**:
  - Backbone: ResNet-50 with Feature Pyramid Network (FPN)
  - Pretrained: COCO weights (`DEFAULT`)
  - Trainable backbone layers: 3 (stages 2, 3, 4 fine-tuned; stage 1 frozen)
  - Optimizer: SGD (lr: 0.005, momentum: 0.9, weight decay: 0.0005)
  - LR Scheduler: StepLR (decay $\times 0.1$ at epochs 6 and 12)
  - Image size: Min 800, Max 1333
  - Batch size: 2 per GPU
  - Epochs: 20
  - Augmentation: Horizontal flip ($p=0.5$) only

---

### EXP-02: Medical Image Domain Augmentation
- **Objective**: Address subtle visual characteristics of pediatric fractures (e.g. greenstick, buckle/torus fractures) without introducing anatomically impossible distortions.
- **Hypothesis**: Contrast-limited adaptive histogram equalization (CLAHE) enhances trabecular bone pattern visibility, and slight rotational jitter accounts for minor patient positioning variations during X-ray acquisition.
- **Ablation Variants**:
  - **Variant A**: Baseline (Horizontal flip only)
  - **Variant B**: + Slight random rotation ($\pm 10^\circ$)
  - **Variant C**: + CLAHE (clip limit 2.0, grid size $8 \times 8$)
  - **Variant D**: Combined (Flip + Rotation + CLAHE + Slight Contrast Jitter $\pm 10\%$)
- **Safety Restriction**: No elastic deformation, aggressive shearing, or arbitrary color hue shifts that corrupt clinical bone densities.

---

### EXP-03: Backbone Architecture Comparison
- **Objective**: Quantify the trade-off between architectural depth, spatial resolution, model capacity, and inference latency.
- **Hypothesis**: Deeper backbones (ResNet-101) capture subtler fracture lines but may risk overfitting on pediatric datasets, while lightweight backbones (MobileNetV3) offer deployment speed.
- **Ablation Variants**:
  - **ResNet-50-FPN** (Baseline, ~41.5M parameters)
  - **ResNet-101-FPN** (Deeper representation, ~60.5M parameters)
  - **MobileNetV3-Large-FPN** (Mobile/Edge deployment candidate, ~19M parameters)

---

### EXP-04: Multi-Scale Image Resolution Analysis
- **Objective**: Determine optimal input resolution balance between GPU VRAM limits (15 GB Tesla T4) and fine anatomical detail retention.
- **Hypothesis**: Hairline and buckle fractures in pediatric bones occupy small bounding box areas ($< 32 \times 32$ pixels at low resolution). Higher minimum resolution preserves fine fracture lines.
- **Ablation Variants**:
  - **Low Resolution**: Min 600, Max 1000
  - **Standard Resolution (Baseline)**: Min 800, Max 1333
  - **High Resolution**: Min 1024, Max 1600 (batch size scaled to 1 with gradient accumulation)

---

### EXP-05: Hyperparameter & Optimizer Optimization
- **Objective**: Identify the most stable convergence trajectory and regularizer for Faster R-CNN on medical X-rays.
- **Hypothesis**: Adaptive optimizers with decoupled weight decay (AdamW) with cosine annealing can converge smoother than standard SGD on noisy medical gradients.
- **Ablation Variants**:
  - **SGD Baseline**: lr = 0.005, StepLR, weight decay = $5 \times 10^{-4}$
  - **SGD Conservative**: lr = 0.001, StepLR, weight decay = $5 \times 10^{-4}$
  - **AdamW Cosine**: lr = $1 \times 10^{-4}$, CosineAnnealingLR, weight decay = $1 \times 10^{-2}$

---

## 3. Results Tracking Matrix (Template)

All experiments record quantitative results in `outputs/metrics/experiments_summary.csv` and are documented using the following schema:

| Exp ID | Architecture | Image Res | Augmentation | Optimizer | Param (M) | Val mAP@0.5 | Val mAP@0.5:0.95 | Val F1 | Test mAP@0.5 | Test mAP@0.5:0.95 | Test F1 | Inference Latency (ms) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **EXP-01** | ResNet50-FPN | 800x1333 | Flip | SGD (0.005) | 41.5 | *Pending* | *Pending* | *Pending* | *Locked* | *Locked* | *Locked* | *Pending* |
| **EXP-02** | ResNet50-FPN | 800x1333 | Medical CLAHE | SGD (0.005) | 41.5 | *Pending* | *Pending* | *Pending* | *Locked* | *Locked* | *Locked* | *Pending* |
| **EXP-03** | ResNet101-FPN | 800x1333 | Flip | SGD (0.005) | 60.5 | *Pending* | *Pending* | *Pending* | *Locked* | *Locked* | *Locked* | *Pending* |
| **EXP-04** | ResNet50-FPN | 1024x1600 | Flip | SGD (0.005) | 41.5 | *Pending* | *Pending* | *Pending* | *Locked* | *Locked* | *Locked* | *Pending* |
| **EXP-05** | ResNet50-FPN | 800x1333 | Flip | AdamW (1e-4) | 41.5 | *Pending* | *Pending* | *Pending* | *Locked* | *Locked* | *Locked* | *Pending* |

> *Note*: Test metrics remain strictly locked until the completion of validation-based model selection.
