# Faster R-CNN Project Software Architecture

This document specifies the software architecture, modular layering, data flow, and design patterns implemented in **GRAZPEDWRI_Faster_RCNN**.

---

## 1. Architectural System Overview

The project is structured into distinct, decoupled functional layers designed for reproducibility, modular testing, and seamless migration between local development (VS Code) and GPU cloud compute (Google Colab).

```
                      ┌─────────────────────────────────────────┐
                      │             Configuration               │
                      │    (configs/baseline.yaml & env vars)   │
                      └────────────────────┬────────────────────┘
                                           │
  ┌────────────────────────────────────────┴────────────────────────────────────────┐
  ▼                                        ▼                                        ▼
┌───────────────────┐            ┌───────────────────┐            ┌───────────────────┐
│    Data Layer     │            │    Model Layer    │            │   Engine Layer    │
│                   │            │                   │            │                   │
│ • annotation.py   │            │ • faster_rcnn.py  │            │ • trainer.py      │
│ • dataset.py      │──(batches)►│   - ResNet50-FPN  │◄───────────│ • evaluator.py    │
│ • transforms.py   │            │   - 2 Classes     │            │                   │
└───────────────────┘            └─────────┬─────────┘            └─────────┬─────────┘
                                           │                                │
                                           ▼                                ▼
                                 ┌───────────────────┐            ┌───────────────────┐
                                 │  Inference Layer  │            │    Utils Layer    │
                                 │                   │            │                   │
                                 │ • predictor.py    │            │ • metrics.py      │
                                 │   - Checkpoint    │            │ • visualization.py│
                                 │   - Thresholding  │            │ • checkpoint.py   │
                                 │   - Box Drawing   │            │ • logging.py      │
                                 └───────────────────┘            └───────────────────┘
```

---

## 2. Layer Specifications

### 2.1 Data Layer (`src/data/`)

Responsible for raw file reading, format translation, bounding box normalization, data augmentation, and PyTorch dataset batch construction.

- **`annotation.py`**:
  - **Coordinate Translation**: Converts YOLO normalized center-based coordinates $(x_c, y_c, w, h) \in [0, 1]$ into pixel-space Pascal-VOC bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$:
    $$x_{\min} = \max(0, (x_c - w/2) \cdot W)$$
    $$y_{\min} = \max(0, (y_c - h/2) \cdot H)$$
    $$x_{\max} = \min(W, (x_c + w/2) \cdot W)$$
    $$y_{\max} = \min(H, (y_c + h/2) \cdot H)$$
  - **Class Mapping**: Translates YOLO class `0` (Fracture) $\rightarrow$ Torchvision label `1`. Label `0` is strictly reserved by Faster R-CNN for the background.
  - **Negative Samples**: Empty annotation files are represented by empty tensors:
    - `boxes`: `torch.zeros((0, 4), dtype=torch.float32)`
    - `labels`: `torch.zeros((0,), dtype=torch.int64)`
    - `area`: `torch.zeros((0,), dtype=torch.float32)`
    - `iscrowd`: `torch.zeros((0,), dtype=torch.int64)`
  - **Validation & Clamping**: Ensures $x_{\min} < x_{\max}$ and $y_{\min} < y_{\max}$ with a minimal epsilon width/height check to reject degeneracies.

- **`dataset.py`**:
  - Subclasses `torch.utils.data.Dataset`.
  - Dynamically resolves image filenames from split files (`train.txt`, `val.txt`, `test.txt`) against the configured images directory (`GRAZPEDWRI_DATA_ROOT` or YAML path).
  - Returns `image: torch.FloatTensor [C, H, W]` in range $[0.0, 1.0]$ and a dictionary `target` complying with Torchvision requirements.
  - Custom `collate_fn(batch)` packages variable-length bounding boxes into tuples.

- **`transforms.py`**:
  - Implements Albumentations / Torchvision transforms.
  - Ensures bounding boxes are transformed identically to the underlying image pixels.
  - Implements domain-safe intensity transforms (e.g. CLAHE, gentle contrast shifts) suitable for pediatric radiograms.

---

### 2.2 Model Layer (`src/models/`)

- **`faster_rcnn.py`**:
  - Constructs `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
  - Configures backbone feature pyramid network (FPN) extracting feature maps from stages C2 through C5 ($P_2, P_3, P_4, P_5, P_6$).
  - Modifies Fast R-CNN classification & regression box head:
    ```python
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    ```
  - Configures anchor generator scales and aspect ratios suited for elongated bone fractures.
  - Factory function `build_model(config)` builds configurable variants (ResNet50, ResNet101, MobileNetV3).

---

### 2.3 Training Layer (`src/engine/trainer.py`)

- **Multi-task Loss Optimization**:
  Torchvision Faster R-CNN in training mode returns a dictionary of 4 distinct losses:
  1. `loss_classifier`: Cross-entropy classification loss of the RoI head.
  2. `loss_box_reg`: Smooth L1 / GIoU box regression loss of the RoI head.
  3. `loss_objectness`: Binary cross-entropy loss of the Region Proposal Network (RPN).
  4. `loss_rpn_box_reg`: Smooth L1 box regression loss of the RPN.
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{rpn\_obj}} + \mathcal{L}_{\text{rpn\_box}}$$
- **Optimization Features**:
  - Automatic Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`) and `GradScaler`.
  - Gradient clipping (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients.
  - Checkpoint persistence: saves `checkpoint_epoch_XX.pth`, `best_model.pth`, and `last_model.pth`.

---

### 2.4 Evaluation Layer (`src/engine/evaluator.py` & `src/utils/metrics.py`)

- **Standard COCO Evaluation**:
  - Uses `torchmetrics.detection.MeanAveragePrecision(box_format="xyxy", iou_type="bbox")`.
  - Computes $\text{mAP@0.5}$ and $\text{mAP@0.5:0.95}$.
  - Computes Precision, Recall, and F1-score across all IoU thresholds and candidate confidence levels.
- **Support for True Negatives**:
  - Negative samples (healthy X-rays without fracture) are included during evaluation. Any predicted bounding box above the operating confidence threshold on a negative image is penalized as a False Positive (FP).
- **Threshold Calibration**:
  - Sweep validation set predictions over confidence range $[0.05, 0.95]$ to determine operating point maximizing F1-score.

---

### 2.5 Inference Layer (`src/inference/predictor.py` & `scripts/predict.py`)

- **High-Level Predictor (`FracturePredictor`)**:
  - Accepts raw file paths, PIL Images, or NumPy arrays (`uint8` or `uint16`).
  - Automatically detects 16-bit uint16 dynamic range and scales by $65535.0$; 8-bit uint8 scaled by $255.0$.
  - Executes evaluation forward pass on configured device (`cuda` or `cpu`).
  - Filters predictions by score threshold $\tau$ (default: 0.5) and IoU NMS threshold (default: 0.5).
  - Measures execution latency with microsecond accuracy.
  - Generates clinical diagnostic overlays with prominent top header banners:
    - **CRIMSON BANNER**: `DIAGNOSIS: FRACTURE DETECTED (N fractures found, Max Conf: XX.X%)`
    - **GREEN BANNER**: `DIAGNOSIS: NO FRACTURE DETECTED (Max Conf: XX.X%)`
  - Returns structured results:
    ```python
    {
        "boxes": [[x1, y1, x2, y2], ...],
        "labels": ["fracture", ...],
        "scores": [0.89, ...],
        "image_size": (width, height),
        "num_detections": 1,
        "has_fracture": True,
        "latency_ms": 38.4
    }
    ```
- **Production CLI (`scripts/predict.py`)**:
  - Supports `--image <path>` for single radiographs or `--input-dir <path>` for recursive folder processing.
  - Automatically exports visual overlays (`pred_*.png`), structured detections JSON (`predictions.json`), and tabular summary CSV (`predictions.csv`).

---

### 2.6 Experimentation Layer (`src/utils/experiment_tracker.py` & `scripts/run_experiment.py`)

- **Modular Configuration Inheritance**:
  - Deep-merges baseline configuration (`configs/baseline.yaml`) with experimental overrides (`configs/experiments.yaml`).
  - Isolates checkpoints, TensorBoard event logs, and evaluation metrics per experiment in `experiments/exp_XX_<name>/`.
- **Ablation Studies**:
  - **EXP-01: Baseline**: Standard Faster R-CNN ResNet-50-FPN, SGD, StepLR.
  - **EXP-02: Augmentation**: Contrast-Limited Adaptive Histogram Equalization (CLAHE) + subtle random rotation ($\pm 10^\circ$).
  - **EXP-03: Backbone**: Architecture comparison across ResNet-50, ResNet-101, and MobileNetV3-Large.
  - **EXP-04: Resolution**: Multi-scale input resolution scaling ($600 \times 1000$ vs $800 \times 1333$ vs $1024 \times 1600$).
  - **EXP-05: Hyperparameters**: Optimization algorithm (AdamW vs SGD) and learning rate scheduling (Cosine Annealing vs StepLR).
- **Automated Summary Tracking**:
  - `ExperimentTracker` maintains `outputs/metrics/experiments_summary.csv` and auto-generates Markdown comparison tables.

---

### 2.7 Verification & Testing Layer (`tests/`)

Comprehensive automated unit and integration test suite structured across 9 test modules:
1. `test_annotations.py`: YOLO coordinate parsing, Pascal-VOC conversion, edge clamping, negative tensor structures.
2. `test_audit.py`: Split disjointness, patient-level leakage detection, intensity profiling, annotation syntax validation.
3. `test_dataset.py`: PyTorch dataset instantiation, 8-bit/16-bit loading, positive/negative sample collation.
4. `test_model.py`: Faster R-CNN 2-class head, ResNet-50/101/MobileNetV3 backbones, forward/backward gradient flows.
5. `test_trainer.py`: Multi-task training step, checkpoint serialization and resumption, loss convergence.
6. `test_evaluator.py`: Standard COCO mAP, threshold sweep, false-positive penalties on negative controls.
7. `test_transforms.py`: Medical-safe augmentation, CLAHE, horizontal flips with box coordination.
8. `test_experiments.py`: Config deep-merging, experiment runner overrides, summary table logging.
9. `test_predictor.py`: End-to-end inference API, 16-bit uint16 scaling, batch prediction, diagnostic visual overlays.

**Test Suite Metric**: 51/51 automated tests passing (`OK`).

---

## 3. Path Resolution & Environment Independence

To ensure the codebase runs seamlessly on local workstations, testing runners, and Google Colab:
- All file operations accept relative paths or dynamic root overrides.
- Priority order:
  1. CLI argument `--data-root`
  2. Environment variable `GRAZPEDWRI_DATA_ROOT`
  3. Configuration file entry `data.root_dir` (default: `/content/dataset`)
- Split lists contain filenames or paths; the data loader extracts `os.path.basename(path)` to avoid hardcoded `/content/dataset/images` assumptions.
