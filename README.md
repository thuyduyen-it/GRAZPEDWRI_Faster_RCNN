# Pediatric Wrist Fracture Detection Using Faster R-CNN (GRAZPEDWRI-DX)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-orange.svg)](https://pytorch.org/vision/)
[![Tests Passing](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic and clinical-grade deep learning system for automated pediatric wrist fracture detection and localization on digital radiographs using **Faster R-CNN** with a **ResNet-50-FPN** backbone, developed on the **GRAZPEDWRI-DX** benchmark dataset.

---

## 1. Clinical Context & Executive Summary

Pediatric wrist injuries account for up to 25% of all pediatric fractures. Evaluating pediatric radiographs is notoriously challenging due to:
- **Incomplete ossification**: Presence of cartilaginous growth plates (physis) and secondary ossification centers that mimic fracture lines.
- **Subtle pediatric-specific fractures**: Buckle (torus) and greenstick fractures exhibiting only microscopic cortical buckling without obvious cortical disruption.
- **High false-negative rates**: Especially during emergency night shifts when non-specialist clinicians read radiographs.

This repository provides an end-to-end, reproducible, modular computer vision pipeline built with PyTorch and Torchvision. The system detects fracture regions via bounding box localization while maintaining strict patient-level separation to guarantee generalization and avoid data leakage.

---

## 2. Dataset Architecture & Verification

The system is trained and evaluated on **GRAZPEDWRI-DX**, a specialized pediatric wrist trauma radiograph dataset collected at the University Hospital Graz.

### 2.1 Dataset Statistics

| Metric | Verified Value | Clinical Significance |
|---|---:|---|
| **Total Radiographs** | **20,327** | High-volume single-center pediatric cohort |
| **Fracture-Positive Images** | **13,550** (66.7%) | Images containing one or more verified fractures |
| **True Negative Images** | **6,777** (33.3%) | Normal radiographs retained as negative controls |
| **Total Fracture Bounding Boxes** | **18,090** | Dense localization annotations |
| **Patients in Dataset** | **6,091** | Unique pediatric individuals |
| **Patient-Level Leakage** | **0** (0.0%) | Formally audited; zero patient overlap across splits |
| **Invalid Annotations** | **0** | All coordinates satisfy $0 \le x_c, y_c, w, h \le 1$ |

### 2.2 Patient-Level Split Summary

To prevent optimistic performance estimates caused by anatomical features unique to an individual child, splits are strictly partitioned at the **patient identifier level**:

| Split | Images | Unique Patients | Fracture-Positive | True Negatives | Total Boxes |
|---|---:|---:|---:|---:|---:|
| **Train** | 14,210 | 4,262 | 9,491 | 4,719 | 12,689 |
| **Validation** | 3,032 | 913 | 2,038 | 994 | 2,711 |
| **Test** | 3,085 | 916 | 2,021 | 1,064 | 2,690 |
| **Total** | **20,327** | **6,091** | **13,550** | **6,777** | **18,090** |

### 2.3 Key Technical Protocols
- **Label Remapping**: YOLO annotations designate class `0` as Fracture. Faster R-CNN strictly reserves label `0` for Background. The data layer transparently maps YOLO class `0` $\rightarrow$ Faster R-CNN label `1`.
- **True Background Controls**: Negative images (6,777 radiographs) are loaded with empty target tensors (`boxes`: `[0, 4]`, `labels`: `[0]`). Any detection predicted on negative controls during evaluation is strictly penalized as a False Positive (FP).
- **16-Bit Radiograph Support**: Native 16-bit grayscale radiographs (`uint16`) are dynamically scaled by $65535.0$ to $[0.0, 1.0]$, preserving full dynamic range.

---

## 3. Project Architecture

The codebase follows clean architecture principles with decoupled functional layers:

```
GRAZPEDWRI_Faster_RCNN/
├── configs/
│   ├── baseline.yaml            # Baseline ResNet-50-FPN hyperparameters
│   └── experiments.yaml         # 5 Ablation study configurations (EXP-01 to EXP-05)
├── data/
│   ├── metadata/                # dataset.csv, dataset.yaml, split_by_patient.csv
│   └── splits/                  # train.txt, val.txt, test.txt (patient-partitioned)
├── src/
│   ├── data/
│   │   ├── annotation.py        # YOLO -> Pascal-VOC parsing, coordinate clamping, empty targets
│   │   ├── dataset.py           # GRAZPEDWRIDataset, custom collate_fn, dynamic path resolution
│   │   └── transforms.py        # 16-bit normalization, CLAHE, random horizontal flips
│   ├── models/
│   │   └── faster_rcnn.py       # Modular Faster R-CNN builder (ResNet-50/101, MobileNetV3)
│   ├── engine/
│   │   ├── trainer.py           # Multi-task training loop, AMP, gradient clipping, StepLR/Cosine
│   │   └── evaluator.py         # COCO mAP@0.5/0.5:0.95, threshold sweep, FP on negatives
│   ├── inference/
│   │   └── predictor.py         # FracturePredictor API, latency timer, clinical banners
│   └── utils/
│       ├── metrics.py           # Standalone mAP, precision, recall, F1 computation
│       ├── visualization.py     # Medical X-ray bounding box drawing & status banners
│       ├── checkpoint.py        # Fault-tolerant checkpoint manager (best_model, last_model)
│       ├── logging.py           # Structured dual-output logger (stdout + file)
│       ├── seed.py              # Reproducible seed initialization
│       └── experiment_tracker.py# Experiment metrics tracking & Markdown table generator
├── scripts/
│   ├── inspect_dataset.py       # Phase 1: Dataset audit, leakage checks, distribution profiling
│   ├── prepare_dataset.py       # Phase 2: DataLoader verification and batch sample inspection
│   ├── train.py                 # Phase 4: Baseline training script
│   ├── evaluate.py              # Phase 5: Comprehensive evaluation and diagnostic visualization
│   ├── run_experiment.py        # Phase 6: Automated ablation experiment harness
│   └── predict.py               # Phase 7: Production single & batch radiograph inference CLI
├── notebooks/
│   └── GRAZPEDWRI_Faster_RCNN.ipynb # End-to-end Google Colab GPU execution notebook
├── checkpoints/                 # Saved model weights (*.pth, git-ignored)
├── experiments/                 # Per-experiment directories (exp_01 to exp_05)
├── outputs/
│   ├── logs/                    # Training and execution logs
│   ├── loss_curves/             # Scalar loss curves & CSV histories
│   ├── predictions/             # Inference visual overlays and diagnostics
│   └── metrics/                 # JSON audit results & summary CSVs
├── tests/                       # 9 Unit test suites (51 automated tests, 100% pass)
├── docs/
│   ├── DATASET_REPORT.md        # Detailed Phase 1 audit report
│   ├── DEVELOPMENT_PLAN.md      # 8-Phase engineering roadmap & completion record
│   ├── EXPERIMENT_PLAN.md       # Ablation hypotheses, specs, and metrics table
│   ├── TRAINING_GUIDE.md        # Colab GPU training & checkpoint recovery guide
│   ├── EVALUATION_REPORT.md     # Phase 5 evaluation methodology & threshold calibration
│   └── PROJECT_ARCHITECTURE.md  # Detailed software architecture & data flow
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 4. Quickstart Guide

### 4.1 Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/GRAZPEDWRI_Faster_RCNN.git
cd GRAZPEDWRI_Faster_RCNN

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Verify Local Installation (51 Unit Tests)

Run the full automated test suite covering annotations, data loading, transformations, backbones, training steps, metric calculation, and inference:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected output:
```text
Ran 51 tests in ~110s
OK
```

---

## 5. Command-Line Interface (CLI)

### 5.1 Dataset Inspection & Audit (Phase 1)
Audits file integrity, patient leakage, label bounds, and intensity statistics:
```bash
python scripts/inspect_dataset.py --config configs/baseline.yaml
```

### 5.2 DataLoader Verification (Phase 2)
Verifies positive and negative batch collation and exports sample visualizations:
```bash
python scripts/prepare_dataset.py --config configs/baseline.yaml --split val --verify-samples 5
```

### 5.3 Model Training (Phase 4)
Runs Faster R-CNN training with Automatic Mixed Precision (AMP), multi-loss tracking, and checkpointing:
```bash
python scripts/train.py --config configs/baseline.yaml
```
*To resume an interrupted run:*
```bash
python scripts/train.py --config configs/baseline.yaml --resume checkpoints/last_model.pth
```

### 5.4 Model Evaluation (Phase 5)
Computes mAP@0.5, mAP@0.5:0.95, performs threshold sweeping, penalizes false positives on negative controls, and exports diagnostic overlays:
```bash
python scripts/evaluate.py \
  --config configs/baseline.yaml \
  --checkpoint checkpoints/best_model.pth \
  --split test \
  --output-dir outputs/predictions/eval_test
```

### 5.5 Ablation Experiment Runner (Phase 6)
Executes controlled ablation studies defined in `configs/experiments.yaml`:
```bash
# Run specific ablation (e.g. CLAHE augmentation)
python scripts/run_experiment.py --experiment exp_02_augmentation

# Run mobile lightweight backbone ablation
python scripts/run_experiment.py --experiment exp_03_backbone --variant mobilenet_v3

# Run high-resolution ablation
python scripts/run_experiment.py --experiment exp_04_image_resolution --variant high_res
```

### 5.6 Radiograph Inference (Phase 7)

#### Single Radiograph Screening:
```bash
python scripts/predict.py \
  --image path/to/wrist_xray.png \
  --checkpoint checkpoints/best_model.pth \
  --score-threshold 0.5 \
  --output-dir outputs/predictions/screening
```

#### Batch Radiograph Screening:
```bash
python scripts/predict.py \
  --input-dir path/to/xray_folder/ \
  --checkpoint checkpoints/best_model.pth \
  --score-threshold 0.5 \
  --output-dir outputs/predictions/batch_screening
```
*Outputs generated:*
- Visual annotated overlays with clinical status banners (`pred_*.png`).
- Structured detections JSON (`predictions.json`).
- Tabular screening summary CSV (`predictions.csv`).

---

## 6. Python API Quickstart

Integrate the fracture detector directly into Python applications:

```python
import cv2
from src.inference.predictor import FracturePredictor

# Initialize predictor with trained checkpoint
predictor = FracturePredictor(
    checkpoint_path="checkpoints/best_model.pth",
    device="cuda",  # or "cpu"
    score_threshold=0.5,
    nms_iou_threshold=0.5
)

# Predict from image file path
result = predictor.predict("path/to/radiograph.png")
print(f"Fracture detected: {result['has_fracture']}")
print(f"Found {result['num_detections']} fracture region(s)")
print(f"Latency: {result['latency_ms']:.1f} ms")

for box, score in zip(result["boxes"], result["scores"]):
    print(f" - BBox [xmin, ymin, xmax, ymax]: {box}, Confidence: {score:.3f}")

# Generate clinical visual overlay with status banner
annotated_img, result = predictor.predict_and_visualize(
    "path/to/radiograph.png",
    output_path="outputs/predictions/screening_result.png"
)
```

---

## 7. Experimental Matrix & Benchmark Framework

Five controlled ablation studies are implemented in `configs/experiments.yaml`:

| Experiment ID | Study Focus | Key Configuration Overrides | Clinical / Technical Objective |
|---|---|---|---|
| **EXP-01** | Baseline | Faster R-CNN ResNet-50-FPN, SGD, StepLR | Establish reference benchmark |
| **EXP-02** | Augmentation | Medical CLAHE contrast + subtle rotation ($\pm 10^\circ$) | Improve visibility of subtle cortical cracks |
| **EXP-03** | Backbone Architecture | ResNet-101-FPN vs MobileNetV3-Large-FPN | Evaluate capacity vs mobile/edge deployment |
| **EXP-04** | Input Resolution | $600 \times 1000$ vs $800 \times 1333$ vs $1024 \times 1600$ | Assess trade-off between small fracture recall and memory |
| **EXP-05** | Hyperparameters | AdamW + Cosine Annealing schedule | Explore modern adaptive optimization dynamics |

All experiments log comparative metrics to `outputs/metrics/experiments_summary.csv`.

---

## 8. Google Colab GPU Workflow

Full model training on all 14,210 training radiographs should be executed on Google Colab (Tesla T4 GPU):

1. **Mount Google Drive** containing original archives:
   - `images_part1.zip` to `images_part4.zip`
   - `fracture_labels_yolo.zip`
2. **Open Notebook**: Open `notebooks/GRAZPEDWRI_Faster_RCNN.ipynb` in Google Colab.
3. **Set Runtime**: Select `Runtime > Change runtime type > T4 GPU`.
4. **Run Sequenced Cells**:
   - Section 1: Environment verification & GPU detection.
   - Section 2: Repository clone / pull from GitHub.
   - Section 3: Dataset extraction to `/content/dataset/images` and `/content/dataset/labels`.
   - Section 4: Phase 1 dataset audit & leakage check.
   - Section 5: Phase 4 baseline training (`python scripts/train.py`).
   - Section 6: Checkpoint synchronization back to Google Drive (`/content/drive/MyDrive/checkpoints/`).
   - Section 7: Phase 5 evaluation on test set.
   - Section 8: Phase 6 ablation experiments.
   - Section 9: Phase 7 clinical inference demo.

For step-by-step guidance, consult [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).

---

## 9. Development Roadmap & Completion Status

| Phase | Milestone | Deliverables & Verification | Status |
|:---:|---|---|:---:|
| **0** | Project Initialization | Repository layout, configuration schemas, package specs | **COMPLETED** ✅ |
| **1** | Dataset Audit | 20,327 images audited, 0 patient leakage, `dataset_audit.json` | **COMPLETED** ✅ |
| **2** | Data Preparation & Loader | 16-bit uint16 scaling, negative tensors, `dataset.py`, `transforms.py` | **COMPLETED** ✅ |
| **3** | Faster R-CNN Architecture | 2-class head, ResNet-50/101, MobileNetV3, anchor generators | **COMPLETED** ✅ |
| **4** | Baseline Training Engine | AMP, StepLR/Cosine, checkpoint manager, `trainer.py`, `train.py` | **COMPLETED** ✅ |
| **5** | Evaluation Engine | COCO mAP@0.5/0.5:0.95, threshold sweep, FP on negatives, diagnostics | **COMPLETED** ✅ |
| **6** | Experimental Studies | 5 ablation configurations (EXP-01 to EXP-05), tracker, summary CSV | **COMPLETED** ✅ |
| **7** | Inference & Packaging | `FracturePredictor` API, clinical status banners, batch CLI | **COMPLETED** ✅ |
| **8** | Final Documentation | Comprehensive README, architecture docs, unit test suite (51/51 OK) | **COMPLETED** ✅ |

---

## 10. Citation & Academic Reference

If you utilize this codebase or experimental protocol in your research, please cite:

```bibtex
@article{grazpedwri_faster_rcnn_2026,
  title={Pediatric Wrist Fracture Detection on X-ray Images Using Faster R-CNN with Patient-Level Partitioning},
  author={GRAZPEDWRI-DX Research Initiative},
  year={2026},
  journal={Computer Methods and Programs in Biomedicine}
}
```

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
