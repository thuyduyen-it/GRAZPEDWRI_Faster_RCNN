# GRAZPEDWRI-DX Faster R-CNN Training Guide (Phase 4)

## 1. Overview & Objectives

This guide documents the procedures, configurations, and best practices for training the **Faster R-CNN with ResNet-50-FPN** architecture on the pediatric wrist fracture dataset (**GRAZPEDWRI-DX**).

The training pipeline implemented in `src/engine/trainer.py` and invoked via `scripts/train.py` satisfies medical imaging requirements:
- **Binary Detection Target**: Background (0) and Fracture (1).
- **Patient-Level Leakage Free**: Strictly partitioned splits (14,210 train, 3,032 val, 3,085 test).
- **16-bit Native Normalization**: Automatic range-aware floating-point scaling ($x / 65535.0$).
- **True Negative Sample Inclusion**: 6,777 non-fracture radiographs integrated with zero-box targets (`[0, 4]` boxes, `[0]` labels) to suppress false positives.
- **Robust Multi-Task Loss Tracking**: Classification loss, Bounding box regression loss, RPN objectness loss, and RPN box regression loss.
- **Production Checkpoint & Resume**: Automatic state tracking with `last_model.pth` and `best_model.pth`, preserving optimizer, scheduler, epoch, and validation metrics for seamless resumption after Colab disconnections.

---

## 2. Training Hardware Requirements

| Specification | Minimum (Verification) | Recommended (Full Training) |
|---|---|---|
| **Platform** | Local Machine (CPU or GPU) | Google Colab / GPU Cloud Server |
| **GPU** | Optional (Intel/AMD CPU) | NVIDIA Tesla T4 (16 GB VRAM) or A100 |
| **CUDA Version** | N/A | CUDA 11.8 / 12.x |
| **RAM** | 8 GB System RAM | 12+ GB System RAM |
| **Disk Space** | 2 GB (Code + Sample data) | 25 GB (Unzipped GRAZPEDWRI images & labels) |

---

## 3. Configuration & Hyperparameters

Baseline configuration is defined in `configs/baseline.yaml`:

```yaml
experiment:
  name: "baseline_faster_rcnn_resnet50"
  seed: 42

model:
  name: "fasterrcnn_resnet50_fpn"
  backbone: "resnet50"
  num_classes: 2              # 0: Background, 1: Fracture
  pretrained: true            # COCO pre-trained weights for transfer learning
  trainable_backbone_layers: 3

training:
  epochs: 20
  batch_size: 4               # Optimized for Tesla T4 16GB
  mixed_precision: true       # Automatic Mixed Precision (torch.amp) for 2x speedup
  clip_grad_norm: 10.0
  val_interval: 1
  save_interval: 1
  early_stopping:
    patience: 5
    min_delta: 0.001

optimizer:
  type: "SGD"
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005

lr_scheduler:
  type: "StepLR"
  step_size: 6
  gamma: 0.1
```

---

## 4. Google Colab Execution Workflow

### Step 4.1: Connect Runtime & Mount Google Drive
In Google Colab, select runtime: **Runtime -> Change runtime type -> T4 GPU**.

```python
from google.colab import drive
import os
drive.mount('/content/drive')
```

### Step 4.2: Extract Dataset to Colab Local SSD
To maximize I/O throughput, extract dataset ZIP files directly to `/content/dataset`:

```bash
mkdir -p /content/dataset/images /content/dataset/labels

# Extract labels
unzip -q "/content/drive/MyDrive/GRAZPEDWRI/fracture_labels_yolo.zip" -d /content/dataset/labels

# Extract image parts (images_part1 to part4)
unzip -q "/content/drive/MyDrive/GRAZPEDWRI/images_part1.zip" -d /content/dataset/images
unzip -q "/content/drive/MyDrive/GRAZPEDWRI/images_part2.zip" -d /content/dataset/images
unzip -q "/content/drive/MyDrive/GRAZPEDWRI/images_part3.zip" -d /content/dataset/images
unzip -q "/content/drive/MyDrive/GRAZPEDWRI/images_part4.zip" -d /content/dataset/images
```

### Step 4.3: Set Dataset Root Variable
```bash
export GRAZPEDWRI_DATA_ROOT=/content/dataset
```

### Step 4.4: Clone / Navigate to Project Directory
```bash
cd /content/drive/MyDrive/GRAZPEDWRI_Faster_RCNN
pip install -r requirements.txt
```

---

## 5. Training Commands

### 5.1 Pipeline Dry-Run Verification (Sanity Check)
Runs 1 epoch on sample batches to ensure that data loaders, transforms, forward pass, loss calculation, backward gradient step, validation mAP calculation, and checkpoint writing execute with zero faults:

```bash
python scripts/train.py --config configs/baseline.yaml --dry-run
```

### 5.2 Full Baseline Training (20 Epochs on Tesla T4)
Launches full model training using multi-task loss, StepLR scheduling, and AMP:

```bash
python scripts/train.py --config configs/baseline.yaml
```

### 5.3 Resuming from Interrupted Checkpoint
If Google Colab disconnects or reaches the idle limit, reconnect and resume from the latest saved state without losing optimizer momentum or learning rate schedule:

```bash
python scripts/train.py --config configs/baseline.yaml --resume checkpoints/last_model.pth
```

---

## 6. Monitoring & Output Artifacts

### 6.1 Real-Time TensorBoard Visualization
Launch TensorBoard directly inside Google Colab:
```python
%load_ext tensorboard
%tensorboard --logdir outputs/logs
```
Tracked metrics include:
- `Loss/train_total`, `Loss/classifier`, `Loss/box_reg`, `Loss/rpn_objectness`, `Loss/rpn_box_reg`
- `Metrics/val_mAP_50`, `Metrics/val_mAP_50_95`
- `Training/learning_rate`

### 6.2 Loss History CSV
Per-epoch metrics are automatically saved to `outputs/loss_curves/training_history.csv` for post-training reporting, plotting, and ablation analysis.

### 6.3 Model Checkpoints
- `checkpoints/best_model.pth`: Checkpoint achieving the highest validation mAP@0.5.
- `checkpoints/last_model.pth`: Latest epoch checkpoint with full state dict (weights, optimizer, scheduler, epoch, metric).

---

## 7. Troubleshooting & Recovery

| Issue | Root Cause | Solution |
|---|---|---|
| **CUDA Out of Memory (OOM)** | Batch size 4 exceeds VRAM during large image batches | Reduce batch size to 2 via CLI or config: `data.loader.batch_size: 2`. |
| **Colab Session Disconnect** | 12-hour session timeout or network idle | Save checkpoints to Google Drive (`/content/drive/MyDrive/...`) and pass `--resume checkpoints/last_model.pth`. |
| **Network Timeout on Backbone Download** | Pretrained weights attempt download in an offline environment | Set `model.pretrained: false` or pre-download `resnet50-0676ba61.pth` to Torch cache directory. |
| **Missing Images Warning** | Path mismatch in split text files | The loader automatically strips `/content/dataset/images/` and resolves filenames dynamically against `GRAZPEDWRI_DATA_ROOT`. |
