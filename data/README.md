# Dataset Directory Layout & Metadata Documentation

This directory manages dataset metadata, patient-level splits, and schema definitions for the **GRAZPEDWRI-DX** pediatric wrist fracture detection project.

---

## 1. Directory Structure

```text
data/
├── metadata/
│   ├── dataset.csv            # Tabular metadata describing all 20,327 images, patients, and diagnoses
│   ├── dataset.yaml           # YAML configuration declaring dataset classes and splits
│   └── split_by_patient.csv   # Patient-level partition mapping each patient and image to Train/Val/Test
│
├── splits/
│   ├── train.txt              # List of image file paths/names for training (14,210 images)
│   ├── val.txt                # List of image file paths/names for validation (3,032 images)
│   └── test.txt               # List of image file paths/names for testing (3,085 images)
│
└── README.md                  # This documentation file
```

---

## 2. Dataset Overview (GRAZPEDWRI-DX)

- **Total Images**: 20,327
- **Total Annotation Files**: 20,327
- **Positive Images (with fracture)**: 13,550
- **Negative Images (without fracture)**: 6,777 (retained as true negative background samples)
- **Total Fracture Bounding Boxes**: 18,090
- **Invalid / Out-of-range Annotations**: 0
- **Target Detection Class**: Fracture

### Critical Label Mapping Distinction
- **YOLO Format**: Class ID `0` represents Fracture.
- **Torchvision Faster R-CNN Format**:
  - Class ID `0`: Background (reserved by Torchvision)
  - Class ID `1`: Fracture (target object)
- **Converter Rule**: The dataset loader strictly maps YOLO `class_id = 0` $\rightarrow$ Faster R-CNN `label = 1`.

### Dedicated Annotation Source
- **Fracture Labels**: Annotations must come from `fracture_labels_yolo.zip` (single class: fracture).
- **Caution**: Do **NOT** use `folder_structure.zip` annotations directly for model training, as it contains multi-class labels (text markers, foreign objects, non-fracture conditions) that deviate from the single-class fracture detection task.

---

## 3. Patient-Level Splitting & Leakage Prevention

The splits are strictly constructed at the **patient level** to eliminate data leakage across evaluation folds:

| Split | Images | Unique Patients | Images with Fracture |
|---|---:|---:|---:|
| **Train** | 14,210 | 4,262 | 9,491 |
| **Validation** | 3,032 | 913 | 2,038 |
| **Test** | 3,085 | 916 | 2,021 |
| **Total** | **20,327** | **6,091** | **13,550** |

- Patient overlap between splits: **0**
- Test set is strictly isolated and held out for final evaluation.

---

## 4. Runtime Dataset Path Resolution

The image and label files are stored externally (e.g. in Google Colab under `/content/dataset/` or on an external drive) and are **never** committed to version control.

`train.txt`, `val.txt`, and `test.txt` may contain legacy absolute paths (e.g. `/content/dataset/images/image_001.png`). The project data loaders resolve image paths dynamically:
1. Check `GRAZPEDWRI_DATA_ROOT` environment variable.
2. If unset, fall back to `configs/baseline.yaml` (`data.root_dir`).
3. Safely extract filename basename from `train.txt` entries and join with the resolved `images/` directory.
