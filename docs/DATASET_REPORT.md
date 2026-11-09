# GRAZPEDWRI-DX Dataset Audit & Statistical Report

**Dataset**: GRAZPEDWRI-DX (Pediatric Wrist Radiograph Collection)  
**Target Class**: Fracture (Class ID 0)  
**Overall Audit Verdict**: **PASS**  
**Audit Timestamp**: 2026-11-03  

---

## 1. Executive Summary & Verification Matrix

| Audit Check | Scope / Criterion | Expected Reference | Actual Calculated | Status |
|---|---|---:|---:|:---:|
| **1. Metadata Files** | `dataset.csv`, `dataset.yaml`, `split_by_patient.csv` | 3 files present | 3 files present | **PASS** |
| **2. Split Definitions** | `train.txt`, `val.txt`, `test.txt` | 3 files present | 3 files present | **PASS** |
| **3. Path Resolution** | Configurable root via `GRAZPEDWRI_DATA_ROOT` | Dynamic resolution | Resolved (\content\dataset) | **PASS** |
| **4. Image-Label Matching** | Paired filenames across dataset records | 20,327 matched | 20327 matched | **PASS** |
| **5. Positive/Negative Balance** | Positive: 13,550 \| Negative: 6,777 | Pos: 13,550 / Neg: 6,777 | Pos: 13550 / Neg: 6777 | **PASS** |
| **6. Bounding Box Counts** | Class 0 (Fracture) lesion count | 18,090 boxes | 18090 boxes | **PASS** |
| **7. YOLO Syntax & Coords** | Normalized $0 \le x,y,w,h \le 1$, positive area | 0 syntax errors | 0 errors | **PASS** |
| **8. Split Isolation** | Inter-split image overlap (Train, Val, Test) | 0 overlap | 0 overlap | **PASS** |
| **9. Patient Leakage** | Patient appearing in multiple splits | 0 patient leakage | 0 leaking patients | **PASS** |
| **10. Label vs Metadata Match** | `has_fracture` consistency with bounding boxes | 100% agreement | 100% agreement | **PASS** |
| **11. Pixel Depth & Mode** | Bit depth, dimensions, intensity range | 16-bit uint16 profile | uint16 (16-bit) | **PASS** |
| **12. Sample Visualizations** | Generated overlays of positive and negative samples | Generated | 6 images saved | **PASS** |

---

## 2. Patient-Level Split Verification

The dataset partition is strictly executed at the **patient identifier level** to eliminate intra-patient correlation leakage across folds:

| Split | Images | Unique Patients | Images with Fracture | True Negative Images |
|---|---:|---:|---:|---:|
| **Train** | 14,210 | 4,262 | 9,491 | 4,719 |
| **Validation** | 3,032 | 913 | 2,038 | 994 |
| **Test** | 3,085 | 916 | 2,021 | 1,064 |
| **Total** | **20,327** | **6,091** | **13,550** | **6,777** |

### Split Leakage Audit
- **Patient overlap (Train $\cap$ Val)**: 0
- **Patient overlap (Train $\cap$ Test)**: 0
- **Patient overlap (Val $\cap$ Test)**: 0
- **Patient-level leakage check**: **PASSED**

---

## 3. Radiographic Image Characteristics & Intensity Profile

Empirical analysis on representative GRAZPEDWRI-DX studies:

| Sample Image | Dimensions (W x H) | Channels | Data Type | Bit Depth | Min Intensity | Max Intensity | Mean $\pm$ Std |
|---|---|---:|---|---:|---:|---:|---|
| `0001_1297860395_01_WRI-L1_M014.png` | 494 x 727 | 1 | `uint16` | 16-bit | 0.0 | 65535.0 | 26840.19 $\pm$ 10861.45 |
| `0001_1297860435_01_WRI-L2_M014.png` | 536 x 836 | 1 | `uint16` | 16-bit | 0.0 | 65535.0 | 26004.74 $\pm$ 15421.42 |
| `0002_0354485735_01_WRI-R1_F012.png` | 509 x 1204 | 1 | `uint16` | 16-bit | 0.0 | 65535.0 | 20736.24 $\pm$ 11012.01 |
| `0002_0354485759_01_WRI-R2_F012.png` | 469 x 1198 | 1 | `uint16` | 16-bit | 0.0 | 65535.0 | 15436.12 $\pm$ 11158.93 |
| `0003_0662359226_01_WRI-R1_M011.png` | 704 x 1130 | 1 | `uint16` | 16-bit | 0.0 | 65535.0 | 21709.05 $\pm$ 11421.6 |

> [!IMPORTANT]
> **Radiographic Insight**: The raw GRAZPEDWRI-DX X-ray images are stored as **16-bit grayscale (`uint16`)** PNG files spanning intensities up to 65,535. The `MedicalCompose` and `ToTensor` preprocessing pipeline correctly handles 16-bit normalization ($x / 65535.0$) to avoid truncating high-dynamic-range bone density details.

---

## 4. Sample Visualizations

Visual verification samples with bounding boxes are stored in:
`outputs/predictions/dataset_samples/`
