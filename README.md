# Hyperspectral Object Detection Challenge 2026: HS-SAFD

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](environment.yml)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Model: Single Detector](https://img.shields.io/badge/Single%20Model-Rules%20Compliant-brightgreen.svg)]()

**HS-SAFD** (**H**yperspectral **S**pectral-**A**ttention **F**usion **D**etector) is a spectral-spatial deep learning framework designed specifically for the **Hyperspectral Object Detection Challenge 2026**.

The system utilizes all **16 contiguous spectral bands (460–600 nm)** alongside multi-scale spatial pyramid features to detect and localize **18 fine-grained object categories**, with special emphasis on material-level discrimination between visually identical **Authentic vs. Counterfeit** material pairs.

---

## 1. Official Competition Rules & Constraints Compliance

| Requirement | Official Rule | HS-SAFD Implementation |
| :--- | :--- | :--- |
| **Training Dataset** | 3,000 16-band HSI cubes | Supported formats: `.npy`, `.npz`, `.tif`, `.mat` |
| **Annotations** | Pascal VOC XML format | Native `datasets/voc_parser.py` |
| **Test Dataset** | 1,000 unlabeled images | Handled in Phase 1 & Phase 2 pipelines |
| **Ranking Dataset** | 1,000 unlabeled images | Inference-only strictly isolated |
| **Spectral Range** | 16 bands (460–600 nm) | Band-aware normalization & projection |
| **Target Classes** | 18 categories | 9 Authentic vs Counterfeit paired classes |
| **Official Metric** | **mAP@[0.50:0.95]** | Multi-threshold COCO AP with CIoU Loss |
| **Final Submission** | **Single Detection Model** | **HS-SAFD Single Model** (No multi-model ensemble) |
| **Model Ensemble** | ❌ **PROHIBITED** | Multi-model WBF/voting strictly disabled |
| **Ranking-Set Usage** | ❌ **Inference Only** | Zero training, zero fine-tuning, zero BN update |
| **Inference TTA** | ✅ **ALLOWED** | Single-model multi-scale + flip TTA (`inference/tta.py`) |
| **Top 10 Code Review** | Mandatory verification | Complete `release/` bundle with SHA256 and FLOPs |

---

## 2. HS-SAFD Architecture Overview

```
                          16-BAND HYPERSPECTRAL INPUT
                          [B, 16, H, W] (460 – 600 nm)
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   Band-Aware Standardize  │
                         │ X[b] = (X[b]-μ[b])/(σ[b]+ε)│
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │    Spectral Projection    │
                         │  1×1 Conv: 16 → 32        │
                         │  3×3 Conv: 32 → 64        │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │     Spectral Attention    │
                         │   Global Pooling → MLP    │
                         │   Sigmoid → 16 Band Wts   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      Spatial Attention    │
                         │   AvgPool & MaxPool → 7×7 │
                         │   Foreground Region Mask  │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Multi-Scale Spatial CNN  │
                         │   C3 (/8), C4 (/16), C5   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      FPN / PAN Neck       │
                         │ Bi-directional Aggregation│
                         │     P3, P4, P5 Features   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Decoupled Detection Head │
                         │   • 18 Class Logits       │
                         │   • CIoU Box Regression   │
                         │   • Objectness Confidence │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Single-Model TTA + NMS   │
                         │ Multi-scale + Flips fused │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                                 submission.csv
```

---

## 3. Target Categories & Real vs. Counterfeit Pairs

The competition features 18 target categories grouped into 9 Authentic / Counterfeit material pairs:

| ID | Class Name | Paired ID | Paired Class Name | Material Diagnostic |
| :---: | :--- | :---: | :--- | :--- |
| `0` | `real_leather` | `1` | `fake_leather` | Organic protein vs synthetic PVC/PU |
| `2` | `real_silk` | `3` | `synthetic_silk` | Fibroin reflectance vs polyester |
| `4` | `real_gemstone` | `5` | `glass_gemstone` | Mineral crystal absorption vs amorphous glass |
| `6` | `real_banknote` | `7` | `counterfeit_banknote` | Security ink / rag paper vs commercial dye |
| `8` | `real_wood` | `9` | `laminated_wood` | Lignin/cellulose vs melamine resin |
| `10` | `real_wool` | `11` | `synthetic_wool` | Keratin fiber vs acrylic polymer |
| `12` | `real_jade` | `13` | `counterfeit_jade` | Nephrite/jadeite vs dyed serpentine/quartz |
| `14` | `authentic_chip` | `15` | `counterfeit_chip` | Silicon substrate / epoxy packaging |
| `16` | `natural_honey` | `17` | `adulterated_honey` | Fructose/glucose vs corn syrup |

---

## 4. Repository Structure

```
hyperspectral-object-detection/
├── configs/
│   ├── base.yaml              # Base configurations
│   ├── baseline.yaml          # 3-band RGB baseline
│   ├── hyperspectral.yaml     # 16-band standard CNN baseline
│   └── final.yaml             # Full HS-SAFD single model configuration
├── data/
│   ├── raw/                   # train/ (3000), test/ (1000), ranking/ (1000), annotations/
│   ├── processed/             # Cached spectral statistics and pre-processed cubes
│   └── splits/                # Stratified train.txt and val.txt splits
├── datasets/
│   ├── hyperspectral_dataset.py # PyTorch Dataset (16-band cubes, on-the-fly transforms)
│   ├── voc_parser.py          # Native Pascal VOC XML annotation parser
│   ├── transforms.py          # Continuous spectral augmentations & spatial letterbox
│   ├── collate.py             # Custom variable-box collate function
│   └── visualization.py       # Pseudo-RGB, false-color & spectral curve visualizer
├── models/
│   ├── hyperspectral_detector.py # End-to-end HS-SAFD and baseline models
│   ├── backbone/              # SpatialBackbone, SpectralBackbone, HybridBackbone
│   ├── attention/             # SpectralAttention, SpatialAttention, ChannelAttention
│   ├── neck/                  # FeaturePyramidNetwork, FPN, PAN
│   └── head/                  # DecoupledHead, MultiScaleDetectionHead
├── losses/
│   ├── detection_loss.py      # Unified multi-task loss
│   ├── hyperspectral_loss.py  # CIoU box loss + Focal loss + Objectness BCE
│   ├── iou_loss.py            # Complete IoU (CIoU) math
│   └── classification_loss.py # Multi-label Focal Loss & Label Smoothing
├── training/
│   ├── train.py               # Training CLI entry point
│   ├── trainer.py             # Training loop, AMP, gradient accumulation, validation
│   ├── optimizer.py           # AdamW with weight decay separation
│   ├── scheduler.py           # Cosine annealing with linear warmup
│   └── checkpoint.py          # State serialization and recovery
├── evaluation/
│   ├── evaluate.py            # Complete validation CLI
│   ├── metrics.py             # mAP@[0.50:0.95], AP75, AP90, Precision, Recall
│   ├── per_class.py           # Class-by-class AP table & Real vs Counterfeit delta
│   ├── confusion_matrix.py    # 19x19 confusion matrix generator
│   └── error_analysis.py      # 6-bucket diagnostic error analysis
├── inference/
│   ├── predict.py             # Single-model test/ranking inference with TTA
│   ├── tta.py                 # Multi-scale & flip Test-Time Augmentation
│   ├── postprocess.py         # Coordinate unpadding, clipping & single-model NMS
│   └── nms.py                 # Fast greedy & Soft-NMS
├── submission/
│   ├── generate_submission.py # Generates Kaggle-compliant submission.csv
│   ├── validate_submission.py # Strict 12-point submission validator (Phase 1 & 2)
│   └── sample_submission.csv  # Example submission file
├── release/
│   ├── reproduction.sh        # Official bash reproduction script
│   ├── reproduction.bat       # Official Windows batch reproduction script
│   ├── model_stats.txt        # Parameters count (18.59M) & FLOPs (23.87 GFLOPs)
│   └── submission.sha256      # SHA256 checksum of submission.csv
├── scripts/
│   ├── inspect_dataset.py     # Automatic dataset diagnostics and class inspection
│   ├── prepare_data.py        # Generates splits and spectral normalization stats
│   ├── create_split.py        # Stratified train/val split generator
│   └── calculate_stats.py     # FLOPs and per-band dataset statistics calculator
├── tests/                     # Full automated unit test suite (17 passed)
└── requirements.txt           # Project dependencies
```

---

## 5. Quickstart & Command-Line Workflow

### Step 1: Environment Setup
```bash
# Using Conda
conda env create -f environment.yml
conda activate hsi-detect

# Or using Pip
pip install -r requirements.txt
```

### Step 2: Dataset Inspection & VOC XML Parsing
```bash
py -3.10 scripts/inspect_dataset.py --data-dir data/raw
```
Reports total scene count, dimensions, 16-band statistics, VOC annotations count, objects per image, and class frequencies.

### Step 3: Data Preparation & Stratified Split
```bash
py -3.10 scripts/prepare_data.py --raw-dir data/raw --val-ratio 0.2
```
Calculates per-band mean/std ($\mu[b], \sigma[b]$) and creates `data/splits/train.txt` and `data/splits/val.txt`.

### Step 4: Model Complexity & FLOPs Calculation
```bash
py -3.10 scripts/calculate_stats.py --config configs/final.yaml
```
Outputs model parameter count (**18,593,703**) and FLOPs (**23.87 GFLOPs**) into `release/model_stats.txt` for official code review.

### Step 5: Training the Final Single Model (HS-SAFD)
```bash
py -3.10 training/train.py --config configs/final.yaml
```
Features:
- Auto Mixed Precision (AMP) when running on CUDA
- Multi-task Loss: Focal Classification + CIoU Box Regression + Objectness
- Evaluates validation mAP@[0.50:0.95] every epoch
- Checkpoints saved to `checkpoints/exp04_final/best.pt`

### Step 6: Evaluation & Per-Class Material Analysis
```bash
py -3.10 evaluation/evaluate.py \
    --config configs/final.yaml \
    --checkpoint checkpoints/exp04_final/best.pt
```
Outputs:
- Official `mAP@[0.50:0.95]`, `mAP@0.50`, `mAP@0.75`, `mAP@0.90`
- Per-class AP table for all 18 categories
- Real vs. Counterfeit paired discrimination metrics
- 6-bucket error classification (Correct, False Positive, False Negative, Poor Localization, Wrong Class, Duplicate)
- Confusion matrix plot saved to `outputs/metrics/confusion_matrix.png`

### Step 7: Single-Model Inference with TTA
```bash
py -3.10 inference/predict.py \
    --config configs/final.yaml \
    --checkpoint checkpoints/final.pt \
    --input data/raw/test \
    --output outputs/predictions \
    --tta
```

### Step 8: Submission Generation
```bash
py -3.10 submission/generate_submission.py \
    --predictions outputs/predictions/predictions.json \
    --output outputs/submissions/submission.csv
```

### Step 9: Submission Validation
```bash
# Phase 1: Test set only
py -3.10 submission/validate_submission.py \
    --input outputs/submissions/submission.csv \
    --phase phase1

# Phase 2: Test set (1,000) + Ranking set (1,000)
py -3.10 submission/validate_submission.py \
    --input outputs/submissions/submission.csv \
    --phase phase2
```
Checks:
- Header and column order (`id,image_id,class_id,confidence,x1,y1,x2,y2`)
- Sequential IDs (0 to N-1) with no duplicates
- Valid class IDs in `[0, 17]`
- Confidence scores in `[0.0, 1.0]`
- Bounding box geometric validity ($x_1 < x_2$, $y_1 < y_2$)
- Non-negative coordinates
- Zero NaN and zero Inf values
- Test and ranking image representation

---

## 6. Official Code Review & Reproduction Check (Top 10)

The competition organizers require top 10 finalists to reproduce their exact `submission.csv` via automated script:

```bash
# On Linux / Bash
bash release/reproduction.sh

# On Windows
release\reproduction.bat
```

The script runs single-model inference, generates `submission.csv`, validates all 12 constraints, and calculates its SHA256 checksum:
```bash
cat release/submission.sha256
```

---

## 7. Model Statistics Summary

From `release/model_stats.txt`:
- **Model Name**: HS-SAFD (Hyperspectral Spectral-Attention Fusion Detector)
- **Input Channels**: 16 bands (460 – 600 nm)
- **Input Resolution**: $512 \times 512$
- **Total Parameters**: 18,593,703
- **Trainable Parameters**: 18,593,703
- **Estimated FLOPs**: 23.87 GFLOPs
- **Model Checkpoint Size**: ~70.9 MB (FP32)

---

## 8. Citation

```bibtex
@misc{hyperspectral-object-detection-challenge-2026,
    author = {HotTracking2025},
    title = {Hyperspectral Object Detection Challenge 2026},
    year = {2026},
    howpublished = {\url{https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026}},
    note = {Kaggle}
}
```

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
