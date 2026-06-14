# Pneumonia Classification from Pediatric Chest X-Rays

This project trains and evaluates deep learning models for pneumonia classification from pediatric chest X-ray images. It covers two tasks: binary classification (NORMAL vs PNEUMONIA) as a baseline, and a three-class extension (NORMAL vs BACTERIAL PNEUMONIA vs VIRAL PNEUMONIA) as the main innovation. Models compared include a custom CNN baseline, ResNet18, and DenseNet121, evaluated with classification metrics, inference time, Grad-CAM visual explanations, calibration analysis, and ensemble methods.

The project is based on the public Kaggle/Kermany Chest X-Ray Images (Pneumonia) dataset. This repository is for educational experimentation only and is not intended for clinical diagnosis or real-world triage.

## Project Information

- Authors: Carolina Reis, University of Aveiro, Portugal; Jakub Błaszczyk, University of Aveiro, Portugal and Lodz University of Technology, Poland.
- Course unit: Complements of Machine Learning.
- Academic year: 2025/2026.
- Date: 16/04/2026.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Baseline: Binary Classification](#baseline-binary-classification)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Grad-CAM and Result Tables](#grad-cam-and-result-tables)
- [Innovation: Three-Class Classification and Beyond](#innovation-three-class-classification-and-beyond)
  - [Three-Class Training](#three-class-training)
  - [Three-Class Evaluation](#three-class-evaluation)
  - [Ensemble](#ensemble)
  - [Calibration Analysis](#calibration-analysis)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Notes](#notes)

## Repository Structure

```text
.
├── configs/                      YAML experiment configs (default.yaml, three_class.yaml)
├── data/                         Dataset placement (not committed to Git)
├── docs/                         Technical documentation (training and evaluation guides)
├── notebooks/                    Exploratory and results notebooks
├── reports/
│   ├── Final Project Report/     Final report (PDF + LaTeX source)
│   └── Project Proposal Report/  Initial project proposal and scope definition
├── results/                      Generated outputs (metrics, figures, Grad-CAM, checkpoints)
├── scripts/                      Command-line entry points for all tasks
├── src/
│   └── pneumonia_classifier/     Reusable library (data, models, training, metrics, Grad-CAM)
├── pyproject.toml                Package metadata
└── requirements.txt              Python dependencies
```

**Project Proposal:** [ProjectProposalReport.pdf](reports/Project%20Proposal%20Report/ProjectProposalReport.pdf)

**Final Report:** [FinalProjectReport.pdf](reports/Final%20Project%20Report/FinalProjectReport.pdf)

## Key Results

### Binary Classification (NORMAL vs PNEUMONIA)

| Model | Accuracy | F1 | AUROC |
|---|---|---|---|
| Custom CNN | baseline | -- | -- |
| ResNet18 | ~0.855 | ~0.896 | ~0.963 |
| DenseNet121 | ~0.855 | ~0.896 | ~0.963 |
| **Ensemble** | **0.8558** | **0.8963** | **0.9630** |

### Three-Class Classification (NORMAL / BACTERIA / VIRUS)

| Model | Accuracy | F1 macro | AUROC |
|---|---|---|---|
| ResNet18 | 0.769 +/- 0.005 | 0.755 +/- 0.002 | 0.933 +/- 0.016 |
| DenseNet121 | 0.814 +/- 0.029 | 0.801 +/- 0.032 | 0.956 +/- 0.006 |
| **Ensemble** | **0.8173** | **0.8035** | **0.9613** |

Per-class F1 (ensemble): NORMAL 0.774 | BACTERIA 0.914 | VIRUS 0.722

## Dataset

Download the dataset from Kaggle:

```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Place it at `data/chest_xray/` with this structure:

```
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

The dataset is not committed to Git. Three-class labels (BACTERIA / VIRUS) are derived automatically from the filenames inside the PNEUMONIA directories. No manual annotation required.

## Setup

**Requirements:** Python 3.10+, CUDA-compatible GPU recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `pneumonia_classifier` package is installed in editable mode via `pyproject.toml`.

## Quick Start

Run the full pipeline end-to-end (example with DenseNet121, seed 0):

```powershell
# 1. Train (binary)
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0

# 2. Evaluate
python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt --seed 0

# 3. Grad-CAM
python scripts/generate_gradcam.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt

# 4. Result table
python scripts/make_result_tables.py
```

See the sections below for the full multi-seed pipeline.

---

## Baseline: Binary Classification

Binary classification between NORMAL and PNEUMONIA using a custom CNN, ResNet18, and DenseNet121.

All models were trained with seeds 0, 1, and 2 for reproducibility.

### Training

**Custom CNN baseline:**

```powershell
python scripts/train_baseline.py --config configs/default.yaml --seed 0
python scripts/train_baseline.py --config configs/default.yaml --seed 1
python scripts/train_baseline.py --config configs/default.yaml --seed 2
```

**ResNet18:**

```powershell
python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 0
python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 1
python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 2
```

**DenseNet121:**

```powershell
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 1
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 2
```

### Evaluation

```powershell
python scripts/evaluate_model.py --model custom_cnn --checkpoint results/checkpoints/custom_cnn_seed0.pt --seed 0
python scripts/evaluate_model.py --model custom_cnn --checkpoint results/checkpoints/custom_cnn_seed1.pt --seed 1
python scripts/evaluate_model.py --model custom_cnn --checkpoint results/checkpoints/custom_cnn_seed2.pt --seed 2

python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed0.pt --seed 0
python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed1.pt --seed 1
python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed2.pt --seed 2

python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt --seed 0
python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed1.pt --seed 1
python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed2.pt --seed 2
```

### Grad-CAM and Result Tables

```powershell
# Grad-CAM visualisation (binary)
python scripts/generate_gradcam.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt

# Summary table
python scripts/make_result_tables.py
```

---

## Innovation: Three-Class Classification and Beyond

The Kermany dataset encodes bacterial and viral subtypes in the filenames (`person100_bacteria_4.jpeg` vs `person154_virus_2.jpeg`). Since bacterial and viral pneumonia require different treatments, we extended the task to three-class classification: NORMAL, BACTERIA, and VIRUS.

Additional innovations:
- **Ensemble** -- probability averaging across all 6 checkpoints (ResNet18 + DenseNet121 x 3 seeds)
- **Calibration analysis** -- ECE, reliability diagrams, optimal decision thresholds for two clinical scenarios

### Three-Class Training

ResNet18 and DenseNet121 are trained with seeds 0, 1, and 2:

```powershell
python scripts/train_three_class.py --model resnet18 --seed 0
python scripts/train_three_class.py --model resnet18 --seed 1
python scripts/train_three_class.py --model resnet18 --seed 2

python scripts/train_three_class.py --model densenet121 --seed 0
python scripts/train_three_class.py --model densenet121 --seed 1
python scripts/train_three_class.py --model densenet121 --seed 2
```

### Three-Class Evaluation

```powershell
python scripts/evaluate_three_class.py --model resnet18 --checkpoint results/checkpoints/resnet18_3class_seed0.pt --seed 0
python scripts/evaluate_three_class.py --model resnet18 --checkpoint results/checkpoints/resnet18_3class_seed1.pt --seed 1
python scripts/evaluate_three_class.py --model resnet18 --checkpoint results/checkpoints/resnet18_3class_seed2.pt --seed 2

python scripts/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed0.pt --seed 0
python scripts/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed1.pt --seed 1
python scripts/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed2.pt --seed 2

# Summary table
python scripts/make_result_tables.py --task three_class

# Grad-CAM (three-class)
python scripts/generate_gradcam.py --config configs/three_class.yaml --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed1.pt
```

### Ensemble

```powershell
python scripts/ensemble.py                                        # binary
python scripts/ensemble.py --config configs/three_class.yaml     # three-class
```

### Calibration Analysis

Analyses model calibration (ECE, reliability diagrams) and finds optimal decision thresholds for two clinical scenarios: screening (maximise recall) and assisted diagnosis (maximise F1). Requires the `_probs.json` files produced by the evaluation scripts above.

```powershell
python scripts/calibration_analysis.py
```

### Figures

All result figures (confusion matrices, training curves, and the metric-comparison bar charts) are regenerated from the real per-seed and ensemble metric JSONs, so they always match the metrics tables. Run the evaluation and ensemble steps first, then:

```powershell
python scripts/generate_plots_binary.py        # binary figures
python scripts/generate_plots_three_class.py   # three-class figures
```

---

## Outputs

| Path | Contents |
|---|---|
| `results/metrics/` | Per-seed JSON metrics, CSV summaries, markdown tables, calibration results |
| `results/figures/` | Confusion matrices, training curves, metric comparison, reliability diagrams |
| `results/gradcam/` | Grad-CAM overlays (binary TP/TN/FP/FN and three-class correct/wrong examples) |
| `results/checkpoints/` | Saved model checkpoints (not committed to Git) |

## Reproducibility

- All experiments use fixed seeds (0, 1, 2) via `torch.manual_seed` and `random.seed`
- Results are reported as mean +/- std across 3 seeds
- Python 3.10+, PyTorch 2.x, see `requirements.txt` for full dependency list
- GPU used: NVIDIA (CUDA) -- CPU fallback supported but significantly slower
- Images resized to 224 x 224; grayscale X-rays converted to 3-channel RGB; ImageNet normalisation for pretrained models

## Notes

- The provided Kaggle split is used as-is for train/validation/test.
- Images are resized to `224 x 224`.
- Grayscale X-rays are converted to 3-channel RGB for pretrained models.
- ImageNet normalisation is used for ResNet18 and DenseNet121.
- Pneumonia recall is especially important because false negatives are clinically costly.
- Three-class labels are derived from filenames without any manual annotation.
