# Pneumonia Classification from Pediatric Chest X-Rays

This project trains and evaluates deep learning models for pneumonia classification from pediatric chest X-ray images. It covers two tasks: binary classification (NORMAL vs PNEUMONIA) as a baseline, and a three-class extension (NORMAL vs BACTERIAL PNEUMONIA vs VIRAL PNEUMONIA) as the main innovation. Models compared include a custom CNN baseline, ResNet18, and DenseNet121, evaluated with classification metrics, inference time, Grad-CAM visual explanations, calibration analysis, and ensemble methods.

The project is based on the public Kaggle/Kermany Chest X-Ray Images (Pneumonia) dataset. This repository is for educational experimentation only and is not intended for clinical diagnosis or real-world triage.

## Project Information

- Authors: Carolina Reis, University of Aveiro, Portugal; Jakub Błaszczyk, University of Aveiro, Portugal and Lodz University of Technology, Poland.
- Course unit: Complements of Machine Learning.
- Academic year: 2025/2026.
- Date: 16/04/2026.

## Table of Contents

- [Project Information](#project-information)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
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
- [Notes](#notes)

## Project Structure

```text
Project/
  Additional Files/           Reports, presentation, template, and supporting material
  configs/                    Experiment configuration files (default.yaml, three_class.yaml)
  data/                       Dataset placement instructions only
  notebooks/                  Dataset checks and final experiment notebook
  results/                    Generated metrics, figures, Grad-CAM outputs, checkpoints
  scripts/                    Command-line entry points for all tasks (binary and three-class)
  src/pneumonia_classifier/   Reusable project code
```

## Dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Place it locally using this structure:

```text
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

The raw dataset should not be committed to Git. The three-class labels (BACTERIA / VIRUS) are derived automatically from the filenames inside the PNEUMONIA directories: no additional annotation is needed.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Baseline: Binary Classification

Binary classification between NORMAL and PNEUMONIA using a custom CNN, ResNet18, and DenseNet121.

### Training

All models were trained with seeds 0, 1, and 2 for reproducibility.

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

Generate Grad-CAM examples:

```powershell
python scripts/generate_gradcam.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt
```

Build final result tables:

```powershell
python scripts/make_result_tables.py
```

---

## Innovation: Three-Class Classification and Beyond

The Kermany dataset encodes bacterial and viral subtypes in the filenames (`person100_bacteria_4.jpeg` vs `person154_virus_2.jpeg`). Since bacterial and viral pneumonia require different treatments, we extended the problem to three-class classification: NORMAL, BACTERIA, and VIRUS. This section also covers an ensemble of the best-performing models and a calibration analysis.

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
```

Build result tables for the three-class task:

```powershell
python scripts/make_result_tables.py --task three_class
```

Generate Grad-CAM examples for three-class models:

```powershell
python scripts/generate_gradcam.py --config configs/three_class.yaml --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed0.pt
```

### Ensemble

Averages the probabilities of all ResNet18 and DenseNet121 checkpoints:

```powershell
python scripts/ensemble.py                                    # binary
python scripts/ensemble.py --config configs/three_class.yaml  # three-class
```

### Calibration Analysis

Analyses model calibration (ECE, reliability diagrams) and finds optimal decision thresholds for two clinical scenarios: screening (maximise recall) and assisted diagnosis (maximise F1). Requires the `_probs.json` files produced by the evaluation scripts above.

```powershell
python scripts/calibration_analysis.py
```

---

## Outputs

- Metrics saved under `results/metrics/`.
- Figures saved under `results/figures/`.
- Grad-CAM examples saved under `results/gradcam/`.
- Local checkpoints saved under `results/checkpoints/`.

## Notes

- The provided Kaggle split is used as-is for train/validation/test.
- Images are resized to `224 x 224`.
- Grayscale X-rays are converted to 3-channel RGB for pretrained models.
- ImageNet normalisation is used for ResNet18 and DenseNet121.
- Pneumonia recall is especially important because false negatives are clinically costly.
- Three-class labels are derived from filenames without any manual annotation.
