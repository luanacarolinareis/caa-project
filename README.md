# Pneumonia Classification from Pediatric Chest X-Rays

This project trains and evaluates lightweight deep learning models for binary pneumonia classification from pediatric chest X-ray images. The planned comparison includes a small custom CNN baseline, ResNet18, and DenseNet121, evaluated with classification metrics, inference time, and Grad-CAM visual sanity checks.

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
- [Example Usage](#example-usage)
- [Planned Outputs](#planned-outputs)
- [Notes](#notes)

## Project Structure

```text
Project/
  Additional Files/           Reports, presentation, template, and supporting material
  configs/                    Experiment configuration files
  data/                       Dataset placement instructions only
  notebooks/                  Dataset checks and final experiment notebook
  results/                    Generated metrics, figures, Grad-CAM outputs, checkpoints
  scripts/                    Command-line entry points
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

The raw dataset should not be committed to Git.

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

## Example Usage

These commands are the intended workflow.

Train the custom CNN baseline:

```powershell
python scripts/train_baseline.py --config configs/default.yaml --seed 0
```

Train a transfer-learning model:

```powershell
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0
```

Evaluate a trained checkpoint:

```powershell
python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt
```

Generate Grad-CAM examples:

```powershell
python scripts/generate_gradcam.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt
```

Build final result tables:

```powershell
python scripts/make_result_tables.py
```

## Planned Outputs

- Metrics saved under `results/metrics/`.
- Figures saved under `results/figures/`.
- Grad-CAM examples saved under `results/gradcam/`.
- Local checkpoints saved under `results/checkpoints/`.

## Notes

- The provided Kaggle split will be used as-is for train/validation/test.
- Images will be resized to `224 x 224`.
- Grayscale X-rays will be converted to 3-channel RGB for pretrained models.
- ImageNet normalization will be used for ResNet18 and DenseNet121.
- Pneumonia recall is especially important because false negatives are clinically costly.
