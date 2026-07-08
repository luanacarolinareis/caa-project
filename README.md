[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vfAwOX_j)

# Pneumonia Classification from Pediatric Chest X-Rays

This project trains and evaluates deep learning models for pneumonia classification from pediatric chest X-ray images. It covers two tasks: binary classification (NORMAL vs PNEUMONIA) as a baseline, and a three-class extension (NORMAL vs BACTERIAL PNEUMONIA vs VIRAL PNEUMONIA) as the main innovation. Models compared include a custom CNN baseline, ResNet18, and DenseNet121, evaluated with classification metrics, inference time, Grad-CAM visual explanations, calibration analysis, ensembling, and a hierarchical two-stage variant.

The project is based on the public Kaggle/Kermany Chest X-Ray Images (Pneumonia) dataset. This repository is for educational experimentation only and is not intended for clinical diagnosis or real-world triage.

The trained model weights are published on the Hugging Face Hub: **[luanacarolina/pneumonia-chest-xray-classifier](https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier)**.

## Project Information

- Authors: Carolina Reis, University of Aveiro, Portugal; Jakub Błaszczyk, University of Aveiro, Portugal and Lodz University of Technology, Poland.
- Course unit: Complements of Machine Learning.
- Academic year: 2025/2026.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Key Documents and Notebooks](#key-documents-and-notebooks)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Setup](#setup)
- [One-Shot Pipelines](#one-shot-pipelines)
- [Quick Start](#quick-start)
- [Baseline: Binary Classification](#baseline-binary-classification)
- [Innovation: Three-Class Classification and Beyond](#innovation-three-class-classification-and-beyond)
  - [Three-Class Training and Evaluation](#three-class-training-and-evaluation)
  - [Hierarchical Two-Stage Variant](#hierarchical-two-stage-variant)
  - [Ensemble](#ensemble)
  - [Calibration Analysis](#calibration-analysis)
  - [Figures](#figures)
- [Scripts Reference](#scripts-reference)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Notes](#notes)

## Repository Structure

```text
.
├── configs/                      YAML experiment configs
│   ├── bacteria_virus.yaml       Stage B of the hierarchical variant (BACTERIA vs VIRUS)
│   ├── default.yaml              Binary task
│   └── three_class.yaml          Three-class task (patient-aware split, class weights)
├── data/                         Dataset placement (not committed to Git)
│   └── README.md                 Where to download the dataset and the expected layout
├── notebooks/                    Exploratory and results notebooks (see below)
├── reports/
│   ├── Final Project Report/     Final report (PDF + LaTeX source)
│   └── Project Proposal Report/  Initial project proposal and scope definition
├── results/                      Generated outputs (metrics, figures, Grad-CAM, checkpoints, run logs)
├── scripts/                      Command-line entry points for all tasks (see Scripts Reference)
├── src/
│   └── pneumonia_classifier/     Reusable library (data, models, training, metrics, Grad-CAM)
├── pyproject.toml                Package metadata
└── requirements.txt              Python dependencies
```

## Key Documents and Notebooks

- **Final Report:** [FinalProjectReport.pdf](reports/Final%20Project%20Report/FinalProjectReport.pdf)
- **Project Proposal:** [ProjectProposalReport.pdf](reports/Project%20Proposal%20Report/ProjectProposalReport.pdf)
- **Notebook 0 (Dataset check):** [00_dataset_check.ipynb](notebooks/00_dataset_check.ipynb) — dataset overview, class distribution, and NORMAL/BACTERIA/VIRUS X-ray examples.
- **Notebook 1 (Binary experiments):** [01_final_experiments.ipynb](notebooks/01_final_experiments.ipynb) — binary results, training curves, confusion matrices, Grad-CAM, and Hugging Face upload of the binary checkpoints.
- **Notebook 2 (Innovation results):** [02_innovation_results.ipynb](notebooks/02_innovation_results.ipynb) — three-class results, hierarchical comparison, calibration, Grad-CAM, and Hugging Face upload of the three-class / hierarchical checkpoints.
- **Data pipeline notes:** [data_pipeline.md](src/pneumonia_classifier/docs/data_pipeline.md) — how labels and the patient-aware split are built.
- **Dataset setup:** [data/README.md](data/README.md) — where to download the dataset, the expected folder layout, and per-split image counts.

## Key Results

All metrics are computed on the official 624-image test set. Single-model rows are mean ± std over seeds {0, 1, 2}; the ensemble is deterministic (averaged probabilities of the six ResNet18 + DenseNet121 checkpoints).

### Binary Classification (NORMAL vs PNEUMONIA)

| Model | Accuracy | Recall | Specificity | F1 | AUROC |
|---|---|---|---|---|---|
| Custom CNN | 0.756 ± 0.090 | 0.914 ± 0.076 | 0.494 ± 0.359 | 0.828 ± 0.040 | 0.886 ± 0.016 |
| ResNet18 | 0.830 ± 0.018 | 0.991 ± 0.005 | 0.561 ± 0.056 | 0.879 ± 0.011 | 0.949 ± 0.005 |
| DenseNet121 | 0.841 ± 0.020 | 0.997 ± 0.002 | 0.581 ± 0.053 | 0.887 ± 0.012 | 0.966 ± 0.007 |
| **Ensemble** | **0.856** | **0.997** | **0.620** | **0.896** | **0.963** |

The ensemble misses only **1 of 390** pneumonia cases (recall 0.997).

### Three-Class Classification (NORMAL / BACTERIA / VIRUS)

Evaluated with a leakage-free **patient-aware** validation split, inverse-frequency class weights, and macro-F1 checkpoint selection. Macro-averaged metrics.

| Model | Accuracy | F1 macro | AUROC |
|---|---|---|---|
| ResNet18 | 0.769 ± 0.089 | 0.751 ± 0.099 | 0.944 ± 0.019 |
| DenseNet121 | 0.799 ± 0.029 | 0.784 ± 0.033 | 0.944 ± 0.012 |
| DenseNet121 (hierarchical) | 0.790 ± 0.019 | 0.774 ± 0.021 | 0.944 ± 0.002 |
| **Ensemble** | **0.822** | **0.808** | **0.960** |

Per-class F1 (ensemble): NORMAL 0.782 | BACTERIA 0.924 | VIRUS 0.718. Viral pneumonia is the hardest class, and the dominant error is NORMAL predicted as VIRUS. The hierarchical two-stage variant did **not** beat the flat softmax, indicating the viral/bacterial boundary is the intrinsic bottleneck.

## Dataset

See [data/README.md](data/README.md) for full download instructions, the expected folder layout, and per-split image counts.

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

The dataset is not committed to Git. Three-class labels (BACTERIA / VIRUS) are derived automatically from the filenames inside the PNEUMONIA directories (`person100_bacteria_4.jpeg` vs `person154_virus_2.jpeg`). No manual annotation required.

## Setup

**Requirements:** Python 3.10+, CUDA-compatible GPU recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `pneumonia_classifier` package is installed in editable mode via `pyproject.toml`.

## One-Shot Pipelines

Two PowerShell scripts run an entire task end-to-end (train all seeds, evaluate, ensemble, figures, Grad-CAM, result tables). Each run produces two logs: the **full raw terminal output** as a timestamped `.log` under `results/logs/`, and a **clean per-epoch summary table** as a Markdown file in `results/`.

```powershell
# Binary baseline: trains custom CNN + ResNet18 + DenseNet121 (3 seeds each),
# evaluates, builds the ensemble, generates figures, Grad-CAM, and tables.
# Raw log -> results/logs/binary_run_<timestamp>.log
# Summary -> results/run_binary.md
.\scripts\run_binary.ps1

# Innovation: three-class training + evaluation, hierarchical two-stage variant,
# ensemble, calibration, figures, Grad-CAM, and tables.
# Raw log -> results/logs/innovation_run_<timestamp>.log
# Summary -> results/run_innovation.md
.\scripts\run_innovation.ps1
```

The latest committed summaries are at [run_binary.md](results/run_binary.md) and [run_innovation.md](results/run_innovation.md); the corresponding raw terminal logs live in [results/logs/](results/logs/).

## Quick Start

Run a single model end-to-end (DenseNet121, seed 0, binary):

```powershell
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0
python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt --seed 0
python scripts/generate_gradcam.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt
python scripts/make_result_tables.py
```

---

## Baseline: Binary Classification

Binary classification between NORMAL and PNEUMONIA using a custom CNN, ResNet18, and DenseNet121, each trained with seeds 0, 1, and 2.

```powershell
# Train (repeat --seed 0/1/2 for each)
python scripts/train_baseline.py --config configs/default.yaml --seed 0
python scripts/train_transfer.py --model resnet18 --config configs/default.yaml --seed 0
python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0

# Evaluate (repeat per model and seed)
python scripts/evaluate_model.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt --seed 0

# Grad-CAM (binary) and summary table
python scripts/generate_gradcam.py --model densenet121 --checkpoint results/checkpoints/densenet121_seed0.pt
python scripts/make_result_tables.py
```

---

## Innovation: Three-Class Classification and Beyond

The Kermany dataset encodes bacterial and viral subtypes in the filenames. Since bacterial and viral pneumonia require different treatments, we extended the task to three-class classification: NORMAL, BACTERIA, and VIRUS. Additional components:

- **Ensemble**: probability averaging across all 6 transfer-learning checkpoints (ResNet18 + DenseNet121 × 3 seeds).
- **Hierarchical two-stage variant**: a binary NORMAL/PNEUMONIA stage (reusing the binary checkpoints) followed by a dedicated BACTERIA/VIRUS classifier.
- **Calibration analysis**: ECE, reliability diagrams, and optimal decision thresholds for two clinical scenarios.

### Three-Class Training and Evaluation

```powershell
# Train (repeat --seed 0/1/2 for resnet18 and densenet121)
python scripts/train_three_class.py --model densenet121 --seed 0

# Evaluate (repeat per model and seed)
python scripts/evaluate_three_class.py --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed0.pt --seed 0

# Summary table and Grad-CAM (three-class)
python scripts/make_result_tables.py --task three_class
python scripts/generate_gradcam.py --config configs/three_class.yaml --model densenet121 --checkpoint results/checkpoints/densenet121_3class_seed1.pt
```

### Hierarchical Two-Stage Variant

Stage A is the existing binary checkpoint; only Stage B (BACTERIA vs VIRUS) is trained. Evaluation reuses the binary checkpoint for Stage A and reports the same three-class metrics for a direct comparison.

```powershell
# Train Stage B (BACTERIA vs VIRUS) for DenseNet121, all seeds
python scripts/train_stage_b.py --model densenet121 --seed 0

# Evaluate the two-stage pipeline (Stage A = binary checkpoint, Stage B = above)
python scripts/evaluate_hierarchical.py --model densenet121 --seed 0
```

### Ensemble

```powershell
python scripts/ensemble.py                                     # binary
python scripts/ensemble.py --config configs/three_class.yaml   # three-class
```

### Calibration Analysis

Computes ECE and reliability diagrams, and finds optimal decision thresholds for two clinical scenarios: screening (maximise recall) and assisted diagnosis (maximise F1). Requires the `_probs.json` files produced by the evaluation scripts.

```powershell
python scripts/calibration_analysis.py
```

### Figures

All result figures are regenerated from the per-seed and ensemble metric JSONs, so they always match the metrics tables. Run evaluation and ensemble first, then:

```powershell
python scripts/generate_plots_binary.py        # binary figures
python scripts/generate_plots_three_class.py   # three-class figures
```

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `train_baseline.py` | Train the custom CNN baseline (binary). |
| `train_transfer.py` | Train a transfer-learning backbone (ResNet18 / DenseNet121) on the binary task. |
| `train_three_class.py` | Train a backbone on the three-class task (patient-aware split, class weights, F1 selection). |
| `train_stage_b.py` | Train the BACTERIA-vs-VIRUS classifier (Stage B of the hierarchical variant). |
| `evaluate_model.py` | Evaluate a binary checkpoint on the test set; writes metrics and `_probs.json`. |
| `evaluate_three_class.py` | Evaluate a three-class checkpoint; writes metrics and `_probs.json`. |
| `evaluate_hierarchical.py` | Evaluate the two-stage pipeline (binary Stage A + Stage B) with three-class metrics. |
| `ensemble.py` | Average probabilities across checkpoints (binary or three-class). |
| `calibration_analysis.py` | Compute ECE, reliability diagrams, and optimal decision thresholds. |
| `generate_gradcam.py` | Produce Grad-CAM overlays (binary or three-class) with descriptive filenames. |
| `generate_plots_binary.py` | Regenerate all binary figures from metric JSONs. |
| `generate_plots_three_class.py` | Regenerate all three-class figures from metric JSONs. |
| `make_result_tables.py` | Build the CSV/Markdown summary tables (`--task three_class` for the 3-class table). |
| `log_to_markdown.py` | Convert a captured terminal log into a Markdown file (`--output`, `--title`). |
| `run_binary.ps1` | One-shot binary pipeline (train → evaluate → ensemble → figures → Grad-CAM → tables), logged to Markdown. |
| `run_innovation.ps1` | One-shot innovation pipeline (three-class + hierarchical + calibration), logged to Markdown. |

## Outputs

| Path | Contents |
|---|---|
| `results/metrics/` | Per-seed JSON metrics, CSV summaries, Markdown tables, calibration results |
| `results/figures/` | Confusion matrices, training curves, metric comparisons, reliability diagrams, dataset samples |
| `results/gradcam/` | Grad-CAM overlays (binary and three-class, with descriptive filenames) |
| `results/checkpoints/` | Saved model checkpoints (not committed to Git) |
| `results/logs/` | Full raw terminal output of each pipeline run (timestamped `.log`) |
| `results/run_binary.md`, `results/run_innovation.md` | Clean per-epoch pipeline summaries |

## Reproducibility

- All experiments use fixed seeds (0, 1, 2) via `torch.manual_seed` and `random.seed`.
- Single-model results are reported as mean ± std across 3 seeds.
- Python 3.10+, PyTorch 2.x; see `requirements.txt` for the full, pinned dependency list.
- GPU used: NVIDIA (CUDA); CPU fallback is supported but significantly slower.
- Images resized to 224 × 224; grayscale X-rays converted to 3-channel RGB; ImageNet normalisation for the pretrained models.

## Notes

- The binary task uses the provided Kaggle train/val/test split as-is.
- The three-class task uses a **patient-aware** stratified validation split (whole patients held out) to avoid the data leakage that an image-level split would introduce on the Kermany dataset.
- Pneumonia recall is prioritised because false negatives are clinically costly.
- Three-class labels are derived from filenames without any manual annotation.
