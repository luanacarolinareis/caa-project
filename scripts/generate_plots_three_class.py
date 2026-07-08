"""Generate figures for the three-class classification results.

Produces:
  - Confusion matrices for each model/seed and the ensemble
  - Training curves (head-only + fine-tune) for each model/seed
  - Metric comparison bar chart across models

Usage:
    python scripts/generate_plots_three_class.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.visualization.plots import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_training_curves,
)

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
MODELS = ["resnet18", "densenet121"]
SEEDS = [0, 1, 2]
CLASS_NAMES = ["NORMAL", "BACTERIA", "VIRUS"]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def generate_confusion_matrices() -> None:
    for model in MODELS:
        for seed in SEEDS:
            path = METRICS_DIR / f"{model}_3class_seed{seed}.json"
            if not path.exists():
                print(f"  WARNING: missing {path}")
                continue
            data = load_json(path)
            cm = np.array(data["confusion_matrix"])
            out = plot_confusion_matrix(
                cm,
                model_name=f"{model}_3class_seed{seed}",
                class_names=CLASS_NAMES,
                output_dir=FIGURES_DIR,
            )
            print(f"Saved: {out}")

    # Ensemble confusion matrix
    ensemble_path = METRICS_DIR / "ensemble_resnet18_densenet121_3class.json"
    if ensemble_path.exists():
        data = load_json(ensemble_path)
        cm = np.array(data["confusion_matrix"])
        out = plot_confusion_matrix(
            cm,
            model_name="ensemble_resnet18_densenet121_3class",
            class_names=CLASS_NAMES,
            output_dir=FIGURES_DIR,
        )
        print(f"Saved: {out}")


def _extract_history(history_json: list[dict]) -> dict[str, list[float]]:
    """Convert the training history JSON format to plot_training_curves format."""
    train_loss, val_loss = [], []
    train_f1, val_f1 = [], []
    for row in history_json:
        train_loss.append(row["train"]["loss"])
        val_loss.append(row["val"]["loss"])
        train_f1.append(row["train"].get("f1_macro", row["train"].get("f1", 0)))
        val_f1.append(row["val"].get("f1_macro", row["val"].get("f1", 0)))
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_f1,
        "val_acc": val_f1,
    }


def generate_training_curves() -> None:
    for model in MODELS:
        for seed in SEEDS:
            for phase in ["head_only", "fine_tune"]:
                path = METRICS_DIR / f"{model}_3class_seed{seed}_{phase}_history.json"
                if not path.exists():
                    print(f"  WARNING: missing {path}")
                    continue
                data = load_json(path)
                history = _extract_history(data["history"])
                label = f"{model}_3class_seed{seed}_{phase}"
                out = plot_training_curves(history, model_name=label, output_dir=FIGURES_DIR)
                print(f"Saved: {out}")


def generate_metric_comparison() -> None:
    results = {}
    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auroc"]

    for model in MODELS:
        values: dict[str, list[float]] = {m: [] for m in metrics_to_plot}
        for seed in SEEDS:
            path = METRICS_DIR / f"{model}_3class_seed{seed}.json"
            if not path.exists():
                continue
            data = load_json(path)
            for m in metrics_to_plot:
                if data.get(m) is not None:
                    values[m].append(data[m])
        if values["accuracy"]:
            results[model] = {m: float(np.mean(v)) for m, v in values.items() if v}

    # Add ensemble
    ensemble_path = METRICS_DIR / "ensemble_resnet18_densenet121_3class.json"
    if ensemble_path.exists():
        data = load_json(ensemble_path)
        results["ensemble"] = {m: data[m] for m in metrics_to_plot if data.get(m) is not None}

    if not results:
        print("No 3-class metrics found.")
        return

    out = plot_metric_comparison(
        results,
        metrics=metrics_to_plot,
        output_dir=FIGURES_DIR,
    )
    renamed = FIGURES_DIR / "metric_comparison_3class.png"
    out.replace(renamed)
    print(f"Saved: {renamed}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Confusion matrices ===")
    generate_confusion_matrices()
    print("\n=== Training curves ===")
    generate_training_curves()
    print("\n=== Metric comparison ===")
    generate_metric_comparison()


if __name__ == "__main__":
    main()
