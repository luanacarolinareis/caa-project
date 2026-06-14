"""Generate figures for the binary classification results.

Produces (all from the real per-seed and ensemble metric JSONs, so the figures always match the metrics tables):
  - Confusion matrices for each model/seed and the ensemble
  - Training curves (head-only + fine-tune) for the transfer models, and the single-phase curve for the custom CNN
  - Metric comparison bar chart across models

Usage:
    python scripts/generate_plots_binary.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.visualization.plots import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_training_curves,
)

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
MODELS = ["custom_cnn", "resnet18", "densenet121"]
TRANSFER_MODELS = ["resnet18", "densenet121"]
SEEDS = [0, 1, 2]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
ENSEMBLE_MODEL = "ensemble_resnet18_densenet121"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _cm_from_dict(cm: dict) -> np.ndarray:
    """Build a 2x2 confusion matrix [[TN, FP], [FN, TP]] from the JSON dict."""
    return np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])


def generate_confusion_matrices() -> None:
    for model in MODELS:
        for seed in SEEDS:
            path = METRICS_DIR / f"{model}_seed{seed}.json"
            if not path.exists():
                print(f"  WARNING: missing {path}")
                continue
            data = load_json(path)
            cm = _cm_from_dict(data["confusion_matrix"])
            out = plot_confusion_matrix(
                cm,
                model_name=f"{model}_seed{seed}",
                class_names=CLASS_NAMES,
                output_dir=FIGURES_DIR,
            )
            print(f"Saved: {out}")

    # Ensemble confusion matrix
    ensemble_path = METRICS_DIR / f"{ENSEMBLE_MODEL}.json"
    if ensemble_path.exists():
        data = load_json(ensemble_path)
        cm = _cm_from_dict(data["confusion_matrix"])
        out = plot_confusion_matrix(
            cm,
            model_name="ensemble_binary",
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
    # Custom CNN: single-phase history
    for seed in SEEDS:
        path = METRICS_DIR / f"custom_cnn_seed{seed}_history.json"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        data = load_json(path)
        history = _extract_history(data["history"])
        out = plot_training_curves(
            history, model_name=f"custom_cnn_seed{seed}", output_dir=FIGURES_DIR
        )
        print(f"Saved: {out}")

    # Transfer models: head-only + fine-tune phases
    for model in TRANSFER_MODELS:
        for seed in SEEDS:
            for phase in ["head_only", "fine_tune"]:
                path = METRICS_DIR / f"{model}_seed{seed}_{phase}_history.json"
                if not path.exists():
                    print(f"  WARNING: missing {path}")
                    continue
                data = load_json(path)
                history = _extract_history(data["history"])
                label = f"{model}_seed{seed}_{phase}"
                out = plot_training_curves(history, model_name=label, output_dir=FIGURES_DIR)
                print(f"Saved: {out}")


def generate_metric_comparison() -> None:
    results = {}
    metrics_to_plot = ["accuracy", "precision", "recall", "specificity", "f1", "auroc"]
    display_names = {"custom_cnn": "Custom CNN", "resnet18": "ResNet18", "densenet121": "DenseNet121"}

    for model in MODELS:
        values: dict[str, list[float]] = {m: [] for m in metrics_to_plot}
        for seed in SEEDS:
            path = METRICS_DIR / f"{model}_seed{seed}.json"
            if not path.exists():
                continue
            data = load_json(path)
            for m in metrics_to_plot:
                if data.get(m) is not None:
                    values[m].append(data[m])
        if values["accuracy"]:
            results[display_names[model]] = {m: float(np.mean(v)) for m, v in values.items() if v}

    if not results:
        print("No binary metrics found.")
        return

    out = plot_metric_comparison(results, metrics=metrics_to_plot, output_dir=FIGURES_DIR)
    renamed = FIGURES_DIR / "metric_comparison_binary.png"
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
