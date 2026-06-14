"""Calibration analysis and threshold optimisation for binary classification.

For each model, loads the raw probability files (*_probs.json) produced by evaluate_model and computes:
  - Expected Calibration Error (ECE)
  - Reliability diagram (confidence vs accuracy per bin)
  - Optimal threshold for maximum recall (screening scenario)
  - Optimal threshold for maximum F1 (assisted diagnosis scenario)

Usage:
    python innovation/calibration_analysis.py
    python innovation/calibration_analysis.py --models resnet18 densenet121
    python innovation/calibration_analysis.py --seeds 0 1 2 --bins 15
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.utils import save_json

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
ALL_MODELS = ["custom_cnn", "resnet18", "densenet121", "ensemble_resnet18_densenet121"]
SEEDS = [0, 1, 2]

# Calibration helpers

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centres, mean_accuracy, mean_confidence) for each bin."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, confs = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        accs.append(y_true[mask].mean())
        confs.append(y_prob[mask].mean())
    return np.array(centres), np.array(accs), np.array(confs)


def optimal_threshold_recall(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.99
) -> float:
    """Highest threshold that still achieves at least min_recall (screening)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one extra point: align with thresholds
    valid = recall[:-1] >= min_recall
    if not valid.any():
        return float(thresholds[np.argmax(recall[:-1])])
    return float(thresholds[valid][-1])


def optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises F1 score."""
    _, _, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    return float(thresholds[np.argmax(f1s)])

# Plotting

def plot_reliability_diagram(models_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path, n_bins: int,
) -> None:
    """Plot reliability diagrams for all models on one figure."""
    n = len(models_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax, (model_name, (centres, accs, confs)) in zip(axes.flat, models_data.items()):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.bar(centres, accs, width=1 / n_bins, alpha=0.6, label="Accuracy", color="steelblue")
        ax.plot(confs, accs, "o-", color="firebrick", lw=1.5, ms=4, label="Model")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(model_name)
        ax.legend(fontsize=7)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Reliability Diagrams", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reliability diagram: {output_path}")

# Per-model analysis

def _load_probs(model: str, seed: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Load y_true / y_prob for a model+seed combination."""
    if seed is None:
        # Ensemble: single file, no seed suffix
        path = METRICS_DIR / f"{model}_probs.json"
    else:
        path = METRICS_DIR / f"{model}_seed{seed}_probs.json"

    if not path.exists():
        return None

    with path.open() as f:
        data = json.load(f)

    y_true = np.asarray(data["y_true"], dtype=int)
    y_prob = np.asarray(data["y_prob"], dtype=float)
    return y_true, y_prob


def analyse_model(model: str, seeds: list[int] | None, n_bins: int,
) -> dict | None:
    """Aggregate calibration metrics across seeds for one model."""
    is_ensemble = seeds is None
    seed_list = [None] if is_ensemble else seeds

    ece_values, thresh_recall_values, thresh_f1_values = [], [], []
    diagram_data: tuple | None = None

    for seed in seed_list:
        result = _load_probs(model, seed)
        if result is None:
            label = "ensemble" if seed is None else f"seed={seed}"
            print(f"  WARNING: probs file not found for {model} ({label}), skipping")
            continue

        y_true, y_prob = result
        ece_values.append(compute_ece(y_true, y_prob, n_bins))
        thresh_recall_values.append(optimal_threshold_recall(y_true, y_prob))
        thresh_f1_values.append(optimal_threshold_f1(y_true, y_prob))

        # Use last seed's data for the reliability diagram
        diagram_data = reliability_diagram_data(y_true, y_prob, n_bins)

    if not ece_values:
        return None

    summary = {
        "model": model,
        "ece_mean": float(np.mean(ece_values)),
        "ece_std": float(np.std(ece_values)) if len(ece_values) > 1 else None,
        "threshold_screening_mean": float(np.mean(thresh_recall_values)),
        "threshold_f1_mean": float(np.mean(thresh_f1_values)),
    }

    if diagram_data is not None:
        centres, accs, confs = diagram_data
        summary["reliability_diagram"] = {
            "bin_centres": centres.tolist(),
            "mean_accuracy": accs.tolist(),
            "mean_confidence": confs.tolist(),
        }

    return summary

# Main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration analysis for trained models.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="Models to analyse.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seeds to aggregate over.")
    parser.add_argument("--bins", type=int, default=10, help="Number of calibration bins.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    diagram_inputs: dict[str, tuple] = {}

    for model in args.models:
        is_ensemble = model == "ensemble_resnet18_densenet121"
        seeds = None if is_ensemble else args.seeds
        print(f"Analysing {model} ...")
        summary = analyse_model(model, seeds, args.bins)
        if summary is None:
            continue

        results.append(summary)

        if "reliability_diagram" in summary:
            rd = summary["reliability_diagram"]
            diagram_inputs[model] = (
                np.array(rd["bin_centres"]),
                np.array(rd["mean_accuracy"]),
                np.array(rd["mean_confidence"]),
            )

    if not results:
        print("No probability files found. Re-run evaluate_model.py for each checkpoint first.")
        return

    # Save JSON summary
    output_json = METRICS_DIR / "calibration_results.json"
    save_json(results, output_json)
    print(f"\nSaved calibration results: {output_json}")

    # Plot reliability diagrams
    if diagram_inputs:
        plot_reliability_diagram(diagram_inputs, FIGURES_DIR / "reliability_diagrams.png", args.bins)

    print(f"\n{'Model':<40} {'ECE':>8}  {'Thr (recall)':>13}  {'Thr (F1)':>10}")
    print("-" * 78)
    for r in results:
        ece_str = f"{r['ece_mean']:.4f}"
        if r.get("ece_std") is not None:
            ece_str += f" ±{r['ece_std']:.4f}"
        print(
            f"{r['model']:<40} {ece_str:>8}  "
            f"{r['threshold_screening_mean']:>13.4f}  "
            f"{r['threshold_f1_mean']:>10.4f}"
        )


if __name__ == "__main__":
    main()
