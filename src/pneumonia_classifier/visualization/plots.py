# python src/pneumonia_classifier/visualization/plots.py
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path("results/figures")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(counts: dict[str, dict[str, int]], output_dir: Path = FIGURES_DIR,) -> Path:
    """Bar chart of class counts per split.

    counts format: {"train": {"NORMAL": 1341, "PNEUMONIA": 3875}, ...}
    """
    df = pd.DataFrame(counts).T
    splits = list(df.index)
    classes = list(df.columns)
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})

    # left panel: train + test (main scale)
    main_splits = [s for s in splits if s != "val"]
    ax = axes[0]
    for i, cls in enumerate(classes):
        ax.bar(
            np.arange(len(main_splits)) + i * width - width / 2,
            df.loc[main_splits, cls],
            width,
            label=cls,
        )
    ax.set_title("Class distribution:train & test")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of images")
    ax.set_xticks(np.arange(len(main_splits)))
    ax.set_xticklabels(main_splits)
    ax.legend(title="Class")
    for container in ax.containers:
        ax.bar_label(container, padding=3)

    # right panel: val (own scale so the tiny bars are visible)
    ax2 = axes[1]
    for i, cls in enumerate(classes):
        ax2.bar(i * width, df.loc["val", cls], width, label=cls)
    ax2.set_title("Validation split")
    ax2.set_xlabel("")
    ax2.set_ylabel("Number of images")
    ax2.set_xticks([])
    ax2.legend(title="Class")
    for container in ax2.containers:
        ax2.bar_label(container, padding=3)

    plt.suptitle("Class distribution by split", fontsize=13)
    plt.tight_layout()
    return _save(fig, Path(output_dir) / "class_distribution.png")


def plot_training_curves(history: dict[str, list[float]], model_name: str, output_dir: Path = FIGURES_DIR,) -> Path:
    """Loss and accuracy curves from training history.

    history format: {"train_loss": [...], "val_loss": [...],
                     "train_acc": [...],  "val_acc":  [...]}
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title(f"{model_name}: loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title(f"{model_name}: accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    plt.tight_layout()
    return _save(fig, Path(output_dir) / f"{model_name}_training_curves.png")


def plot_confusion_matrix(cm: np.ndarray, model_name: str, class_names: list[str] | None = None,  output_dir: Path = FIGURES_DIR,) -> Path:
    """Heatmap of a confusion matrix (supports any NxN size)."""
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{model_name}: confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return _save(fig, Path(output_dir) / f"{model_name}_confusion_matrix.png")


def plot_metric_comparison(results: dict[str, dict[str, float]], metrics: list[str] | None = None, output_dir: Path = FIGURES_DIR,) -> Path:
    """Grouped bar chart comparing metrics across models.

    results format: {"densenet121": {"f1": 0.95, "auroc": 0.97, ...}, ...}
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
    df = pd.DataFrame(results).T[metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_title("Metric comparison across models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    # Lower bound adapts to the data so low-valued metrics (e.g. specificity) stay visible instead of being clipped.
    y_min = float(df.to_numpy().min())
    ax.set_ylim(max(0.0, min(0.7, y_min - 0.05)), 1.02)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    return _save(fig, Path(output_dir) / "metric_comparison.png")


if __name__ == "__main__":
    out = Path("results/_demo")
    print(f"[smoke test] writing example figures to {out} (synthetic data)")

    # class distribution
    p = plot_class_distribution(
        {"train": {"NORMAL": 1341, "PNEUMONIA": 3875},
         "val": {"NORMAL": 8, "PNEUMONIA": 8},
         "test": {"NORMAL": 234, "PNEUMONIA": 390}},
        output_dir=out,
    )
    print(f"Saved: {p}")

    # training curves
    n = 10
    p = plot_training_curves(
        {"train_loss": list(np.linspace(0.6, 0.2, n)),
         "val_loss": list(np.linspace(0.65, 0.25, n)),
         "train_acc": list(np.linspace(0.6, 0.93, n)),
         "val_acc": list(np.linspace(0.58, 0.90, n))},
        model_name="example",
        output_dir=out,
    )
    print(f"Saved: {p}")

    # confusion matrix
    p = plot_confusion_matrix(
        np.array([[210, 24], [18, 372]]),
        model_name="example",
        output_dir=out,
    )
    print(f"Saved: {p}")

    # metric comparison
    p = plot_metric_comparison(
        {"custom_cnn": {"accuracy": 0.82, "precision": 0.83, "recall": 0.88, "f1": 0.85, "auroc": 0.88},
         "resnet18": {"accuracy": 0.91, "precision": 0.90, "recall": 0.94, "f1": 0.92, "auroc": 0.96},
         "densenet121": {"accuracy": 0.93, "precision": 0.92, "recall": 0.96, "f1": 0.94, "auroc": 0.97}},
        output_dir=out,
    )
    print(f"Saved: {p}")
