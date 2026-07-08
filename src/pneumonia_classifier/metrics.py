"""Metrics for pneumonia classification (binary and multi-class)."""

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auroc"] = None
    return metrics


def compute_multiclass_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[list[float]] | np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute metrics for multi-class classification."""
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    num_classes = len(set(labels) | set(predictions))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }

    # Per-class metrics
    per_class_precision = precision_score(
        labels, predictions, average=None, zero_division=0, labels=list(range(len(class_names)))
    )
    per_class_recall = recall_score(
        labels, predictions, average=None, zero_division=0, labels=list(range(len(class_names)))
    )
    per_class_f1 = f1_score(
        labels, predictions, average=None, zero_division=0, labels=list(range(len(class_names)))
    )
    metrics["precision_per_class"] = {
        name: float(val) for name, val in zip(class_names, per_class_precision)
    }
    metrics["recall_per_class"] = {
        name: float(val) for name, val in zip(class_names, per_class_recall)
    }
    metrics["f1_per_class"] = {
        name: float(val) for name, val in zip(class_names, per_class_f1)
    }
    metrics["confusion_matrix"] = cm.tolist()

    # AUROC (one-vs-rest)
    if y_prob is not None:
        try:
            prob_array = np.asarray(y_prob, dtype=float)
            metrics["auroc"] = float(
                roc_auc_score(labels, prob_array, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auroc"] = None
    else:
        metrics["auroc"] = None

    return metrics
