"""Ensemble of ResNet18 + DenseNet121 via probability averaging.

Loads all per-seed checkpoints for both transfer-learning models, averages
their softmax/sigmoid outputs, and evaluates the ensemble on the test set.
Results are saved alongside individual-model metrics so make_result_tables.py
can include them in the summary table.

Usage:
    python innovation/ensemble.py
    python innovation/ensemble.py --config configs/three_class.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import get_num_classes, is_three_class, load_config
from pneumonia_classifier.data import get_dataloaders, get_three_class_dataloaders
from pneumonia_classifier.evaluate import predict_probabilities
from pneumonia_classifier.metrics import compute_binary_metrics, compute_multiclass_metrics
from pneumonia_classifier.models.transfer_learning import build_model
from pneumonia_classifier.utils import get_device, load_checkpoint, save_json


ENSEMBLE_MODELS = ["resnet18", "densenet121"]
SEEDS = [0, 1, 2]


def _load_model(model_name: str, checkpoint_path: Path, num_classes: int, device: torch.device,
) -> torch.nn.Module:
    model = build_model(model_name, pretrained=False, num_classes=num_classes)
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()
    return model


def _checkpoint_path(model_name: str, seed: int, three_class: bool, checkpoints_dir: Path) -> Path:
    suffix = "_3class" if three_class else ""
    return checkpoints_dir / f"{model_name}{suffix}_seed{seed}.pt"


def run_ensemble(config: dict, device: torch.device) -> dict:
    three_class = is_three_class(config)
    num_classes = get_num_classes(config)
    checkpoints_dir = Path(config["outputs"]["checkpoints_dir"])
    class_names = config["data"].get("class_names", [])

    dataloaders = (get_three_class_dataloaders(config) if three_class else get_dataloaders(config))
    test_loader = dataloaders["test"]

    # Collect probability arrays from every checkpoint
    all_probs: list[np.ndarray] = []
    y_true: list[int] | None = None

    for model_name in ENSEMBLE_MODELS:
        for seed in SEEDS:
            ckpt = _checkpoint_path(model_name, seed, three_class, checkpoints_dir)
            if not ckpt.exists():
                print(f"WARNING: checkpoint not found, skipping {ckpt}")
                continue

            model = _load_model(model_name, ckpt, num_classes, device)
            labels, probs, _ = predict_probabilities(model, test_loader, device, three_class)

            if y_true is None:
                y_true = labels
            all_probs.append(np.asarray(probs, dtype=float))
            print(f"  loaded {model_name} seed={seed}")

    if not all_probs or y_true is None:
        raise RuntimeError("No checkpoints found. Train the models before running the ensemble.")

    # Average probabilities across all members
    avg_probs = np.mean(all_probs, axis=0)

    if three_class:
        y_pred = avg_probs.argmax(axis=1).tolist()
        metrics = compute_multiclass_metrics(y_true, y_pred, y_prob=avg_probs.tolist(), class_names=class_names,)
    else:
        threshold = config["evaluation"].get("threshold", 0.5)
        metrics = compute_binary_metrics(y_true, avg_probs.tolist(), threshold=threshold)

    metrics.update({
        "model": "ensemble_resnet18_densenet121",
        "members": [f"{m}_seed{s}" for m in ENSEMBLE_MODELS for s in SEEDS],
        "num_members": len(all_probs),
        "num_test_images": len(y_true),
    })
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble ResNet18 + DenseNet121.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config (default or three_class).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    three_class = is_three_class(config)
    suffix = "_3class" if three_class else ""

    print(f"Config : {args.config}")
    print(f"Task   : {'3-class' if three_class else 'binary'}")
    print(f"Device : {device}")
    print(f"Members: {ENSEMBLE_MODELS} × seeds {SEEDS}\n")

    metrics = run_ensemble(config, device)

    output_path = (Path(config["outputs"]["metrics_dir"]) / f"ensemble_resnet18_densenet121{suffix}.json")
    save_json(metrics, output_path)

    print(f"\nSaved: {output_path}")
    print(f"\n{'='*55}")
    print(f"  ENSEMBLE ({len(metrics['members'])} members)")
    print(f"{'='*55}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    if three_class:
        print(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
        print(f"  AUROC (OvR): {metrics.get('auroc') or 'n/a'}")
        print(f"\n  Per-class F1:")
        for name, val in metrics.get("f1_per_class", {}).items():
            print(f"    {name:12s}: {val:.4f}")
    else:
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1         : {metrics['f1']:.4f}")
        print(f"  AUROC      : {metrics.get('auroc') or 'n/a'}")


if __name__ == "__main__":
    main()
