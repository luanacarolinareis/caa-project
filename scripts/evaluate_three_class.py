"""Evaluate a trained 3-class checkpoint on the test split.

Usage:
    python scripts/evaluate_three_class.py \
        --model densenet121 \
        --checkpoint results/checkpoints/densenet121_3class_seed0.pt

    python scripts/evaluate_three_class.py \
        --model resnet18 \
        --checkpoint results/checkpoints/resnet18_3class_seed0.pt \
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config, get_num_classes
from pneumonia_classifier.data import get_three_class_dataloaders, THREE_CLASS_NAMES
from pneumonia_classifier.evaluate import evaluate_model
from pneumonia_classifier.models.custom_cnn import build_custom_cnn
from pneumonia_classifier.models.transfer_learning import build_model
from pneumonia_classifier.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a 3-class model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/three_class.yaml"),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom_cnn", "resnet18", "densenet121"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_config(
        load_config(args.config), {"training": {"seed": args.seed}},
    )
    set_seed(args.seed)
    device = get_device()
    num_classes = get_num_classes(config)

    model = (
        build_custom_cnn(num_classes=num_classes)
        if args.model == "custom_cnn"
        else build_model(args.model, pretrained=False, num_classes=num_classes)
    )
    load_checkpoint(model, args.checkpoint, device)

    dataloaders = get_three_class_dataloaders(config)
    output_path = (
        Path(config["outputs"]["metrics_dir"])
        / f"{args.model}_3class_seed{args.seed}.json"
    )
    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        config=config,
        device=device,
        model_name=args.model,
        seed=args.seed,
        output_path=output_path,
    )

    print(f"\nSaved metrics: {output_path}")
    print(f"\n{'='*50}")
    print(f"  3-CLASS EVALUATION — {args.model}")
    print(f"{'='*50}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)     : {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro) : {metrics['recall_macro']:.4f}")
    if metrics.get("auroc") is not None:
        print(f"  AUROC (OvR)    : {metrics['auroc']:.4f}")
    print(f"  Inference      : {metrics['inference_time_ms_per_image']:.3f} ms/image")

    print(f"\n  Per-class F1:")
    for name, val in metrics.get("f1_per_class", {}).items():
        print(f"    {name:12s}: {val:.4f}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    class_names = config["data"].get("class_names", THREE_CLASS_NAMES)
    cm = np.array(metrics["confusion_matrix"])
    header = "            " + "  ".join(f"{c:>10s}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{int(v):>10d}" for v in row)
        print(f"  {class_names[i]:>10s}  {row_str}")


if __name__ == "__main__":
    main()
