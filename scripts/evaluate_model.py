"""Evaluate a trained checkpoint on the test split.

Usage:
    python scripts/evaluate_model.py --model resnet18 --checkpoint results/checkpoints/resnet18_seed0.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config
from pneumonia_classifier.data import get_dataloaders
from pneumonia_classifier.evaluate import evaluate_model
from pneumonia_classifier.models.custom_cnn import build_custom_cnn
from pneumonia_classifier.models.transfer_learning import build_model
from pneumonia_classifier.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
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
    config = merge_config(load_config(args.config), {"training": {"seed": args.seed}})
    set_seed(args.seed)
    device = get_device()

    model = (
        build_custom_cnn()
        if args.model == "custom_cnn"
        else build_model(args.model, pretrained=False)
    )
    load_checkpoint(model, args.checkpoint, device)

    dataloaders = get_dataloaders(config)
    output_path = Path(config["outputs"]["metrics_dir"]) / f"{args.model}_seed{args.seed}.json"
    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        config=config,
        device=device,
        model_name=args.model,
        seed=args.seed,
        output_path=output_path,
    )
    print(f"Saved metrics: {output_path}")
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"auroc={metrics['auroc']}"
    )
    print(f"inference_time_ms_per_image={metrics['inference_time_ms_per_image']:.3f}")


if __name__ == "__main__":
    main()
