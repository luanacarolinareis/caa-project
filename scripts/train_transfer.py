"""Entry point for transfer-learning training (DenseNet121 / ResNet18).

Usage:
    python scripts/train_transfer.py --model densenet121 --config configs/default.yaml --seed 0
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config
from pneumonia_classifier.models.transfer_learning import (build_model, freeze_backbone, unfreeze_backbone,)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transfer-learning model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["densenet121", "resnet18"],
        required=True,
        help="Model name to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(config, {"training": {"seed": args.seed}})

    transfer_cfg = config["models"]["transfer_learning"]
    pretrained: bool = transfer_cfg["pretrained"]
    head_only_epochs: int = transfer_cfg["head_only_epochs"]
    max_epochs: int = transfer_cfg["max_epochs"]
    checkpoint_dir = Path(config["outputs"]["checkpoints_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model       : {args.model}")
    print(f"Config      : {args.config}")
    print(f"Seed        : {args.seed}")
    print(f"Pretrained  : {pretrained}")
    print(f"Head epochs : {head_only_epochs}")
    print(f"Max epochs  : {max_epochs}")
    print(f"Checkpoints : {checkpoint_dir}")

    model = build_model(args.model, pretrained=pretrained)
    print(f"\nModel instantiated: {args.model}")

    # Phase 1: head-only training
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 -> head only, trainable params: {trainable:,}")

    # TODO (Jakub -> data.py)
    #   dataloaders = get_dataloaders(config, seed=args.seed)
    # TODO (Jakub -> train.py)
    #   train(model, dataloaders, config, epochs=head_only_epochs, phase="head_only")

    # Phase 2: full fine-tuning
    unfreeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 -> full fine-tune, trainable params: {trainable:,}")

    # TODO (Jakub -> train.py)
    #   train(model, dataloaders, config, epochs=max_epochs, phase="fine_tune")

    # TODO (Jakub -> train.py/utils.py)
    checkpoint_path = checkpoint_dir / f"{args.model}_seed{args.seed}.pt"
    print(f"\nCheckpoint will be saved to: {checkpoint_path}")
    print("(Training loop not yet connected!)")


if __name__ == "__main__":
    main()
