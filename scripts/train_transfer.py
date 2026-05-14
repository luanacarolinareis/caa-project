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
from pneumonia_classifier.data import get_dataloaders
from pneumonia_classifier.models.transfer_learning import (
    build_model,
    freeze_backbone,
    unfreeze_backbone,
)
from pneumonia_classifier.train import train_model
from pneumonia_classifier.utils import get_device, load_checkpoint, set_seed

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
    set_seed(args.seed)

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
    dataloaders = get_dataloaders(config)
    device = get_device()
    print(f"\nModel instantiated: {args.model}")
    print(f"Device      : {device}")

    # Phase 1: head-only training
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 -> head only, trainable params: {trainable:,}")

    checkpoint_path = checkpoint_dir / f"{args.model}_seed{args.seed}.pt"
    head_history_path = Path(config["outputs"]["metrics_dir"]) / (
        f"{args.model}_seed{args.seed}_head_only_history.json"
    )
    train_model(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        model_name=args.model,
        seed=args.seed,
        max_epochs=head_only_epochs,
        checkpoint_path=checkpoint_path,
        history_path=head_history_path,
        phase="head_only",
    )

    # Phase 2: full fine-tuning
    load_checkpoint(model, checkpoint_path, device)
    unfreeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 -> full fine-tune, trainable params: {trainable:,}")

    fine_history_path = Path(config["outputs"]["metrics_dir"]) / (
        f"{args.model}_seed{args.seed}_fine_tune_history.json"
    )
    train_model(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        model_name=args.model,
        seed=args.seed,
        max_epochs=max_epochs,
        checkpoint_path=checkpoint_path,
        history_path=fine_history_path,
        phase="fine_tune",
    )

    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved histories: {head_history_path}, {fine_history_path}")


if __name__ == "__main__":
    main()
