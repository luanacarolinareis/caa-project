"""Train a transfer-learning model for 3-class pneumonia classification.

Normal / Bacterial Pneumonia / Viral Pneumonia.

Usage:
    python scripts/train_three_class.py --model densenet121 --seed 0
    python scripts/train_three_class.py --model resnet18 --config configs/three_class.yaml --seed 0
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config, get_num_classes
from pneumonia_classifier.data import get_three_class_dataloaders, compute_class_weights
from pneumonia_classifier.models.transfer_learning import (
    build_model,
    freeze_backbone,
    unfreeze_backbone,
)
from pneumonia_classifier.train import train_model
from pneumonia_classifier.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transfer-learning model for 3-class classification.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/three_class.yaml"),
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

    num_classes = get_num_classes(config)
    transfer_cfg = config["models"]["transfer_learning"]
    pretrained: bool = transfer_cfg["pretrained"]
    head_only_epochs: int = transfer_cfg["head_only_epochs"]
    max_epochs: int = transfer_cfg["max_epochs"]
    checkpoint_dir = Path(config["outputs"]["checkpoints_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task        : 3-class classification (Normal / Bacteria / Virus)")
    print(f"Model       : {args.model}")
    print(f"Config      : {args.config}")
    print(f"Seed        : {args.seed}")
    print(f"Num classes : {num_classes}")
    print(f"Pretrained  : {pretrained}")
    print(f"Head epochs : {head_only_epochs}")
    print(f"Max epochs  : {max_epochs}")
    print(f"Checkpoints : {checkpoint_dir}")

    model = build_model(args.model, pretrained=pretrained, num_classes=num_classes)
    dataloaders = get_three_class_dataloaders(config)
    device = get_device()

    # Print class distribution
    for split_name, dl in dataloaders.items():
        ds = dl.dataset
        if hasattr(ds, "get_class_counts"):
            counts = ds.get_class_counts()
            print(f"  {split_name}: {counts}")

    # Inverse-frequency class weights computed from the (resplit) training set
    class_weights = compute_class_weights(dataloaders["train"].dataset)
    if config["training"].get("use_class_weights", False):
        print(f"Class weights: {[round(w, 3) for w in class_weights]}")
    print(f"Selection   : {config['training'].get('select_by', 'loss')}")

    print(f"\nModel instantiated: {args.model} (num_classes={num_classes})")
    print(f"Device      : {device}")

    # Phase 1: head-only training
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 -> head only, trainable params: {trainable:,}")

    checkpoint_path = checkpoint_dir / f"{args.model}_3class_seed{args.seed}.pt"
    head_history_path = Path(config["outputs"]["metrics_dir"]) / (
        f"{args.model}_3class_seed{args.seed}_head_only_history.json"
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
        class_weights=class_weights,
    )

    # Phase 2: full fine-tuning
    load_checkpoint(model, checkpoint_path, device)
    unfreeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 -> full fine-tune, trainable params: {trainable:,}")

    fine_history_path = Path(config["outputs"]["metrics_dir"]) / (
        f"{args.model}_3class_seed{args.seed}_fine_tune_history.json"
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
        class_weights=class_weights,
    )

    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved histories: {head_history_path}, {fine_history_path}")


if __name__ == "__main__":
    main()
