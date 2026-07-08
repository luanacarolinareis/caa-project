"""Train Stage B of the hierarchical approach: BACTERIA vs VIRUS.

This classifier is trained ONLY on pneumonia images and distinguishes
bacterial from viral pneumonia -- the hardest distinction in the dataset and
the one Kermany et al. tackled with a dedicated binary classifier (rather than
a flat 3-way softmax). Stage A (pneumonia vs normal) reuses the existing
binary checkpoints; see scripts/evaluate_hierarchical.py.

It reuses the same training engine, class-weighting and F1-macro checkpoint
selection as the 3-class task, and the same patient-aware validation split.

Usage:
    python scripts/train_stage_b.py --model densenet121 --seed 0
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config, get_num_classes
from pneumonia_classifier.data import (
    get_bacteria_virus_dataloaders,
    compute_class_weights_two,
)
from pneumonia_classifier.models.transfer_learning import (
    build_model,
    freeze_backbone,
    unfreeze_backbone,
)
from pneumonia_classifier.train import train_model
from pneumonia_classifier.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage B (bacteria vs virus) of the hierarchical model.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/bacteria_virus.yaml"))
    parser.add_argument("--model", type=str, choices=["densenet121", "resnet18"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_config(load_config(args.config), {"training": {"seed": args.seed}})
    set_seed(args.seed)

    num_classes = get_num_classes(config)  # 2 for bacteria/virus
    transfer_cfg = config["models"]["transfer_learning"]
    pretrained = transfer_cfg["pretrained"]
    head_only_epochs = transfer_cfg["head_only_epochs"]
    max_epochs = transfer_cfg["max_epochs"]
    checkpoint_dir = Path(config["outputs"]["checkpoints_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config["outputs"]["metrics_dir"])

    print("Task        : Stage B - bacteria vs virus (pneumonia images only)")
    print(f"Model       : {args.model}")
    print(f"Config      : {args.config}")
    print(f"Seed        : {args.seed}")
    print(f"Num classes : {num_classes}")

    model = build_model(args.model, pretrained=pretrained, num_classes=num_classes)
    dataloaders = get_bacteria_virus_dataloaders(config)
    device = get_device()

    for split_name, dl in dataloaders.items():
        ds = dl.dataset
        if hasattr(ds, "get_class_counts"):
            print(f"  {split_name}: {ds.get_class_counts()}")

    class_weights = compute_class_weights_two(dataloaders["train"].dataset)
    if config["training"].get("use_class_weights", False):
        print(f"Class weights: {[round(w, 3) for w in class_weights]}")
    print(f"Selection   : {config['training'].get('select_by', 'loss')}")
    print(f"Device      : {device}")

    # Phase 1: head-only
    freeze_backbone(model)
    checkpoint_path = checkpoint_dir / f"{args.model}_bactvirus_seed{args.seed}.pt"
    head_history_path = metrics_dir / f"{args.model}_bactvirus_seed{args.seed}_head_only_history.json"
    train_model(
        model=model, dataloaders=dataloaders, config=config, device=device,
        model_name=args.model, seed=args.seed, max_epochs=head_only_epochs,
        checkpoint_path=checkpoint_path, history_path=head_history_path,
        phase="head_only", class_weights=class_weights,
    )

    # Phase 2: full fine-tune
    load_checkpoint(model, checkpoint_path, device)
    unfreeze_backbone(model)
    fine_history_path = metrics_dir / f"{args.model}_bactvirus_seed{args.seed}_fine_tune_history.json"
    train_model(
        model=model, dataloaders=dataloaders, config=config, device=device,
        model_name=args.model, seed=args.seed, max_epochs=max_epochs,
        checkpoint_path=checkpoint_path, history_path=fine_history_path,
        phase="fine_tune", class_weights=class_weights,
    )

    print(f"\nSaved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
