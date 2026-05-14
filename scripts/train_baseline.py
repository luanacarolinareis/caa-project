"""Train the custom CNN baseline.

Usage:
    python scripts/train_baseline.py --config configs/default.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config
from pneumonia_classifier.data import get_dataloaders
from pneumonia_classifier.models.custom_cnn import build_custom_cnn
from pneumonia_classifier.train import train_model
from pneumonia_classifier.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom CNN baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_config(load_config(args.config), {"training": {"seed": args.seed}})
    set_seed(args.seed)
    device = get_device()
    model_name = config["models"]["baseline"]["name"]
    max_epochs = config["models"]["baseline"]["max_epochs"]

    dataloaders = get_dataloaders(config)
    model = build_custom_cnn()
    checkpoint_path = Path(config["outputs"]["checkpoints_dir"]) / f"{model_name}_seed{args.seed}.pt"
    history_path = Path(config["outputs"]["metrics_dir"]) / f"{model_name}_seed{args.seed}_history.json"

    print(f"Training {model_name} on {device} with seed {args.seed}")
    train_model(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        model_name=model_name,
        seed=args.seed,
        max_epochs=max_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        phase="baseline",
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
