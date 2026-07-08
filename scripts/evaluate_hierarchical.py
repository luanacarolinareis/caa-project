"""Evaluate the hierarchical (two-stage) approach against the flat 3-class model.

Pipeline at inference time, for each test image:
  Stage A (NORMAL vs PNEUMONIA): the existing binary checkpoint.
      - if it predicts NORMAL                 -> final label NORMAL
      - if it predicts PNEUMONIA -> Stage B    -> BACTERIA or VIRUS
  Stage B (BACTERIA vs VIRUS): the checkpoint trained by train_stage_b.py.

This mirrors Kermany et al. (a dedicated bacterial-vs-viral classifier rather
than a flat 3-way softmax) and produces the SAME 3-class metrics as
evaluate_three_class.py so the two approaches are directly comparable.

Three-class label convention (matches data.THREE_CLASS_MAP):
    0 = NORMAL, 1 = BACTERIA, 2 = VIRUS

Usage:
    python scripts/evaluate_hierarchical.py --model densenet121 --seed 0
    python scripts/evaluate_hierarchical.py --model densenet121 --seed 0 \
        --stage-a results/checkpoints/densenet121_seed2.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import load_config, merge_config
from pneumonia_classifier.data import (ThreeClassDataset, THREE_CLASS_NAMES, get_transforms)
from pneumonia_classifier.metrics import compute_multiclass_metrics
from pneumonia_classifier.models.transfer_learning import build_model
from pneumonia_classifier.utils import get_device, save_json
from torch.utils.data import DataLoader


def _load(model_name: str, num_classes: int, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the hierarchical two-stage model.")
    p.add_argument("--config", type=Path, default=Path("configs/three_class.yaml"))
    p.add_argument("--model", type=str, choices=["densenet121", "resnet18"], default="densenet121")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stage-a",
        type=Path,
        default=None,
        help="Stage A (binary NORMAL/PNEUMONIA) checkpoint. "
        "Defaults to results/checkpoints/<model>_seed<seed>.pt",
    )
    p.add_argument(
        "--stage-b",
        type=Path,
        default=None,
        help="Stage B (bacteria/virus) checkpoint. "
        "Defaults to results/checkpoints/<model>_bactvirus_seed<seed>.pt",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Stage A pneumonia threshold.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_config(load_config(args.config), {"training": {"seed": args.seed}})
    device = get_device()
    image_size = config["data"]["image_size"]

    ckpt_dir = Path(config["outputs"]["checkpoints_dir"])
    stage_a_path = args.stage_a or ckpt_dir / f"{args.model}_seed{args.seed}.pt"
    stage_b_path = args.stage_b or ckpt_dir / f"{args.model}_bactvirus_seed{args.seed}.pt"
    for label, path in [("Stage A", stage_a_path), ("Stage B", stage_b_path)]:
        if not path.exists():
            print(f"ERROR: {label} checkpoint not found: {path}")
            sys.exit(1)

    print("Hierarchical evaluation (two-stage)")
    print(f"  Model    : {args.model}")
    print(f"  Stage A  : {stage_a_path}  (NORMAL vs PNEUMONIA, binary)")
    print(f"  Stage B  : {stage_b_path}  (BACTERIA vs VIRUS)")

    stage_a = _load(args.model, 1, stage_a_path, device)   # binary BCE head
    stage_b = _load(args.model, 2, stage_b_path, device)   # 2-way softmax

    # Test set with three-class labels (NORMAL=0, BACTERIA=1, VIRUS=2)
    data_root = Path(config["data"]["root_dir"])
    test_ds = ThreeClassDataset(
        root=data_root / config["data"].get("test_dir", "test"),
        transform=get_transforms(config, "test"),
    )
    loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []
    # Pseudo 3-class probabilities for AUROC: combine the two stages
    #   P(normal)   = 1 - p_pneumonia
    #   P(bacteria) = p_pneumonia * P_B(bacteria)
    #   P(virus)    = p_pneumonia * P_B(virus)
    y_prob: list[list[float]] = []

    num_images = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            p_pneumonia = torch.sigmoid(stage_a(images)).squeeze(1)        # (B,)
            pb = torch.softmax(stage_b(images), dim=1)                     # (B, 2): [bacteria, virus]

            p_normal = 1.0 - p_pneumonia
            p_bacteria = p_pneumonia * pb[:, 0]
            p_virus = p_pneumonia * pb[:, 1]
            probs = torch.stack([p_normal, p_bacteria, p_virus], dim=1)    # (B, 3)

            is_pneumonia = p_pneumonia >= args.threshold
            # default NORMAL (0); if pneumonia, pick bacteria(1)/virus(2) from Stage B
            preds = torch.zeros_like(p_pneumonia, dtype=torch.long)
            stage_b_pred = pb.argmax(dim=1) + 1  # 1=bacteria, 2=virus
            preds = torch.where(is_pneumonia, stage_b_pred, preds)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
            num_images += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_image = (time.perf_counter() - start) / max(num_images, 1) * 1000

    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob=y_prob, class_names=THREE_CLASS_NAMES)
    metrics.update(
        {
            "model": f"{args.model}_hierarchical",
            "seed": args.seed,
            "approach": "hierarchical",
            "stage_a_checkpoint": str(stage_a_path),
            "stage_b_checkpoint": str(stage_b_path),
            "num_test_images": len(y_true),
            "inference_time_ms_per_image": ms_per_image,
        }
    )

    out_path = Path(config["outputs"]["metrics_dir"]) / f"{args.model}_hierarchical_seed{args.seed}.json"
    save_json(metrics, out_path)
    save_json(
        {"y_true": y_true, "y_prob": y_prob, "model": f"{args.model}_hierarchical", "seed": args.seed},
        out_path.with_name(out_path.stem + "_probs.json"),
    )

    print(f"\nSaved metrics: {out_path}")
    print(f"\n{'='*52}")
    print(f"  HIERARCHICAL 3-CLASS EVALUATION - {args.model}")
    print(f"{'='*52}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)      : {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)  : {metrics['recall_macro']:.4f}")
    if metrics.get("auroc") is not None:
        print(f"  AUROC (OvR)     : {metrics['auroc']:.4f}")
    print("\n  Per-class F1:")
    for name, val in metrics["f1_per_class"].items():
        print(f"    {name:10s}: {val:.4f}")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    cm = np.array(metrics["confusion_matrix"])
    print("            " + "  ".join(f"{c:>10s}" for c in THREE_CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {THREE_CLASS_NAMES[i]:>10s}  " + "  ".join(f"{int(v):>10d}" for v in row))


if __name__ == "__main__":
    main()
