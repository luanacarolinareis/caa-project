"""Generate Grad-CAM overlays for selected test images.

Usage:
    python scripts/generate_gradcam.py \\
        --config configs/default.yaml \\
        --model densenet121 \\
        --checkpoint results/checkpoints/densenet121_seed0.pt

Outputs are saved to results/gradcam/.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import load_config
from pneumonia_classifier.models.transfer_learning import (build_model, get_gradcam_target_layer,)
from pneumonia_classifier.visualization.gradcam import (generate_gradcam, load_image,)


def _pick_images(data_root: Path, model: torch.nn.Module, device: torch.device, image_size: int,) -> dict[str, Path]:
    """Find one TP, TN, FP, FN example from the test set."""
    results: dict[str, Path] = {}
    label_dirs = {"NORMAL": 0, "PNEUMONIA": 1}

    candidates: list[tuple[Path, int]] = []
    for class_name, label in label_dirs.items():
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                candidates.append((p, label))

    model.eval()
    for img_path, true_label in candidates:
        if len(results) == 4:
            break
        tensor, _ = load_image(img_path, image_size=image_size)
        tensor = tensor.to(device)
        with torch.no_grad():
            pred = int(model(tensor.unsqueeze(0)).squeeze() > 0)

        if pred == 1 and true_label == 1 and "tp" not in results:
            results["tp"] = img_path
        elif pred == 0 and true_label == 0 and "tn" not in results:
            results["tn"] = img_path
        elif pred == 1 and true_label == 0 and "fp" not in results:
            results["fp"] = img_path
        elif pred == 0 and true_label == 1 and "fn" not in results:
            results["fn"] = img_path

    return results


def _output_filename(model_name: str, category: str, img_path: Path) -> str:
    stem = img_path.stem[:20]
    label_map = {
        "tp": f"{model_name}_tp_pneumonia_{stem}.png",
        "tn": f"{model_name}_tn_normal_{stem}.png",
        "fp": f"{model_name}_fp_{stem}.png",
        "fn": f"{model_name}_fn_{stem}.png",
    }
    return label_map[category]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--model", default="densenet121", choices=["densenet121", "resnet18"])
    parser.add_argument( "--checkpoint", default=None, help="Path to .pt checkpoint. Defaults to results/checkpoints/<model>_seed0.pt",)
    parser.add_argument("--image-path", default=None, help="Single image to process. If omitted, auto-selects TP/TN/FP/FN examples.",)
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--alpha", type=float, default=0.4, help="Heatmap overlay alpha")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size: int = cfg["data"]["image_size"]
    data_root = Path(cfg["data"]["root_dir"]) / cfg["data"]["test_dir"]
    output_dir = Path(args.output_dir or cfg["outputs"]["gradcam_dir"])

    checkpoint_path = Path(args.checkpoint or f"results/checkpoints/{args.model}_seed0.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model)

    if args.image_path:
        img_path = Path(args.image_path)
        out_name = f"{args.model}_{img_path.stem}.png"
        saved, pred = generate_gradcam(model, target_layer, img_path, output_dir / out_name, image_size=image_size, alpha=args.alpha,)
        print(f"Saved: {saved}  (predicted={'PNEUMONIA' if pred else 'NORMAL'})")
        return

    examples = _pick_images(data_root, model, device, image_size)
    if not examples:
        print("No test images found — check data.root_dir in config.")
        sys.exit(1)

    for category, img_path in examples.items():
        out_name = _output_filename(args.model, category, img_path)
        saved, pred = generate_gradcam(model, target_layer, img_path, output_dir / out_name, image_size=image_size, alpha=args.alpha,
        )
        print(f"[{category.upper()}] Saved: {saved}  (predicted={'PNEUMONIA' if pred else 'NORMAL'})")


if __name__ == "__main__":
    main()
