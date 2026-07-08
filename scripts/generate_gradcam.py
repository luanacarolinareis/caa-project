"""Generate Grad-CAM overlays for selected test images.

Supports both binary (2-class) and three-class classification.

Usage (binary):
    python scripts/generate_gradcam.py \\
        --config configs/default.yaml \\
        --model densenet121 \\
        --checkpoint results/checkpoints/densenet121_seed0.pt

Usage (three-class):
    python scripts/generate_gradcam.py \\
        --config configs/three_class.yaml \\
        --model densenet121 \\
        --checkpoint results/checkpoints/densenet121_3class_seed0.pt

Outputs are saved to results/gradcam/.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pneumonia_classifier.config import load_config, is_three_class, get_num_classes
from pneumonia_classifier.data import THREE_CLASS_NAMES, THREE_CLASS_MAP, _label_from_filename
from pneumonia_classifier.models.transfer_learning import (build_model, get_gradcam_target_layer,)
from pneumonia_classifier.visualization.gradcam import (generate_gradcam, load_image,)


BINARY_LABEL_DIRS = {"NORMAL": 0, "PNEUMONIA": 1}
BINARY_CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def _pick_images_binary(
    data_root: Path, model: torch.nn.Module, device: torch.device, image_size: int,
) -> dict[str, Path]:
    """Find one TP, TN, FP, FN example from the test set (binary)."""
    results: dict[str, Path] = {}
    candidates: list[tuple[Path, int]] = []
    for class_name, label in BINARY_LABEL_DIRS.items():
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


def _pick_images_three_class(
    data_root: Path, model: torch.nn.Module, device: torch.device, image_size: int,
) -> dict[str, tuple[Path, int]]:
    """Find representative correct and incorrect examples for each class."""
    results: dict[str, tuple[Path, int]] = {}
    candidates: list[tuple[Path, int]] = []

    for class_dir_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = data_root / class_dir_name
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                label = _label_from_filename(p.name, class_dir_name)
                candidates.append((p, label))

    model.eval()
    idx_to_name = {v: k for k, v in THREE_CLASS_MAP.items()}

    for img_path, true_label in candidates:
        if len(results) >= 6:
            break
        tensor, _ = load_image(img_path, image_size=image_size)
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))
            pred = int(logits.argmax(dim=1).item())

        true_name = idx_to_name[true_label]
        if pred == true_label:
            key = f"correct_{true_name}"
        else:
            pred_name = idx_to_name[pred]
            key = f"wrong_{true_name}_as_{pred_name}"

        if key not in results:
            results[key] = (img_path, true_label)

    return results


def _output_filename_binary(model_name: str, category: str, img_path: Path) -> str:
    # The class word is the ACTUAL (true) class of the image:
    #   true_positive  -> model said pneumonia, was pneumonia
    #   true_negative  -> model said normal,    was normal
    #   false_positive -> model said pneumonia, was actually normal
    #   false_negative -> model said normal,    was actually pneumonia
    stem = img_path.stem[:20]
    label_map = {
        "tp": f"{model_name}_true_positive_pneumonia_{stem}.png",
        "tn": f"{model_name}_true_negative_normal_{stem}.png",
        "fp": f"{model_name}_false_positive_normal_{stem}.png",
        "fn": f"{model_name}_false_negative_pneumonia_{stem}.png",
    }
    return label_map[category]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--model", default="densenet121", choices=["densenet121", "resnet18"])
    parser.add_argument( "--checkpoint", default=None, help="Path to .pt checkpoint. Defaults to results/checkpoints/<model>_seed0.pt",)
    parser.add_argument("--image-path", default=None, help="Single image to process. If omitted, auto-selects examples.",)
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--alpha", type=float, default=0.4, help="Heatmap overlay alpha")
    args = parser.parse_args()

    cfg = load_config(args.config)
    three_class = is_three_class(cfg)
    num_classes = get_num_classes(cfg)
    image_size: int = cfg["data"]["image_size"]
    data_root = Path(cfg["data"]["root_dir"]) / cfg["data"]["test_dir"]
    output_dir = Path(args.output_dir or cfg["outputs"]["gradcam_dir"])
    class_names = cfg["data"].get("class_names", BINARY_CLASS_NAMES)

    # Default checkpoint path
    if three_class:
        default_ckpt = f"results/checkpoints/{args.model}_3class_seed0.pt"
    else:
        default_ckpt = f"results/checkpoints/{args.model}_seed0.pt"
    checkpoint_path = Path(args.checkpoint or default_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model)

    # Clean up stale overlays from previous runs of THIS model+task. Because the
    # filename embeds the example image (which changes when the model is
    # retrained), old files are otherwise never overwritten and accumulate,
    # producing duplicate/contradictory examples in the grids. We only do this
    # for the auto-selected mode (not for an explicit --image-path)
    if not args.image_path and output_dir.exists():
        prefix = f"{args.model}_3class_" if three_class else f"{args.model}_"
        for old in output_dir.glob(f"{prefix}*.png"):
            # For the binary task, never delete the 3-class overlays.
            if not three_class and "_3class_" in old.name:
                continue
            old.unlink()

    if args.image_path:
        img_path = Path(args.image_path)
        out_name = f"{args.model}_{img_path.stem}.png"
        saved, pred_idx, pred_name = generate_gradcam(
            model, target_layer, img_path, output_dir / out_name,
            image_size=image_size, alpha=args.alpha, class_names=class_names,
        )
        print(f"Saved: {saved}  (predicted={pred_name})")
        return

    if three_class:
        examples = _pick_images_three_class(data_root, model, device, image_size)
        if not examples:
            print("No test images found — check data.root_dir in config.")
            sys.exit(1)

        for category, (img_path, true_label) in examples.items():
            stem = img_path.stem[:20]
            out_name = f"{args.model}_3class_{category}_{stem}.png"
            saved, pred_idx, pred_name = generate_gradcam(
                model, target_layer, img_path, output_dir / out_name,
                image_size=image_size, alpha=args.alpha, class_names=class_names,
            )
            true_name = THREE_CLASS_NAMES[true_label] if true_label < len(THREE_CLASS_NAMES) else str(true_label)
            print(f"[{category.upper()}] Saved: {saved}  (true={true_name}, predicted={pred_name})")
    else:
        examples = _pick_images_binary(data_root, model, device, image_size)
        if not examples:
            print("No test images found — check data.root_dir in config.")
            sys.exit(1)

        for category, img_path in examples.items():
            out_name = _output_filename_binary(args.model, category, img_path)
            saved, pred_idx, pred_name = generate_gradcam(
                model, target_layer, img_path, output_dir / out_name,
                image_size=image_size, alpha=args.alpha, class_names=class_names,
            )
            print(f"[{category.upper()}] Saved: {saved}  (predicted={pred_name})")


if __name__ == "__main__":
    main()
