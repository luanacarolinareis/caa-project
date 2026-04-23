"""Dataset and dataloader helpers for chest X-ray classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_SPLITS = ("train", "val", "test")


def get_transforms(config: dict[str, Any], split: str) -> transforms.Compose:
    """Build image transforms for a dataset split."""
    if split not in EXPECTED_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {EXPECTED_SPLITS}.")

    image_size = config["data"]["image_size"]
    mean = config["preprocessing"]["normalize_mean"]
    std = config["preprocessing"]["normalize_std"]
    augmentation_config = config["preprocessing"].get("augmentation", {})

    transform_steps: list[Any] = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
    ]

    if split == "train" and augmentation_config.get("enabled", False):
        if augmentation_config.get("horizontal_flip", False):
            transform_steps.append(transforms.RandomHorizontalFlip())

        max_rotation_degrees = augmentation_config.get("max_rotation_degrees", 0)
        if max_rotation_degrees:
            transform_steps.append(transforms.RandomRotation(max_rotation_degrees))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transform_steps)


def get_datasets(config: dict[str, Any]) -> dict[str, datasets.ImageFolder]:
    """Create ImageFolder datasets for train, validation, and test splits."""
    data_config = config["data"]
    dataset_root = Path(data_config["root_dir"])
    class_names = data_config["class_names"]
    split_dirs = {
        "train": data_config.get("train_dir", "train"),
        "val": data_config.get("val_dir", "val"),
        "test": data_config.get("test_dir", "test"),
    }

    _validate_dataset_structure(dataset_root, split_dirs, class_names)

    return {
        split: datasets.ImageFolder(
            root=dataset_root / split_dir,
            transform=get_transforms(config, split),
        )
        for split, split_dir in split_dirs.items()
    }


def get_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Create dataloaders for train, validation, and test splits."""
    dataset_map = get_datasets(config)
    batch_size = config["training"]["batch_size"]

    return {
        "train": DataLoader(
            dataset_map["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        "val": DataLoader(
            dataset_map["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        "test": DataLoader(
            dataset_map["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    }


def get_class_counts(dataset_root: str | Path) -> dict[str, dict[str, int]]:
    """Count image files per class for each dataset split."""
    root = Path(dataset_root)
    counts: dict[str, dict[str, int]] = {}

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Download the Kaggle dataset and place it there."
        )

    for split in EXPECTED_SPLITS:
        split_path = root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset split directory: {split_path}")

        counts[split] = {}
        for class_path in sorted(path for path in split_path.iterdir() if path.is_dir()):
            counts[split][class_path.name] = _count_images(class_path)

    return counts


def _validate_dataset_structure(
    dataset_root: Path,
    split_dirs: dict[str, str],
    class_names: list[str],
) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. Download the Kaggle dataset and place it there."
        )

    for split, split_dir in split_dirs.items():
        split_path = dataset_root / split_dir
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split} directory: {split_path}")

        for class_name in class_names:
            class_path = split_path / class_name
            if not class_path.exists():
                raise FileNotFoundError(f"Missing {class_name} directory: {class_path}")


def _count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
