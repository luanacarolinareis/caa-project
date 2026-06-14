"""Dataset and dataloader helpers for chest X-ray classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import random

import torch
from torch.utils.data import DataLoader, Dataset
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


# Three-class dataset: Normal / Bacteria / Virus

THREE_CLASS_NAMES = ["NORMAL", "BACTERIA", "VIRUS"]
THREE_CLASS_MAP = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}


def _label_from_filename(filename: str, parent_dir: str) -> int:
    """Derive the three-class label from the image filename.

    - Images in the NORMAL directory -> 0
    - Filenames containing '_bacteria_' -> 1
    - Filenames containing '_virus_' -> 2
    """
    if parent_dir.upper() == "NORMAL":
        return THREE_CLASS_MAP["NORMAL"]
    name_lower = filename.lower()
    if "_bacteria_" in name_lower or name_lower.startswith("bacteria"):
        return THREE_CLASS_MAP["BACTERIA"]
    if "_virus_" in name_lower or name_lower.startswith("virus"):
        return THREE_CLASS_MAP["VIRUS"]
    # Fallback: treat ambiguous pneumonia images as bacteria (most common)
    return THREE_CLASS_MAP["BACTERIA"]


class ThreeClassDataset(Dataset):
    """Dataset that reads the standard NORMAL/PNEUMONIA directory layout
    and produces three-class labels by parsing PNEUMONIA filenames."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._scan()

    def _scan(self) -> None:
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            parent_name = class_dir.name  # "NORMAL" or "PNEUMONIA"
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    label = _label_from_filename(img_path.name, parent_name)
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict[str, int]:
        """Return per-class image counts."""
        counts = {name: 0 for name in THREE_CLASS_NAMES}
        idx_to_name = {v: k for k, v in THREE_CLASS_MAP.items()}
        for _, label in self.samples:
            counts[idx_to_name[label]] += 1
        return counts


def get_three_class_datasets(
    config: dict[str, Any],
) -> dict[str, ThreeClassDataset]:
    """Create ThreeClassDataset instances for train, val, and test splits.

    If the validation split has fewer than 3 classes represented, a small
    balanced validation set is carved from the training data.
    """
    data_config = config["data"]
    dataset_root = Path(data_config["root_dir"])
    split_dirs = {
        "train": data_config.get("train_dir", "train"),
        "val": data_config.get("val_dir", "val"),
        "test": data_config.get("test_dir", "test"),
    }

    train_ds = ThreeClassDataset(
        root=dataset_root / split_dirs["train"],
        transform=get_transforms(config, "train"),
    )
    val_ds = ThreeClassDataset(
        root=dataset_root / split_dirs["val"],
        transform=get_transforms(config, "val"),
    )
    test_ds = ThreeClassDataset(
        root=dataset_root / split_dirs["test"],
        transform=get_transforms(config, "test"),
    )

    # Fix validation split if it is missing classes
    val_counts = val_ds.get_class_counts()
    missing_classes = [name for name, cnt in val_counts.items() if cnt == 0]
    if missing_classes:
        print(
            f"[data] Validation split is missing classes {missing_classes}. "
            f"Re-sampling a balanced validation set from training data."
        )
        train_ds, val_ds = _resample_val_from_train(
            train_ds, val_ds, config, samples_per_class=8
        )

    return {"train": train_ds, "val": val_ds, "test": test_ds}


def _resample_val_from_train(
    train_ds: ThreeClassDataset,
    val_ds: ThreeClassDataset,
    config: dict[str, Any],
    samples_per_class: int = 8,
) -> tuple[ThreeClassDataset, ThreeClassDataset]:
    """Move *samples_per_class* images per class from train into val."""
    # Group training samples by label
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for sample in train_ds.samples:
        by_class.setdefault(sample[1], []).append(sample)

    rng = random.Random(42)
    moved: list[tuple[Path, int]] = []
    for label in sorted(by_class):
        pool = by_class[label]
        chosen = rng.sample(pool, min(samples_per_class, len(pool)))
        moved.extend(chosen)

    moved_set = {s[0] for s in moved}
    train_ds.samples = [s for s in train_ds.samples if s[0] not in moved_set]

    # Replace val dataset samples entirely
    val_ds.samples = moved
    return train_ds, val_ds


def get_three_class_dataloaders(
    config: dict[str, Any],
) -> dict[str, DataLoader]:
    """Create dataloaders for the three-class task."""
    dataset_map = get_three_class_datasets(config)
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