"""Dataset and dataloader helpers for chest X-ray classification."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import random
import re
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

    The Kaggle validation directory contains only 16 images and no VIRUS
    examples, which is far too small and unbalanced for reliable model
    selection / early stopping. When that is detected (or when
    ``data.val_fraction`` is set), a stratified validation set is carved out
    of the training data instead, holding out a fixed fraction of *each* class.
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

    # Fix the validation split if it is too small / missing a class
    val_fraction = float(data_config.get("val_fraction", 0.12))
    val_counts = val_ds.get_class_counts()
    missing_classes = [name for name, cnt in val_counts.items() if cnt == 0]
    if missing_classes or len(val_ds) < 60:
        print(
            f"[data] Validation split inadequate (counts={val_counts}, "
            f"missing={missing_classes}). Carving a stratified {val_fraction:.0%} "
            f"validation set from training data."
        )
        train_ds, val_ds = _stratified_val_from_train(
            train_ds, val_ds, config, val_fraction=val_fraction
        )
        print(
            f"[data] After resplit -> train={train_ds.get_class_counts()}, "
            f"val={val_ds.get_class_counts()}"
        )

    return {"train": train_ds, "val": val_ds, "test": test_ds}


def _patient_id(path: Path) -> str:
    """Extract a patient identifier from a Kermany chest X-ray filename.

    Pneumonia files are named ``personNNN_bacteria_K`` / ``personNNN_virus_K``;
    normal files use ``IM-XXXX-...`` or ``NORMAL2-IM-XXXX-...``. Patients can
    have several images, so the split must hold out whole patients to avoid
    leaking the same patient into both train and val.
    """
    name = path.stem.lower()
    person = re.match(r"(person\d+)", name)
    if person:
        return person.group(1)
    im = re.search(r"im-?(\d+)", name)
    if im:
        return f"im{im.group(1)}"
    return name  # fall back to the filename itself (treated as its own patient)


def _stratified_val_from_train(
    train_ds: ThreeClassDataset,
    val_ds: ThreeClassDataset,
    config: dict[str, Any],
    val_fraction: float = 0.12,
) -> tuple[ThreeClassDataset, ThreeClassDataset]:
    """Hold out ~``val_fraction`` of *each* class from train to build val,
    splitting by patient so no patient appears in both train and val.

    The validation transform (no augmentation) is applied to the held-out
    images. The split is deterministic for a given seed so runs are
    reproducible across models.
    """
    seed = int(config["training"].get("seed", 0))
    rng = random.Random(1000 + seed)

    # Group patients by their (single) label, then hold out whole patients

    # A patient is assigned to the class of its first-seen image; in this dataset each patient 
    # is single-class anyway
    patients_by_class: dict[int, list[str]] = {}
    patient_label: dict[str, int] = {}
    for path, label in train_ds.samples:
        pid = _patient_id(path)
        if pid not in patient_label:
            patient_label[pid] = label
            patients_by_class.setdefault(label, []).append(pid)

    held_out_patients: set[str] = set()
    for label in sorted(patients_by_class):
        pids = sorted(patients_by_class[label])
        rng.shuffle(pids)
        n_val = max(1, int(round(len(pids) * val_fraction)))
        held_out_patients.update(pids[:n_val])

    moved = [s for s in train_ds.samples if _patient_id(s[0]) in held_out_patients]
    train_ds.samples = [s for s in train_ds.samples if _patient_id(s[0]) not in held_out_patients]

    # The held-out images must use the validation transform
    val_ds.transform = get_transforms(config, "val")
    val_ds.samples = moved
    return train_ds, val_ds


def compute_class_weights(dataset: ThreeClassDataset) -> list[float]:
    """Inverse-frequency class weights, normalised to mean 1.0.

    Used to counter the train imbalance (BACTERIA has ~2x the samples of
    NORMAL/VIRUS), which otherwise biases the model toward BACTERIA.
    """
    counts = dataset.get_class_counts()
    totals = [counts[name] for name in THREE_CLASS_NAMES]
    n_total = sum(totals)
    n_classes = len(totals)
    raw = [n_total / (n_classes * c) if c > 0 else 0.0 for c in totals]
    mean_w = sum(raw) / n_classes
    return [w / mean_w for w in raw]


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


# Stage B of the hierarchical approach: bacteria vs virus.
# Trained ONLY on pneumonia images (the "normal" stage is the existing binary
# model). This mirrors Kermany et al., who built a dedicated bacterial-vs-viral
# classifier rather than a flat 3-way softmax. Labels: 0 = BACTERIA, 1 = VIRUS.

BACT_VIRUS_NAMES = ["BACTERIA", "VIRUS"]


class BacteriaVirusDataset(Dataset):
    """Dataset of PNEUMONIA-only chest X-rays labelled bacteria (0) / virus (1).

    Reads the standard NORMAL/PNEUMONIA layout but keeps only the PNEUMONIA
    images, deriving the label from the filename (``_bacteria_`` / ``_virus_``).
    """

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
            if not class_dir.is_dir() or class_dir.name.upper() == "NORMAL":
                continue
            for img_path in sorted(class_dir.iterdir()):
                if not (img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS):
                    continue
                three_label = _label_from_filename(img_path.name, class_dir.name)
                if three_label == THREE_CLASS_MAP["BACTERIA"]:
                    self.samples.append((img_path, 0))
                elif three_label == THREE_CLASS_MAP["VIRUS"]:
                    self.samples.append((img_path, 1))

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
        counts = {name: 0 for name in BACT_VIRUS_NAMES}
        for _, label in self.samples:
            counts[BACT_VIRUS_NAMES[label]] += 1
        return counts


def _patient_split_two_class(
    train_ds: BacteriaVirusDataset,
    val_ds: BacteriaVirusDataset,
    config: dict[str, Any],
    val_fraction: float = 0.12,
) -> tuple[BacteriaVirusDataset, BacteriaVirusDataset]:
    """Patient-aware stratified hold-out for the bacteria/virus dataset.

    Same patient-grouping logic as the three-class split: whole patients are
    moved to val so no patient leaks across the split.
    """
    seed = int(config["training"].get("seed", 0))
    rng = random.Random(1000 + seed)

    patients_by_class: dict[int, list[str]] = {}
    patient_label: dict[str, int] = {}
    for path, label in train_ds.samples:
        pid = _patient_id(path)
        if pid not in patient_label:
            patient_label[pid] = label
            patients_by_class.setdefault(label, []).append(pid)

    held_out: set[str] = set()
    for label in sorted(patients_by_class):
        pids = sorted(patients_by_class[label])
        rng.shuffle(pids)
        n_val = max(1, int(round(len(pids) * val_fraction)))
        held_out.update(pids[:n_val])

    moved = [s for s in train_ds.samples if _patient_id(s[0]) in held_out]
    train_ds.samples = [s for s in train_ds.samples if _patient_id(s[0]) not in held_out]
    val_ds.transform = get_transforms(config, "val")
    val_ds.samples = moved
    return train_ds, val_ds


def compute_class_weights_two(dataset: BacteriaVirusDataset) -> list[float]:
    """Inverse-frequency weights for the bacteria/virus task (mean 1.0)."""
    counts = dataset.get_class_counts()
    totals = [counts[name] for name in BACT_VIRUS_NAMES]
    n_total = sum(totals)
    n_classes = len(totals)
    raw = [n_total / (n_classes * c) if c > 0 else 0.0 for c in totals]
    mean_w = sum(raw) / n_classes
    return [w / mean_w for w in raw]


def get_bacteria_virus_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Dataloaders for the bacteria-vs-virus stage (pneumonia images only)."""
    data_config = config["data"]
    dataset_root = Path(data_config["root_dir"])
    split_dirs = {
        "train": data_config.get("train_dir", "train"),
        "val": data_config.get("val_dir", "val"),
        "test": data_config.get("test_dir", "test"),
    }

    train_ds = BacteriaVirusDataset(dataset_root / split_dirs["train"], get_transforms(config, "train"))
    val_ds = BacteriaVirusDataset(dataset_root / split_dirs["val"], get_transforms(config, "val"))
    test_ds = BacteriaVirusDataset(dataset_root / split_dirs["test"], get_transforms(config, "test"))

    # The Kaggle val/ has no virus pneumonia, so always carve a patient-aware split.
    val_fraction = float(data_config.get("val_fraction", 0.12))
    val_counts = val_ds.get_class_counts()
    if min(val_counts.values()) == 0 or len(val_ds) < 40:
        print(
            f"[data] Bacteria/virus val split inadequate (counts={val_counts}). "
            f"Carving a patient-aware {val_fraction:.0%} split from training data."
        )
        train_ds, val_ds = _patient_split_two_class(train_ds, val_ds, config, val_fraction)
        print(f"[data] After resplit -> train={train_ds.get_class_counts()}, val={val_ds.get_class_counts()}")

    batch_size = config["training"]["batch_size"]
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    }