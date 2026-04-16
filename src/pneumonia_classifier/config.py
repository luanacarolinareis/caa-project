"""Configuration helpers for the pneumonia classification project."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "data",
    "preprocessing",
    "training",
    "models",
    "evaluation",
    "outputs",
}


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Check that the config contains the sections expected by the project."""
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(config)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Config is missing required section(s): {missing_keys}")

    image_size = config["data"].get("image_size")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("data.image_size must be a positive integer")

    batch_size = config["training"].get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("training.batch_size must be a positive integer")

    learning_rate = config["training"].get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be a positive number")

    class_names = config["data"].get("class_names", [])
    if class_names != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("data.class_names must be ['NORMAL', 'PNEUMONIA']")


def merge_config(
    base_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``base_config`` recursively updated with overrides."""
    if not overrides:
        return deepcopy(base_config)

    merged = deepcopy(base_config)
    _deep_update(merged, overrides)
    validate_config(merged)
    return merged


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
