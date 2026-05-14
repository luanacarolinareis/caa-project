"""Evaluation helpers for trained checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pneumonia_classifier.metrics import compute_binary_metrics
from pneumonia_classifier.utils import save_json


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    model.eval()
    model.to(device)
    y_true: list[int] = []
    y_prob: list[float] = []
    num_images = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits).cpu().squeeze(1).tolist()
        y_prob.extend(probabilities)
        y_true.extend(labels.cpu().tolist())
        num_images += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms_per_image = (elapsed / max(num_images, 1)) * 1000
    return y_true, y_prob, ms_per_image


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    model_name: str,
    seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    y_true, y_prob, ms_per_image = predict_probabilities(model, dataloader, device)
    metrics = compute_binary_metrics(
        y_true,
        y_prob,
        threshold=config["evaluation"].get("threshold", 0.5),
    )
    metrics.update(
        {
            "model": model_name,
            "seed": seed,
            "num_test_images": len(y_true),
            "inference_time_ms_per_image": ms_per_image,
        }
    )
    save_json(metrics, output_path)
    return metrics
