"""Training engine for baseline and transfer-learning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from pneumonia_classifier.metrics import compute_binary_metrics
from pneumonia_classifier.utils import save_checkpoint, save_json


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    y_true: list[int] = []
    y_prob: list[float] = []

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        labels_for_loss = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels_for_loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(torch.sigmoid(logits).detach().cpu().squeeze(1).tolist())

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_prob: list[float] = []

    for images, labels in tqdm(dataloader, desc="val", leave=False):
        images = images.to(device)
        labels_for_loss = labels.float().unsqueeze(1).to(device)
        logits = model(images)
        loss = criterion(logits, labels_for_loss)

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(torch.sigmoid(logits).cpu().squeeze(1).tolist())

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    model_name: str,
    seed: int,
    max_epochs: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    phase: str = "train",
) -> dict[str, Any]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=config["training"]["learning_rate"],
    )
    scheduler_cfg = config["training"]["scheduler"]
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_cfg["factor"],
        patience=scheduler_cfg["patience"],
    )
    early_cfg = config["training"].get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True)
    early_patience = early_cfg.get("patience", 5)
    threshold = config["evaluation"].get("threshold", 0.5)

    model.to(device)
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, threshold
        )
        val_metrics = validate(model, dataloaders["val"], criterion, device, threshold)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "phase": phase,
            "model": model_name,
            "seed": seed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"{phase} epoch {epoch:03d}/{max_epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val": val_metrics, "model": model_name, "seed": seed, "phase": phase},
            )
        else:
            stale_epochs += 1

        if early_enabled and stale_epochs >= early_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    result = {
        "model": model_name,
        "seed": seed,
        "phase": phase,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    save_json(result, history_path)
    return result
