"""Training engine for baseline and transfer-learning experiments."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from pneumonia_classifier.config import is_three_class
from pneumonia_classifier.metrics import compute_binary_metrics, compute_multiclass_metrics
from pneumonia_classifier.utils import save_checkpoint, save_json


def _build_criterion(
    config: dict[str, Any],
    device: torch.device | None = None,
    class_weights: list[float] | None = None,
) -> nn.Module:
    """Return the appropriate loss function for the task.

    For the three-class task, ``class_weights`` (inverse-frequency weights)
    can be supplied to counter class imbalance. They are ignored for the
    binary task.
    """
    loss_name = config["training"].get("loss", "bce_with_logits")
    if loss_name == "cross_entropy":
        weight_tensor = None
        if class_weights is not None and config["training"].get("use_class_weights", False):
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            if device is not None:
                weight_tensor = weight_tensor.to(device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
    return nn.BCEWithLogitsLoss()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    three_class = is_three_class(config)
    threshold = config["evaluation"].get("threshold", 0.5)
    class_names = config["data"].get("class_names", [])

    model.train()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_all: list = []

    # file=sys.stdout: keep the progress bar on stdout so PowerShell does not
    # render it as a red "error" (tqdm writes to stderr by default).
    for images, labels in tqdm(dataloader, desc="train", leave=False, file=sys.stdout):
        images = images.to(device)

        if three_class:
            labels_for_loss = labels.long().to(device)
        else:
            labels_for_loss = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels_for_loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())

        if three_class:
            probs = torch.softmax(logits, dim=1).detach().cpu()
            y_pred.extend(probs.argmax(dim=1).tolist())
            y_prob_all.extend(probs.tolist())
        else:
            probs = torch.sigmoid(logits).detach().cpu().squeeze(1)
            y_prob_all.extend(probs.tolist())

    if three_class:
        metrics = compute_multiclass_metrics(
            y_true, y_pred, y_prob=y_prob_all, class_names=class_names,
        )
    else:
        metrics = compute_binary_metrics(y_true, y_prob_all, threshold=threshold)

    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    three_class = is_three_class(config)
    threshold = config["evaluation"].get("threshold", 0.5)
    class_names = config["data"].get("class_names", [])

    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_all: list = []

    for images, labels in tqdm(dataloader, desc="val", leave=False, file=sys.stdout):
        images = images.to(device)

        if three_class:
            labels_for_loss = labels.long().to(device)
        else:
            labels_for_loss = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels_for_loss)

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())

        if three_class:
            probs = torch.softmax(logits, dim=1).cpu()
            y_pred.extend(probs.argmax(dim=1).tolist())
            y_prob_all.extend(probs.tolist())
        else:
            probs = torch.sigmoid(logits).cpu().squeeze(1)
            y_prob_all.extend(probs.tolist())

    if three_class:
        metrics = compute_multiclass_metrics(
            y_true, y_pred, y_prob=y_prob_all, class_names=class_names,
        )
    else:
        metrics = compute_binary_metrics(y_true, y_prob_all, threshold=threshold)

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
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    three_class = is_three_class(config)
    # For multiclass, select the best checkpoint by validation F1-macro: the
    # held-out val set is small and the loss is noisy, whereas F1-macro tracks
    # the goal of balancing the minority classes (NORMAL / VIRUS)
    select_by_f1 = three_class and config["training"].get("select_by", "loss") == "f1_macro"
    criterion = _build_criterion(config, device=device, class_weights=class_weights)
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

    model.to(device)
    best_val_loss = float("inf")
    best_val_score = -float("inf")  # F1-macro when select_by_f1
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, config
        )
        val_metrics = validate(model, dataloaders["val"], criterion, device, config)
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
            f"val_f1={val_metrics.get('f1_macro', val_metrics.get('f1', 0)):.4f}"
        )

        if select_by_f1:
            current_score = val_metrics.get("f1_macro", 0.0)
            improved = current_score > best_val_score
        else:
            current_score = -val_metrics["loss"]
            improved = val_metrics["loss"] < best_val_loss

        if improved:
            best_val_loss = min(best_val_loss, val_metrics["loss"])
            best_val_score = max(best_val_score, current_score)
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
