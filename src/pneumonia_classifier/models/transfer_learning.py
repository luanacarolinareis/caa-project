from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_densenet121(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = _is_classifier_parameter(name)


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "features"):
        return model.features.denseblock4.denselayer16.conv2
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2
    raise ValueError(f"Unsupported model for Grad-CAM: {type(model).__name__}")


def build_model(model_name: str, pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    if model_name == "resnet18":
        return build_resnet18(pretrained=pretrained, num_classes=num_classes)
    if model_name == "densenet121":
        return build_densenet121(pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name!r}")


def _is_classifier_parameter(name: str) -> bool:
    return name.startswith("classifier.") or name.startswith("fc.")


if __name__ == "__main__":
    for model_name in ("resnet18", "densenet121"):
        for nc in (1, 3):
            model = build_model(model_name, num_classes=nc)
            model.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (1, nc), f"Unexpected output shape: {out.shape}"
            print(f"{model_name} num_classes={nc} output shape: {out.shape}")
