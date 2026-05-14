from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


def build_densenet121(pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    # Last conv block before the global average pool in DenseNet121
    return model.features.denseblock4.denselayer16.conv2


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == "densenet121":
        return build_densenet121(pretrained=pretrained)
    # TODO (Jakub): resnet18 branch here
    raise ValueError(f"Unknown model: {model_name!r}")


if __name__ == "__main__":
    model = build_model("densenet121")
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 1), f"Unexpected output shape: {out.shape}"
    print(f"Output shape: {out.shape}  (logits, compatible with BCEWithLogitsLoss)")
