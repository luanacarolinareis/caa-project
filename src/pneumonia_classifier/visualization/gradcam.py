from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class GradCAM:
    """Grad-CAM for a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(self.target_layer.register_forward_hook(self._save_activations))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _save_activations(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a Grad-CAM heatmap (H x W, float32 in [0, 1])."""
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor

        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.squeeze() > 0)

        self.model.zero_grad()
        target = logits[:, 0] if logits.shape[-1] == 1 else logits[:, class_idx]
        scalar = target if class_idx == 1 else -target
        scalar.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.ndim == 0:
            cam = cam.reshape(1, 1)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_) -> None:
        self.remove_hooks()


def load_image(image_path: str | Path, image_size: int = 224,) -> tuple[torch.Tensor, np.ndarray]:
    """Load image from disk. Returns (tensor, rgb_uint8_array)."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img.resize((image_size, image_size)))
    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
    ])
    tensor = tf(img)
    return tensor, rgb


def overlay_heatmap(heatmap: np.ndarray, image_rgb: np.ndarray, alpha: float = 0.4,) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image. Returns uint8 RGB array."""
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colormap_rgb + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


def save_gradcam(overlay: np.ndarray, output_path: str | Path,) -> Path:
    """Save an overlay image to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out)
    return out


def generate_gradcam(model: nn.Module, target_layer: nn.Module, image_path: str | Path, output_path: str | Path,
    image_size: int = 224, class_idx: int | None = None, alpha: float = 0.4,) -> tuple[Path, int]:
    """End-to-end Grad-CAM: load image, compute heatmap, save overlay.

    Works for any model. Pass the appropriate target layer:
    - DenseNet121: model.features.denseblock4.denselayer16.conv2
    - ResNet18:    model.layer4[-1].conv2
    
    Use ``get_gradcam_target_layer`` from transfer_learning.py to select automatically.

    Returns (saved_path, predicted_class_idx).
    """
    tensor, rgb = load_image(image_path, image_size=image_size)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with GradCAM(model, target_layer) as gc:
        heatmap = gc(tensor, class_idx=class_idx)
        pred_class = int(model(tensor.unsqueeze(0) if tensor.ndim == 3 else tensor).squeeze() > 0)

    overlay = overlay_heatmap(heatmap, rgb, alpha=alpha)
    saved = save_gradcam(overlay, output_path)
    return saved, pred_class
