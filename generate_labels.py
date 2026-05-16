"""
generate_labels.py  –  GradCAM-based Precise Bounding Box Label Generator
--------------------------------------------------------------------------
Instead of whole-image boxes, this script uses the trained CNN (ResNet-50)
with GradCAM to find EXACTLY where the damage is in each image, then writes
a tight YOLO bounding box around that region.

How it works:
  1. Forward pass the image through ResNet-50
  2. Compute gradients of the predicted class score w.r.t. the last conv layer
  3. Weight the feature maps by the gradients → activation heatmap
  4. Threshold the heatmap to find the high-activation region
  5. Fit a tight bounding box around that region
  6. Write YOLO label: <class_id> <cx> <cy> <w> <h>  (normalised 0-1)

For "Clean" panels (no damage), the box covers 80% of the image centre
since there is no localised defect to highlight.

Run:
    python generate_labels.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────
CNN_WEIGHTS  = "cnn_best.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
THRESHOLD    = 0.40   # GradCAM activation threshold (0-1); lower = larger box
MIN_BOX_AREA = 0.05   # minimum box area as fraction of image (avoids tiny boxes)
MAX_BOX_AREA = 0.95   # maximum box area (avoids near-whole-image boxes for damage)
CLEAN_BOX    = 0.80   # Clean panels: box covers this fraction of image centre

CLASSES = [
    "Bird-drop",          # 0
    "Clean",              # 1
    "Dusty",              # 2
    "Electrical-damage",  # 3
    "Physical-Damage",    # 4
    "Snow-Covered",       # 5
]
CLASS_TO_ID  = {c: i for i, c in enumerate(CLASSES)}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SPLITS       = ["train", "val", "test"]

# ── Model ──────────────────────────────────────────────────────────────────
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES)),
    )
    state = torch.load(CNN_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── GradCAM ────────────────────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping on ResNet layer4."""

    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        # Hook onto the last residual block
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor, class_idx):
        """
        Returns a (H, W) numpy heatmap in [0, 1].
        tensor: (1, 3, H, W) on DEVICE
        class_idx: int
        """
        self.model.zero_grad()
        output = self.model(tensor)
        score  = output[0, class_idx]
        score.backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ── Tight bounding box from heatmap ────────────────────────────────────────
def heatmap_to_yolo_box(cam: np.ndarray, threshold: float,
                         min_area: float, max_area: float):
    """
    Threshold the GradCAM heatmap and fit a tight bounding box.
    Returns (cx, cy, w, h) normalised to [0, 1], or None if no region found.
    """
    h_cam, w_cam = cam.shape
    binary = (cam >= threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Merge all contours into one bounding rect
    all_pts = np.vstack(contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)

    # Normalise to [0, 1]
    cx = (x + bw / 2) / w_cam
    cy = (y + bh / 2) / h_cam
    nw = bw / w_cam
    nh = bh / h_cam

    # Clamp
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    nw = np.clip(nw, 0.0, 1.0)
    nh = np.clip(nh, 0.0, 1.0)

    area = nw * nh
    if area < min_area or area > max_area:
        return None

    return float(cx), float(cy), float(nw), float(nh)


def add_padding(cx, cy, w, h, pad=0.08):
    """Add a small padding around the tight box so it doesn't clip edges."""
    w2 = min(w + pad, 1.0)
    h2 = min(h + pad, 1.0)
    cx2 = np.clip(cx, w2 / 2, 1.0 - w2 / 2)
    cy2 = np.clip(cy, h2 / 2, 1.0 - h2 / 2)
    return float(cx2), float(cy2), float(w2), float(h2)


# ── Per-image label generation ──────────────────────────────────────────────
def generate_label(img_path: Path, class_name: str, class_id: int,
                   gradcam: GradCAM) -> str:
    """
    Returns a YOLO label string for one image.
    Falls back to a sensible default if GradCAM fails.
    """
    # ── Clean: no localised damage, use centred 80% box ─────────────────
    if class_name == "Clean":
        s = CLEAN_BOX
        return f"{class_id} 0.5 0.5 {s:.4f} {s:.4f}"

    # ── Load & preprocess ────────────────────────────────────────────────
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        return f"{class_id} 0.5 0.5 0.9 0.9"

    tensor = TF(pil_img).unsqueeze(0).to(DEVICE)

    # ── GradCAM ──────────────────────────────────────────────────────────
    try:
        cam = gradcam(tensor, class_id)
    except Exception:
        return f"{class_id} 0.5 0.5 0.9 0.9"

    # ── Try progressively lower thresholds until we get a valid box ──────
    box = None
    for thr in [THRESHOLD, 0.30, 0.20, 0.15]:
        box = heatmap_to_yolo_box(cam, thr, MIN_BOX_AREA, MAX_BOX_AREA)
        if box is not None:
            break

    if box is None:
        # Fallback: use the peak activation location with a moderate box
        peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
        h_c, w_c = cam.shape
        cx = peak_x / w_c
        cy = peak_y / h_c
        box = (cx, cy, 0.50, 0.50)

    cx, cy, bw, bh = add_padding(*box, pad=0.06)
    return f"{class_id} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}"


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Loading CNN weights: {CNN_WEIGHTS}")
    model   = load_model()
    gradcam = GradCAM(model)
    print("GradCAM ready.\n")

    for split in SPLITS:
        split_dir = Path(split)
        total = 0
        for class_name, class_id in CLASS_TO_ID.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            img_paths = [p for p in class_dir.iterdir()
                         if p.suffix in IMAGE_EXTS]

            for img_path in img_paths:
                label_str  = generate_label(img_path, class_name, class_id, gradcam)
                label_path = img_path.with_suffix(".txt")
                label_path.write_text(label_str + "\n")
                total += 1

            print(f"  [{split}] {class_name}: {len(img_paths)} labels written")

        print(f"  [{split}] Total: {total}\n")

    print("Done. Precise GradCAM bounding box labels written alongside images.")
    print("Now retrain: python train_yolo.py")


if __name__ == "__main__":
    main()
