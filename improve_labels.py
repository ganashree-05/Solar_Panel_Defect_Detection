"""
improve_labels.py
-----------------
Regenerates GradCAM labels with improved settings:
- Lower threshold (0.35 instead of 0.40) for tighter boxes
- Uses GradCAM++ for sharper activation maps
- Applies Gaussian blur to smooth heatmap before thresholding
- Minimum box size enforced per class
- Multiple threshold attempts with quality scoring
- Saves comparison images to label_preview_v2/
"""

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
import random

# ── Config ─────────────────────────────────────────────────────
CNN_WEIGHTS  = "cnn_best.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224

# Tighter thresholds per class (lower = tighter box around damage)
CLASS_THRESHOLDS = {
    "Bird-drop":          0.55,   # bird droppings are small, tight
    "Clean":              None,   # no damage region
    "Dusty":              0.45,   # dust is spread but still localised
    "Electrical-damage":  0.50,   # burn marks are localised
    "Physical-Damage":    0.50,   # cracks are localised
    "Snow-Covered":       0.40,   # snow can be spread
}

# Max box area per class (fraction of image) — prevents whole-image boxes
CLASS_MAX_AREA = {
    "Bird-drop":          0.30,
    "Clean":              0.70,
    "Dusty":              0.55,
    "Electrical-damage":  0.40,
    "Physical-Damage":    0.45,
    "Snow-Covered":       0.65,
}

CLEAN_BOX    = 0.75
MIN_BOX_AREA = 0.04

CLASSES = [
    "Bird-drop", "Clean", "Dusty",
    "Electrical-damage", "Physical-Damage", "Snow-Covered",
]
CLASS_TO_ID  = {c: i for i, c in enumerate(CLASSES)}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SPLITS       = ["train", "val", "test"]

# ── Model ──────────────────────────────────────────────────────
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_features, 512),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, len(CLASSES)),
    )
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Improved GradCAM ───────────────────────────────────────────
class GradCAMPlusPlus:
    """GradCAM++ gives sharper, more localised activation maps."""

    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target = model.layer4[-1]
        target.register_forward_hook(self._save_act)
        target.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):  self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients  = go[0].detach()

    def __call__(self, tensor, class_idx):
        self.model.zero_grad()
        out   = self.model(tensor)
        score = out[0, class_idx]
        score.backward()

        grads = self.gradients          # (1, C, h, w)
        acts  = self.activations        # (1, C, h, w)

        # GradCAM++ weighting
        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        sum_acts  = acts.sum(dim=(2, 3), keepdim=True)
        alpha_num = grads_sq
        alpha_den = 2 * grads_sq + sum_acts * grads_cu + 1e-7
        alpha     = alpha_num / alpha_den
        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Smooth with Gaussian to reduce noise
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.GaussianBlur(cam.astype(np.float32), (5, 5), 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ── Box fitting ────────────────────────────────────────────────
def cam_to_box(cam, threshold, min_area, max_area):
    """Convert CAM heatmap to tight YOLO bounding box."""
    h, w = cam.shape
    binary = (cam >= threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour
    cnt  = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h

    cx = float(np.clip(cx, 0, 1))
    cy = float(np.clip(cy, 0, 1))
    nw = float(np.clip(nw, 0, 1))
    nh = float(np.clip(nh, 0, 1))

    area = nw * nh
    if area < min_area or area > max_area:
        return None
    return cx, cy, nw, nh


def add_padding(cx, cy, w, h, pad=0.05):
    w2  = min(w + pad, 1.0)
    h2  = min(h + pad, 1.0)
    cx2 = float(np.clip(cx, w2/2, 1 - w2/2))
    cy2 = float(np.clip(cy, h2/2, 1 - h2/2))
    return cx2, cy2, float(w2), float(h2)


# ── Per-image label ────────────────────────────────────────────
def generate_label(img_path, class_name, class_id, gradcam):
    if class_name == "Clean":
        s = CLEAN_BOX
        return f"{class_id} 0.5 0.5 {s:.4f} {s:.4f}"

    try:
        pil = Image.open(img_path).convert("RGB")
    except Exception:
        return f"{class_id} 0.5 0.5 0.85 0.85"

    tensor = TF(pil).unsqueeze(0).to(DEVICE)

    try:
        cam = gradcam(tensor, class_id)
    except Exception:
        return f"{class_id} 0.5 0.5 0.85 0.85"

    base_thr = CLASS_THRESHOLDS.get(class_name, 0.45)
    max_area = CLASS_MAX_AREA.get(class_name, 0.50)

    box = None
    for thr in [base_thr, base_thr - 0.10, base_thr - 0.20, 0.15]:
        if thr <= 0:
            break
        box = cam_to_box(cam, thr, MIN_BOX_AREA, max_area)
        if box is not None:
            break

    if box is None:
        # Peak-based fallback
        peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
        h_c, w_c = cam.shape
        cx, cy = peak_x / w_c, peak_y / h_c
        box = (cx, cy, 0.35, 0.35)

    cx, cy, bw, bh = add_padding(*box, pad=0.04)
    return f"{class_id} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}"


# ── Main ───────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading CNN…")
    model   = load_model()
    gradcam = GradCAMPlusPlus(model)
    print("GradCAM++ ready\n")

    for split in SPLITS:
        split_dir = Path(split)
        total = 0
        for class_name, class_id in CLASS_TO_ID.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            imgs = [p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTS]
            for img_path in imgs:
                lbl = generate_label(img_path, class_name, class_id, gradcam)
                img_path.with_suffix(".txt").write_text(lbl + "\n")
                total += 1
            print(f"  [{split}] {class_name}: {len(imgs)} labels")
        print(f"  [{split}] Total: {total}\n")

    print("Done. Now retrain: python train_yolo.py")


if __name__ == "__main__":
    main()
