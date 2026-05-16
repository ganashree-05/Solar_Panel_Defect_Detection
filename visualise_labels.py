"""
visualise_labels.py
-------------------
Quick sanity-check: draws the GradCAM-generated bounding boxes on sample
images from each class so you can visually verify they are tight and accurate.

Run:
    python visualise_labels.py
Output: label_preview/ folder with annotated images
"""

import cv2
import numpy as np
from pathlib import Path
import random

CLASSES = [
    "Bird-drop", "Clean", "Dusty",
    "Electrical-damage", "Physical-Damage", "Snow-Covered",
]
PALETTE = [
    (0, 165, 255), (0, 200, 0), (0, 215, 255),
    (0, 0, 220),   (220, 0, 0), (255, 220, 0),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
OUT_DIR    = Path("label_preview")
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)

for cls_id, cls_name in enumerate(CLASSES):
    cls_dir = Path("train") / cls_name
    imgs    = [p for p in cls_dir.iterdir() if p.suffix in IMAGE_EXTS]
    # Pick up to 3 random samples
    samples = random.sample(imgs, min(3, len(imgs)))
    color   = PALETTE[cls_id]

    for img_path in samples:
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name}  box={bw*bh:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out = OUT_DIR / f"{cls_name}_{img_path.stem}.jpg"
        cv2.imwrite(str(out), img)
        print(f"  Saved: {out}")

print(f"\nPreview images in: {OUT_DIR}/")
