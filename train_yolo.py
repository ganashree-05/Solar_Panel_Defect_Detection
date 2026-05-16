"""
train_yolo.py
-------------
Fine-tunes YOLOv8m for solar-panel defect detection with damage percentage output.

Architecture:
  • Backbone  : CSPDarknet (CNN) – pretrained on COCO
  • Neck      : PANet feature pyramid
  • Head      : Decoupled detection head → bounding boxes + class confidence scores
                (confidence score is used to compute damage %)

The model outputs per-detection:
  • Bounding box (x1, y1, x2, y2)
  • Class label  (Bird-drop / Clean / Dusty / Electrical-damage / Physical-Damage / Snow-Covered)
  • Confidence   (0–1) → used by predict.py to compute damage percentage

Run:
    python train_yolo.py
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_SIZE   = "yolov8m.pt"   # medium model – good accuracy/speed on RTX 3050
DATA_YAML    = "dataset.yaml"
EPOCHS       = 150
IMG_SIZE     = 640
BATCH        = 16             # RTX 3050 6GB – reduce to 8 if OOM
WORKERS      = 4
AMP          = True           # automatic mixed precision (FP16 on RTX)
PROJECT      = "runs"
RUN_NAME     = "solar_v4"     # v4 – GradCAM++ tighter labels, YOLOv8m
PATIENCE     = 25             # early-stopping patience
DEVICE       = 0 if torch.cuda.is_available() else "cpu"

# ── Augmentation ────────────────────────────────────────────────────────────
# Tighter boxes need careful augmentation – avoid aggressive scale/translate
# that would push the damage region out of frame
AUG_PARAMS = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.05,   # reduced – keeps damage region in frame
    scale=0.4,        # reduced – avoids cropping tight boxes
    shear=1.0,
    flipud=0.1,
    fliplr=0.5,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.05,
)


def main():
    print(f"Device  : {DEVICE}  ({'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model   : {MODEL_SIZE}")
    print(f"Epochs  : {EPOCHS}  (early-stop patience={PATIENCE})")
    print(f"Batch   : {BATCH}  |  AMP={AMP}")
    print()

    # Load pretrained YOLOv8 weights
    model = YOLO(MODEL_SIZE)

    # ── Train ───────────────────────────────────────────────────────────────
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        patience=PATIENCE,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        amp=AMP,
        cache=True,
        # Confidence calibration: lower conf threshold during val
        # so the model learns to produce well-calibrated scores
        conf=0.001,
        iou=0.6,
        **AUG_PARAMS,
    )

    print("\n── Training complete ──")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights : {best_pt}")

    # ── Evaluate on test split ──────────────────────────────────────────────
    print("\nEvaluating on test split …")
    model_best = YOLO(str(best_pt))
    metrics = model_best.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        conf=0.25,
        iou=0.6,
        plots=True,
        save_json=True,
    )

    print("\n── Test Metrics ──")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print(f"\nRun inference with:")
    print(f"  python predict.py --source test --yolo {best_pt}")


if __name__ == "__main__":
    main()
