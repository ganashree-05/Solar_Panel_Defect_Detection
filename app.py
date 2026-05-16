"""
app.py  –  Solar Panel Defect Detection  |  Flask Backend
----------------------------------------------------------
Serves the frontend and exposes a /predict API endpoint.

Run:
    python app.py
Then open:  http://localhost:5000
"""

import base64
import glob
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# ── Import helpers from predict.py ─────────────────────────────────────────
from predict import (
    CLASSES,
    DAMAGE_WEIGHTS,
    DIAGNOSIS,
    PALETTE,
    build_suggestion_panel,
    compute_damage_pct,
    draw_box,
    find_best_weights,
    severity_color,
    severity_label,
    _draw_summary_panel,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Load model once at startup ──────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS = find_best_weights("runs/detect/solar_panel_yolo/weights/best.pt")
print(f"Loading model: {YOLO_WEIGHTS}  |  Device: {DEVICE}")
yolo_model   = YOLO(YOLO_WEIGHTS)
print("Model ready.")


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file      = request.files["image"]
    conf_thr  = float(request.form.get("conf", 0.25))

    # Read image
    img_bytes = file.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image"}), 400

    h, w = img_bgr.shape[:2]
    vis  = img_bgr.copy()

    # ── Run YOLO ────────────────────────────────────────────────────────
    # Save to temp file (YOLO needs a path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        results    = yolo_model.predict(source=tmp_path, conf=conf_thr,
                                        imgsz=640, device=DEVICE, verbose=False)
        detections = results[0].boxes
        detections_list = []
        damage_pcts     = []
        primary_class   = None

        if detections is not None and len(detections):
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id     = int(box.cls[0].item())
                conf       = float(box.conf[0].item())
                class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                color      = PALETTE[cls_id % len(PALETTE)]
                dpct       = draw_box(vis, x1, y1, x2, y2, class_name, conf, color)
                damage_pcts.append(dpct)
                primary_class = class_name
                detections_list.append({
                    "class":      class_name,
                    "confidence": round(conf * 100, 1),
                    "damage_pct": dpct,
                    "severity":   severity_label(dpct),
                    "bbox":       [x1, y1, x2, y2],
                })
        else:
            # Fallback: lowest-conf prediction
            res2  = yolo_model.predict(source=tmp_path, conf=0.01,
                                       imgsz=640, device=DEVICE, verbose=False)
            boxes = res2[0].boxes
            if boxes is not None and len(boxes):
                best       = max(boxes, key=lambda b: float(b.conf[0]))
                cls_id     = int(best.cls[0].item())
                conf       = float(best.conf[0].item())
                class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                color      = PALETTE[cls_id % len(PALETTE)]
                dpct       = draw_box(vis, 0, 0, w - 1, h - 1, class_name, conf, color)
                damage_pcts.append(dpct)
                primary_class = class_name
                detections_list.append({
                    "class":      class_name,
                    "confidence": round(conf * 100, 1),
                    "damage_pct": dpct,
                    "severity":   severity_label(dpct),
                    "bbox":       [0, 0, w - 1, h - 1],
                })
    finally:
        os.unlink(tmp_path)

    avg_damage = sum(damage_pcts) / len(damage_pcts) if damage_pcts else 0.0
    primary    = primary_class or "Clean"

    # Draw summary bar on image
    if damage_pcts:
        _draw_summary_panel(vis, avg_damage, primary)

    # Encode annotated image → base64
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # Build diagnosis data
    diag = DIAGNOSIS.get(primary, DIAGNOSIS["Clean"])

    return jsonify({
        "annotated_image": img_b64,
        "primary_class":   primary,
        "avg_damage":      round(avg_damage, 1),
        "severity":        severity_label(avg_damage),
        "detections":      detections_list,
        "diagnosis": {
            "what_happened": diag["what_happened"],
            "impact":        diag["impact"],
            "suggestions":   diag["suggestions"],
        },
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
