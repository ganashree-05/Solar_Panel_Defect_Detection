"""
predict.py
----------
Run inference on a single image or a folder of images using:
  1. The trained YOLOv8 model  → bounding boxes + class labels
  2. The trained CNN classifier → class probabilities (optional overlay)

Usage:
    # Single image
    python predict.py --source path/to/image.jpg

    # Folder
    python predict.py --source path/to/folder

    # Use specific weights
    python predict.py --source test/Clean --yolo runs/detect/solar_panel_yolo/weights/best.pt

    # Also run CNN classifier alongside YOLO
    python predict.py --source test/Clean --cnn cnn_best.pth
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]
NUM_CLASSES = len(CLASSES)

# Colour palette per class (BGR for OpenCV)
PALETTE = [
    (0,   165, 255),   # Bird-drop      – orange
    (0,   200,   0),   # Clean          – green
    (0,   215, 255),   # Dusty          – gold
    (0,     0, 220),   # Electrical     – red
    (220,   0,   0),   # Physical       – blue
    (255, 255,   0),   # Snow           – cyan
]

# Damage severity weights per class (0-100% scale)
# Clean = 0%, others scale with confidence
DAMAGE_WEIGHTS = {
    "Bird-drop":          0.60,  # moderate damage
    "Clean":              0.00,  # no damage
    "Dusty":              0.35,  # light damage
    "Electrical-damage":  0.95,  # severe damage
    "Physical-Damage":    0.90,  # severe damage
    "Snow-Covered":       0.50,  # moderate (temporary)
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ── Diagnosis & Suggestion database ────────────────────────────────────────
# Each entry has:
#   what_happened : plain-language explanation of the defect
#   impact        : how it affects the panel's performance
#   suggestions   : ordered list of recommended actions
DIAGNOSIS = {
    "Bird-drop": {
        "what_happened": "Bird droppings detected on panel surface.",
        "impact": [
            "Creates hot-spots by blocking sunlight on cells.",
            "Acidic waste corrodes anti-reflective coating.",
            "Can reduce output by 10-30% if left untreated.",
        ],
        "suggestions": [
            "1. Clean with soft cloth + diluted vinegar solution.",
            "2. Rinse thoroughly with clean water.",
            "3. Install bird deterrent spikes or netting.",
            "4. Schedule monthly visual inspections.",
            "5. Apply hydrophobic coating after cleaning.",
        ],
    },
    "Clean": {
        "what_happened": "Panel surface is clean. No defects detected.",
        "impact": [
            "Operating at optimal efficiency.",
            "No immediate action required.",
        ],
        "suggestions": [
            "1. Continue regular monthly inspections.",
            "2. Clean every 3-6 months as preventive measure.",
            "3. Monitor output via inverter dashboard.",
            "4. Check mounting hardware annually.",
        ],
    },
    "Dusty": {
        "what_happened": "Dust and particulate accumulation on surface.",
        "impact": [
            "Reduces light transmission to solar cells.",
            "Uniform dust can cut efficiency by 5-25%.",
            "Worse in arid/desert environments.",
        ],
        "suggestions": [
            "1. Rinse with clean water using a soft brush.",
            "2. Clean early morning to avoid thermal shock.",
            "3. Increase cleaning frequency in dry seasons.",
            "4. Consider automated cleaning system.",
            "5. Apply anti-soiling nano-coating.",
        ],
    },
    "Electrical-damage": {
        "what_happened": "Electrical damage detected (burn marks / arc damage).",
        "impact": [
            "Severe: panel may be producing zero output.",
            "Risk of fire or electrical hazard.",
            "Damaged bypass diodes cause string failure.",
            "Can damage inverter if not isolated.",
        ],
        "suggestions": [
            "1. IMMEDIATELY disconnect panel from system.",
            "2. Do NOT attempt DIY repair — call a technician.",
            "3. Inspect wiring, connectors, and junction box.",
            "4. Test with IV-curve tracer to assess cell damage.",
            "5. Replace panel if burn area > 5% of surface.",
            "6. Check inverter for fault codes.",
        ],
    },
    "Physical-Damage": {
        "what_happened": "Physical damage detected (cracks, chips, or impact marks).",
        "impact": [
            "Micro-cracks reduce active cell area.",
            "Water ingress through cracks causes delamination.",
            "Hot-spots form at crack boundaries.",
            "Structural integrity compromised.",
        ],
        "suggestions": [
            "1. Document damage with photos for warranty claim.",
            "2. Check if damage is covered under manufacturer warranty.",
            "3. Use EL (electroluminescence) imaging to map cracks.",
            "4. Seal minor edge chips with UV-resistant sealant.",
            "5. Replace panel if cracks cross multiple cells.",
            "6. Inspect mounting for vibration/stress causes.",
        ],
    },
    "Snow-Covered": {
        "what_happened": "Snow or ice accumulation covering the panel.",
        "impact": [
            "Complete blockage of sunlight — near-zero output.",
            "Weight stress on mounting structure.",
            "Ice expansion can cause micro-cracks.",
        ],
        "suggestions": [
            "1. Allow natural melting when safe to do so.",
            "2. Use soft foam brush to gently remove loose snow.",
            "3. Never use metal tools — risk of scratching glass.",
            "4. Install heating elements for heavy-snow regions.",
            "5. Increase tilt angle (>35 deg) to shed snow faster.",
            "6. Check for ice-induced micro-cracks after thaw.",
        ],
    },
}


# ── Weight resolver ─────────────────────────────────────────────────────────
def find_best_weights(default: str) -> str:
    """
    Resolve the best available YOLO weights file.
    Priority order:
      1. Exact path if it exists
      2. solar_v4 (GradCAM++ tight labels, best model)
      3. solar_precise3 (GradCAM labels)
      4. Newest best.pt by modification time
    """
    if Path(default).exists():
        return default

    # Priority: prefer models trained with tight GradCAM labels
    preferred = [
        "runs/**/solar_v4*/weights/best.pt",
        "runs/**/solar_precise3*/weights/best.pt",
        "runs/**/solar_precise*/weights/best.pt",
    ]
    for pattern in preferred:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            found = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)[0]
            print(f"[INFO] Using preferred model: {found}")
            return found

    # Fallback: newest best.pt
    candidates = sorted(
        glob.glob("runs/**/best.pt", recursive=True),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if candidates:
        found = candidates[0]
        print(f"[INFO] Weights not found at '{default}', auto-resolved to: {found}")
        return found

    raise FileNotFoundError(
        f"No YOLO weights found. Looked for '{default}' and searched runs/**.\n"
        "Make sure training has completed (train_yolo.py) before running predict.py."
    )


# ── CNN model loader ────────────────────────────────────────────────────────
def load_cnn(weights_path: str, device):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


CNN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def cnn_predict(model, img_bgr: np.ndarray, device):
    """Returns (class_name, confidence)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor  = CNN_TF(pil_img).unsqueeze(0).to(device)
    logits  = model(tensor)
    probs   = F.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    return CLASSES[idx.item()], conf.item()


# ── Damage percentage calculator ────────────────────────────────────────────
def compute_damage_pct(class_name: str, yolo_conf: float) -> float:
    """
    Compute damage percentage:
      damage% = class_weight × yolo_confidence × 100
    Clean panels always return 0%.
    Result is clamped to [0, 100].
    """
    weight = DAMAGE_WEIGHTS.get(class_name, 0.5)
    return round(min(max(weight * yolo_conf * 100, 0.0), 100.0), 1)


def severity_label(pct: float) -> str:
    if pct == 0:
        return "No Damage"
    elif pct < 20:
        return "Minimal"
    elif pct < 40:
        return "Low"
    elif pct < 60:
        return "Moderate"
    elif pct < 80:
        return "High"
    else:
        return "Critical"


def severity_color(pct: float):
    """BGR colour for the severity bar: green → yellow → red."""
    if pct == 0:
        return (0, 200, 0)
    elif pct < 40:
        return (0, 220, 255)   # yellow
    elif pct < 70:
        return (0, 140, 255)   # orange
    else:
        return (0, 0, 220)     # red


# ── Drawing helpers ─────────────────────────────────────────────────────────
def draw_box(img, x1, y1, x2, y2, class_name, yolo_conf, color):
    """
    Draw a clean, highly visible bounding box with:
    - Thick coloured rectangle around the damage area
    - Corner accent marks (L-shaped corners) for precision feel
    - Label pill above (or below) the box
    - Damage % badge
    """
    h_img, w_img = img.shape[:2]

    damage_pct = compute_damage_pct(class_name, yolo_conf)
    sev        = severity_label(damage_pct)
    sev_col    = severity_color(damage_pct)

    # ── 1. Main bounding box ─────────────────────────────────────────────
    # Thick outer box
    thickness = max(3, int(min(h_img, w_img) / 180))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # ── 2. Corner accent marks (L-shapes at each corner) ─────────────────
    corner_len = max(12, int(min(x2-x1, y2-y1) * 0.18))
    ct = max(3, thickness + 1)
    # Top-left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, ct)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, ct)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, ct)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, ct)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, ct)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, ct)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, ct)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, ct)

    # ── 3. Label pill ─────────────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.40, min(h_img, w_img) / 700)
    label      = f"{class_name}  {yolo_conf*100:.0f}%  |  {damage_pct:.0f}% dmg"

    (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
    pad = 5
    pill_h = th + bl + pad * 2
    pill_w = tw + pad * 2

    # Position: above box if space, else below
    if y1 - pill_h - 4 >= 0:
        py1 = y1 - pill_h - 4
        py2 = y1 - 4
    else:
        py1 = y2 + 4
        py2 = y2 + pill_h + 4

    px1 = max(0, x1)
    px2 = min(w_img, px1 + pill_w)

    # Filled pill background
    cv2.rectangle(img, (px1, py1), (px2, py2), color, -1)
    # White text
    cv2.putText(img, label,
                (px1 + pad, py2 - bl - pad + th),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 4. Severity dot in top-right corner of box ────────────────────────
    dot_r = max(6, thickness * 2)
    cv2.circle(img, (x2 - dot_r - 2, y1 + dot_r + 2), dot_r, sev_col, -1)
    cv2.circle(img, (x2 - dot_r - 2, y1 + dot_r + 2), dot_r, (255,255,255), 1)

    return damage_pct


def build_suggestion_panel(class_name: str, damage_pct: float, panel_w: int, img_h: int) -> np.ndarray:
    """
    Build a vertical suggestion sidebar showing:
      • What happened to the panel
      • Impact on performance
      • Step-by-step improvement suggestions
    Returns a BGR numpy array of shape (img_h, panel_w, 3).
    """
    diag      = DIAGNOSIS.get(class_name, DIAGNOSIS["Clean"])
    sev       = severity_label(damage_pct)
    sev_col   = severity_color(damage_pct)
    panel     = np.zeros((img_h, panel_w, 3), dtype=np.uint8)
    panel[:]  = (28, 28, 28)

    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
    FONT_REG  = cv2.FONT_HERSHEY_SIMPLEX
    WHITE     = (240, 240, 240)
    GREY      = (160, 160, 160)
    YELLOW    = (0, 220, 255)
    CYAN      = (255, 220, 0)

    title_sc  = max(0.42, panel_w / 540)
    body_sc   = max(0.36, panel_w / 660)
    pad       = max(8, panel_w // 28)
    line_gap  = max(17, int(panel_w / 17))
    y         = pad + int(line_gap * 1.1)

    def put(text, x, cy, font, scale, color, thickness=1):
        cv2.putText(panel, text, (x, cy), font, scale, color, thickness, cv2.LINE_AA)
        return cy + line_gap

    def divider(cy, color=(65, 65, 65)):
        cv2.line(panel, (pad, cy), (panel_w - pad, cy), color, 1)
        return cy + int(line_gap * 0.55)

    def wrap(text, max_chars):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    max_chars = max(16, int(panel_w / (body_sc * 11.5)))

    # ── Title ────────────────────────────────────────────────────────────
    y = put("SOLAR PANEL", pad, y, FONT_BOLD, title_sc, CYAN, 1)
    y = put("DIAGNOSIS REPORT", pad, y, FONT_BOLD, title_sc, CYAN, 1)
    y = divider(y, (90, 90, 90))

    # ── Class + damage ───────────────────────────────────────────────────
    y = put(f"Class : {class_name}", pad, y, FONT_BOLD, body_sc, WHITE, 1)
    y = put(f"Damage: {damage_pct:.1f}%  [{sev}]", pad, y, FONT_REG, body_sc, sev_col, 1)
    y = divider(y)

    # ── What happened ────────────────────────────────────────────────────
    if y < img_h - line_gap * 3:
        y = put("WHAT HAPPENED", pad, y, FONT_BOLD, body_sc, YELLOW, 1)
        for line in wrap(diag["what_happened"], max_chars):
            if y < img_h - line_gap:
                y = put(line, pad + 4, y, FONT_REG, body_sc * 0.92, WHITE)
        y += int(line_gap * 0.2)
        y = divider(y)

    # ── Impact ───────────────────────────────────────────────────────────
    if y < img_h - line_gap * 4:
        y = put("IMPACT ON PANEL", pad, y, FONT_BOLD, body_sc, YELLOW, 1)
        for item in diag["impact"]:
            for line in wrap(item, max_chars):
                if y < img_h - line_gap:
                    y = put(line, pad + 4, y, FONT_REG, body_sc * 0.88, GREY)
        y += int(line_gap * 0.2)
        y = divider(y)

    # ── Suggestions ──────────────────────────────────────────────────────
    if y < img_h - line_gap * 4:
        y = put("HOW TO IMPROVE", pad, y, FONT_BOLD, body_sc, YELLOW, 1)
        for step in diag["suggestions"]:
            for line in wrap(step, max_chars):
                if y < img_h - line_gap:
                    y = put(line, pad + 4, y, FONT_REG, body_sc * 0.88, WHITE)

    # ── Border + left accent stripe ──────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (panel_w - 1, img_h - 1), (60, 60, 60), 1)
    cv2.line(panel, (0, 0), (0, img_h - 1), sev_col, 4)

    return panel


def _draw_summary_panel(img, avg_damage: float, primary_class: str):
    """Draw overall damage summary bar at the very bottom of image."""
    h, w = img.shape[:2]
    panel_h = max(36, int(h * 0.06))
    y1 = h - panel_h

    # Semi-transparent strip — only 6% of image height, doesn't cover boxes
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y1), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    sev  = severity_label(avg_damage)
    scol = severity_color(avg_damage)
    text = f"  Damage: {avg_damage:.1f}%   Severity: {sev}   Class: {primary_class}  "
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.38, min(h, w) / 1100)
    (tw, th), bl = cv2.getTextSize(text, font, scale, 1)
    ty = y1 + (panel_h + th) // 2
    cv2.putText(img, text, (max(0, (w - tw) // 2), ty),
                font, scale, scol, 1, cv2.LINE_AA)


def _draw_cnn_overlay(img, cnn_cls: str, cnn_conf: float):
    """Draw CNN prediction overlay at top-right corner."""
    h, w = img.shape[:2]
    overlay_text = f"CNN: {cnn_cls} ({cnn_conf*100:.1f}%)"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(h, w) / 900)
    (tw, th), bl = cv2.getTextSize(overlay_text, font, scale, 1)
    cv2.rectangle(img, (w - tw - 12, 4), (w - 4, th + bl + 10), (50, 50, 50), -1)
    cv2.putText(img, overlay_text,
                (w - tw - 8, th + 8),
                font, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ── Main inference ──────────────────────────────────────────────────────────
def run_inference(source: str,
                  yolo_weights: str,
                  cnn_weights: str | None,
                  conf_thresh: float,
                  output_dir: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Resolve weights (handles doubled path issue)
    yolo_weights = find_best_weights(yolo_weights)

    # Load YOLO
    print(f"Loading YOLO weights: {yolo_weights}")
    yolo = YOLO(yolo_weights)

    # Load CNN (optional)
    cnn_model = None
    if cnn_weights and Path(cnn_weights).exists():
        print(f"Loading CNN weights : {cnn_weights}")
        cnn_model = load_cnn(cnn_weights, device)
    elif cnn_weights:
        print(f"[WARN] CNN weights not found at '{cnn_weights}', skipping CNN overlay.")

    # Collect images
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = [p for p in source_path.rglob("*") if p.suffix in IMAGE_EXTS]
    else:
        print(f"ERROR: source '{source}' not found.")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s).\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] Cannot read {img_path}, skipping.")
            continue

        h, w = img_bgr.shape[:2]
        vis  = img_bgr.copy()

        # ── YOLO detection ──────────────────────────────────────────────
        results = yolo.predict(
            source=str(img_path),
            conf=conf_thresh,
            imgsz=640,
            device=str(device),
            verbose=False,
        )

        detections  = results[0].boxes
        yolo_label  = None
        damage_pcts = []

        if detections is not None and len(detections):
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id     = int(box.cls[0].item())
                conf       = float(box.conf[0].item())
                class_name = CLASSES[cls_id] if cls_id < NUM_CLASSES else f"cls{cls_id}"
                color      = PALETTE[cls_id % len(PALETTE)]
                dpct       = draw_box(vis, x1, y1, x2, y2, class_name, conf, color)
                damage_pcts.append(dpct)
                yolo_label = class_name
        else:
            # No detection above threshold → use best low-conf prediction
            yolo_res = yolo.predict(
                source=str(img_path),
                conf=0.01,
                imgsz=640,
                device=str(device),
                verbose=False,
            )
            boxes = yolo_res[0].boxes
            if boxes is not None and len(boxes):
                best       = max(boxes, key=lambda b: float(b.conf[0]))
                cls_id     = int(best.cls[0].item())
                conf       = float(best.conf[0].item())
                class_name = CLASSES[cls_id] if cls_id < NUM_CLASSES else f"cls{cls_id}"
                color      = PALETTE[cls_id % len(PALETTE)]
                dpct       = draw_box(vis, 0, 0, w - 1, h - 1, class_name, conf, color)
                damage_pcts.append(dpct)
                yolo_label = class_name
            else:
                cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (128, 128, 128), 2)
                cv2.putText(vis, "No detection", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # ── Overall damage summary panel (bottom of image) ──────────────
        avg_damage = sum(damage_pcts) / len(damage_pcts) if damage_pcts else 0.0
        if damage_pcts:
            _draw_summary_panel(vis, avg_damage, yolo_label or "Unknown")

        # ── CNN overlay (top-right corner of image) ──────────────────────
        if cnn_model is not None:
            cnn_cls, cnn_conf = cnn_predict(cnn_model, img_bgr, device)
            _draw_cnn_overlay(vis, cnn_cls, cnn_conf)

        # ── Suggestion sidebar (attached to right of image) ──────────────
        primary = yolo_label or "Clean"
        sidebar_w = max(260, int(vis.shape[1] * 0.42))
        sidebar   = build_suggestion_panel(primary, avg_damage, sidebar_w, vis.shape[0])
        final     = np.hstack([vis, sidebar])

        # ── Save result ─────────────────────────────────────────────────
        try:
            rel = img_path.relative_to(source_path.parent if source_path.is_file() else source_path)
        except ValueError:
            rel = img_path.name
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), final)

        dmg_str = f"{avg_damage:.1f}%" if damage_pcts else "N/A"
        print(f"  {img_path.name:40s}  Class={primary:20s}  Damage={dmg_str}")

    print(f"\nResults saved to: {out_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Solar Panel Defect Detector – Inference")
    p.add_argument("--source",  required=True,
                   help="Image file or folder")
    p.add_argument("--yolo",    default="runs/detect/runs/solar_panel_yolo2/weights/best.pt",
                   help="Path to YOLO best.pt weights (auto-resolved if not found)")
    p.add_argument("--cnn",     default=None,
                   help="Path to CNN cnn_best.pth weights (optional)")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="YOLO confidence threshold (default 0.25)")
    p.add_argument("--output",  default="predictions",
                   help="Output directory for annotated images")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        source=args.source,
        yolo_weights=args.yolo,
        cnn_weights=args.cnn,
        conf_thresh=args.conf,
        output_dir=args.output,
    )
