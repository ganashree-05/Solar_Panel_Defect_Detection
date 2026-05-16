"""
farm_detect.py
--------------
Solar Farm Grid Detection & Panel Matrix Mapping

For aerial/drone images showing multiple solar panels in a grid:
1. Detect all individual panels using contour analysis + YOLO
2. Cluster panels into rows and columns
3. Assign matrix coordinates (A1, B2, C3 …)
4. For each detected defect, report its grid position
5. Generate a cropped zoom of the affected panel
6. Return a grid map showing which panels are affected
"""

import cv2
import numpy as np
from pathlib import Path
import base64
from typing import List, Dict, Tuple, Optional


# ── Panel detection via image processing ──────────────────────────────────
def detect_panel_regions(img_bgr: np.ndarray) -> List[Dict]:
    """
    Detect individual solar panel regions in a farm/aerial image.
    Uses colour segmentation (blue/dark panels on green/brown background)
    + contour analysis to find rectangular panel shapes.

    Returns list of dicts: {x1, y1, x2, y2, cx, cy, area}
    """
    h, w = img_bgr.shape[:2]

    # ── Convert to HSV for colour-based segmentation ─────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Solar panels are typically dark blue/grey
    # Mask 1: dark blue panels
    mask1 = cv2.inRange(hsv, (90, 20, 20), (140, 255, 180))
    # Mask 2: dark grey/black panels
    mask2 = cv2.inRange(hsv, (0, 0, 10), (180, 60, 120))
    # Mask 3: navy blue
    mask3 = cv2.inRange(hsv, (100, 30, 30), (130, 200, 160))

    panel_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))

    # Find contours
    contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.003   # at least 0.3% of image
    max_area = (w * h) * 0.25    # at most 25% of image
    min_aspect = 0.3              # not too thin
    max_aspect = 5.0

    panels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # Solidity check — panels are fairly solid rectangles
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / max(hull_area, 1)
        if solidity < 0.5:
            continue

        panels.append({
            "x1": x, "y1": y,
            "x2": x + bw, "y2": y + bh,
            "cx": x + bw // 2,
            "cy": y + bh // 2,
            "area": area,
            "w": bw, "h": bh,
        })

    # Remove heavily overlapping boxes (keep larger)
    panels = _nms_panels(panels, iou_thresh=0.4)
    return panels


def _nms_panels(panels: List[Dict], iou_thresh: float = 0.4) -> List[Dict]:
    """Simple NMS to remove overlapping panel detections."""
    if not panels:
        return panels
    panels = sorted(panels, key=lambda p: p["area"], reverse=True)
    keep = []
    for p in panels:
        overlap = False
        for k in keep:
            iou = _iou(p, k)
            if iou > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(p)
    return keep


def _iou(a: Dict, b: Dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


# ── Grid assignment ────────────────────────────────────────────────────────
def assign_grid_positions(panels: List[Dict], row_gap_ratio: float = 0.6) -> List[Dict]:
    """
    Cluster panels into rows and columns, assign matrix labels like A1, B2.

    row_gap_ratio: if vertical gap between panel centres > this × median panel height,
                   treat as a new row.
    """
    if not panels:
        return panels

    # Sort by Y centre first
    panels = sorted(panels, key=lambda p: p["cy"])

    # Estimate median panel height
    heights = [p["h"] for p in panels]
    med_h   = float(np.median(heights)) if heights else 50

    # Cluster into rows by Y proximity
    rows: List[List[Dict]] = []
    current_row = [panels[0]]
    for p in panels[1:]:
        if abs(p["cy"] - current_row[-1]["cy"]) < med_h * row_gap_ratio:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
    rows.append(current_row)

    # Within each row, sort by X
    for row in rows:
        row.sort(key=lambda p: p["cx"])

    # Assign labels: rows → A, B, C … cols → 1, 2, 3 …
    row_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r_idx, row in enumerate(rows):
        row_label = row_labels[r_idx % len(row_labels)]
        for c_idx, panel in enumerate(row):
            panel["grid_row"]   = r_idx
            panel["grid_col"]   = c_idx
            panel["grid_label"] = f"{row_label}{c_idx + 1}"

    return panels


# ── Match YOLO detections to grid panels ──────────────────────────────────
def match_detections_to_grid(
    detections: List[Dict],
    grid_panels: List[Dict],
    img_h: int,
    img_w: int,
) -> List[Dict]:
    """
    For each YOLO detection bbox, find which grid panel it overlaps most.
    Adds 'grid_label', 'grid_row', 'grid_col' to each detection.
    If no grid panels found, assigns position based on image quadrant.
    """
    for det in detections:
        dx1, dy1, dx2, dy2 = det["bbox"]
        det_cx = (dx1 + dx2) / 2
        det_cy = (dy1 + dy2) / 2

        if grid_panels:
            # Find grid panel with highest IoU or containing the detection centre
            best_panel = None
            best_score = -1
            for gp in grid_panels:
                # Check if detection centre is inside this panel
                if gp["x1"] <= det_cx <= gp["x2"] and gp["y1"] <= det_cy <= gp["y2"]:
                    score = 2.0  # strong match
                else:
                    # Fall back to IoU
                    det_dict = {"x1": dx1, "y1": dy1, "x2": dx2, "y2": dy2}
                    score = _iou(det_dict, gp)
                if score > best_score:
                    best_score = score
                    best_panel = gp

            if best_panel and best_score > 0:
                det["grid_label"] = best_panel["grid_label"]
                det["grid_row"]   = best_panel["grid_row"]
                det["grid_col"]   = best_panel["grid_col"]
                det["panel_bbox"] = [best_panel["x1"], best_panel["y1"],
                                     best_panel["x2"], best_panel["y2"]]
            else:
                det["grid_label"] = _quadrant_label(det_cx, det_cy, img_w, img_h)
                det["grid_row"]   = 0
                det["grid_col"]   = 0
                det["panel_bbox"] = [dx1, dy1, dx2, dy2]
        else:
            det["grid_label"] = _quadrant_label(det_cx, det_cy, img_w, img_h)
            det["grid_row"]   = 0
            det["grid_col"]   = 0
            det["panel_bbox"] = [dx1, dy1, dx2, dy2]

    return detections


def _quadrant_label(cx: float, cy: float, w: int, h: int) -> str:
    """Fallback: assign quadrant label when no grid is detected."""
    row = "A" if cy < h / 2 else "B"
    col = 1 if cx < w / 2 else 2
    return f"{row}{col}"


# ── Crop zoomed panel ──────────────────────────────────────────────────────
def crop_panel_zoom(img_bgr: np.ndarray, bbox: List[int], pad_ratio: float = 0.15) -> np.ndarray:
    """
    Crop and zoom into a specific panel with padding.
    Returns the cropped panel image.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = img_bgr[cy1:cy2, cx1:cx2]
    # Resize to a standard display size
    target_w = 400
    scale    = target_w / max(crop.shape[1], 1)
    target_h = int(crop.shape[0] * scale)
    if target_h > 0 and target_w > 0:
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return crop


# ── Grid map visualisation ─────────────────────────────────────────────────
def draw_grid_map(
    grid_panels: List[Dict],
    detections:  List[Dict],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Draw a clean grid map showing panel positions and which ones are affected.
    Returns a BGR numpy image.
    """
    if not grid_panels:
        return _draw_no_grid_map(detections)

    n_rows = max(p["grid_row"] for p in grid_panels) + 1
    n_cols = max(p["grid_col"] for p in grid_panels) + 1

    cell_w, cell_h = 64, 48
    pad = 12
    label_w = 28
    label_h = 22

    map_w = label_w + n_cols * (cell_w + pad) + pad
    map_h = label_h + n_rows * (cell_h + pad) + pad

    canvas = np.ones((map_h, map_w, 3), dtype=np.uint8) * 248  # off-white

    # Affected labels
    affected = {d.get("grid_label"): d for d in detections if d.get("grid_label")}

    row_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Column headers
    for c in range(n_cols):
        x = label_w + c * (cell_w + pad) + pad + cell_w // 2
        cv2.putText(canvas, str(c + 1), (x - 5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    for r in range(n_rows):
        # Row label
        rl = row_labels[r % len(row_labels)]
        y_top = label_h + r * (cell_h + pad) + pad
        cv2.putText(canvas, rl, (4, y_top + cell_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        for c in range(n_cols):
            lbl = f"{rl}{c + 1}"
            x1  = label_w + c * (cell_w + pad) + pad
            y1  = y_top
            x2  = x1 + cell_w
            y2  = y1 + cell_h

            # Check if this cell exists in detected panels
            exists = any(p["grid_row"] == r and p["grid_col"] == c for p in grid_panels)
            if not exists:
                continue

            if lbl in affected:
                det   = affected[lbl]
                color = _hex_to_bgr(det.get("color", "#EF4444"))
                bg    = tuple(int(c * 0.15 + 240 * 0.85) for c in color)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                # Damage % text
                dmg_txt = f"{det.get('damage_pct', 0):.0f}%"
                cv2.putText(canvas, lbl, (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                cv2.putText(canvas, dmg_txt, (x1 + 4, y1 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                # Warning icon
                cv2.putText(canvas, "!", (x2 - 14, y1 + 16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                # Clean panel
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 240, 220), -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (160, 200, 160), 1)
                cv2.putText(canvas, lbl, (x1 + 4, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 100), 1, cv2.LINE_AA)
                cv2.putText(canvas, "OK", (x1 + 4, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 150, 100), 1, cv2.LINE_AA)

    # Legend
    legend_y = map_h - 2
    cv2.putText(canvas, "Green=OK  Coloured=Defect", (pad, legend_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130, 130, 130), 1, cv2.LINE_AA)

    return canvas


def _draw_no_grid_map(detections: List[Dict]) -> np.ndarray:
    """Fallback grid map when no panels were auto-detected."""
    canvas = np.ones((120, 300, 3), dtype=np.uint8) * 248
    cv2.putText(canvas, "Grid map unavailable", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(detections)} defect(s) detected", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Upload aerial/farm image", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


# ── Full farm analysis pipeline ────────────────────────────────────────────
def analyse_farm_image(
    img_bgr:    np.ndarray,
    detections: List[Dict],
) -> Dict:
    """
    Main entry point for farm/aerial image analysis.

    Args:
        img_bgr:    Original image (BGR numpy array)
        detections: List of YOLO detections with bbox, class, confidence, etc.

    Returns dict with:
        grid_panels:   all detected panel regions with grid labels
        detections:    enriched with grid_label, panel_bbox
        grid_map_b64:  base64 PNG of the grid map
        panel_crops:   list of {grid_label, crop_b64, detection} for each defect
        total_panels:  total panels detected
        affected_panels: count of panels with defects
        farm_mode:     True (signals frontend to use farm layout)
    """
    h, w = img_bgr.shape[:2]

    # 1. Detect panel regions
    grid_panels = detect_panel_regions(img_bgr)

    # 2. Assign grid positions
    if grid_panels:
        grid_panels = assign_grid_positions(grid_panels)

    # 3. Match detections to grid
    detections = match_detections_to_grid(detections, grid_panels, h, w)

    # 4. Generate grid map
    grid_map_img = draw_grid_map(grid_panels, detections, h, w)
    _, gm_buf    = cv2.imencode(".png", grid_map_img)
    grid_map_b64 = base64.b64encode(gm_buf).decode("utf-8")

    # 5. Crop zoomed views of each defective panel
    panel_crops = []
    seen_labels = set()
    for det in detections:
        lbl = det.get("grid_label", "?")
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)

        panel_bbox = det.get("panel_bbox", det["bbox"])
        crop       = crop_panel_zoom(img_bgr, panel_bbox, pad_ratio=0.12)
        _, c_buf   = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop_b64   = base64.b64encode(c_buf).decode("utf-8")

        panel_crops.append({
            "grid_label":  lbl,
            "crop_b64":    crop_b64,
            "class":       det.get("class", "Unknown"),
            "confidence":  det.get("confidence", 0),
            "damage_pct":  det.get("damage_pct", 0),
            "severity":    det.get("severity", "Unknown"),
            "color":       det.get("color", "#7C3AED"),
            "bbox":        det["bbox"],
            "panel_bbox":  panel_bbox,
        })

    # 6. Draw grid overlay on original image
    annotated = img_bgr.copy()
    for gp in grid_panels:
        lbl = gp.get("grid_label", "")
        affected = any(d.get("grid_label") == lbl for d in detections)
        color = (0, 180, 0) if not affected else (0, 0, 220)
        thickness = 1 if not affected else 2
        cv2.rectangle(annotated, (gp["x1"], gp["y1"]), (gp["x2"], gp["y2"]), color, thickness)
        cv2.putText(annotated, lbl,
                    (gp["x1"] + 4, gp["y1"] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return {
        "grid_panels":       [{"grid_label": p.get("grid_label",""), "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]} for p in grid_panels],
        "detections":        detections,
        "grid_map_b64":      grid_map_b64,
        "panel_crops":       panel_crops,
        "total_panels":      len(grid_panels),
        "affected_panels":   len(panel_crops),
        "farm_mode":         True,
        "annotated_farm":    annotated,
    }
