"""
backend/app.py  –  SolarScan API + Auth (Email + Google OAuth)
===============================================================
Endpoints:
  POST /api/auth/register        → create account with email
  POST /api/auth/login           → login with email/password
  POST /api/auth/logout          → clear session
  GET  /api/auth/me              → current user info
  GET  /api/auth/google          → start Google OAuth flow
  GET  /api/auth/google/callback → Google OAuth callback
  GET  /api/health               → model status (protected)
  POST /api/predict              → run detection (protected)
  GET  /api/classes              → defect classes (protected)

Run:
    python backend/app.py
    → http://localhost:5000
"""

import base64
import os
import sys
import tempfile
from pathlib import Path
from functools import wraps

import cv2
import numpy as np
import torch
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth

# ── resolve project root so we can import predict.py ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import (
    CLASSES,
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
from farm_detect import analyse_farm_image
from ultralytics import YOLO

# ── Load .env if present ───────────────────────────────────────────────────
env_path = ROOT / ".env"
try:
    if env_path.is_file() and env_path.stat().st_size > 0:
        # utf-8-sig strips BOM if present (Windows PowerShell adds it)
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[API] Loaded .env from {env_path}")
except Exception as e:
    print(f"[API] Warning: could not read .env — {e}")

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SECRET_KEY           = os.environ.get("SECRET_KEY", "solarscan-secret-key-change-in-production")
GOOGLE_CONFIGURED    = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# ── App ────────────────────────────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ── Google OAuth via Authlib ───────────────────────────────────────────────
oauth = OAuth(app)

if GOOGLE_CONFIGURED:
    google = oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    print(f"[API] Google OAuth : ENABLED (client_id={GOOGLE_CLIENT_ID[:20]}…)")
else:
    google = None
    print("[API] Google OAuth : DISABLED (set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env)")

# ── In-memory user store (replace with DB in production) ──────────────────
# Format: { email: { name, password_hash, avatar, provider } }
USERS = {
    "admin@solarscan.com": {
        "name":          "Admin User",
        "password_hash": generate_password_hash("admin123"),
        "avatar":        None,
        "provider":      "email",
    }
}

# ── Auth helpers ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized", "redirect": "/login.html"}), 401
        return f(*args, **kwargs)
    return decorated

# ── Load model once ────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# Priority: solar_v4 (GradCAM++ tight labels) > solar_precise3 > newest
YOLO_WEIGHTS = find_best_weights(str(ROOT / "runs/detect/solar_panel_yolo/weights/best.pt"))
print(f"[API] Loading model : {YOLO_WEIGHTS}")
print(f"[API] Device        : {DEVICE}")
yolo_model   = YOLO(YOLO_WEIGHTS)
print("[API] Model ready. Listening on http://localhost:5000")

# ── Class metadata (sent to frontend) ─────────────────────────────────────
CLASS_META = {
    "Bird-drop":         {"color": "#F97316", "max_damage": 60,  "icon": "🐦"},
    "Clean":             {"color": "#22C55E", "max_damage": 0,   "icon": "✅"},
    "Dusty":             {"color": "#EAB308", "max_damage": 35,  "icon": "🌫️"},
    "Electrical-damage": {"color": "#DC2626", "max_damage": 95,  "icon": "⚡"},
    "Physical-Damage":   {"color": "#3B82F6", "max_damage": 90,  "icon": "💥"},
    "Snow-Covered":      {"color": "#06B6D4", "max_damage": 50,  "icon": "❄️"},
}


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_email" not in session:
        return send_from_directory(str(FRONTEND_DIR), "login.html")
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/login.html")
def login_page():
    return send_from_directory(str(FRONTEND_DIR), "login.html")


@app.route("/signup.html")
def signup_page():
    return send_from_directory(str(FRONTEND_DIR), "signup.html")


@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return send_from_directory(str(FRONTEND_DIR), "login.html")
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# Static file routes — explicit paths only, never catch API routes
@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(FRONTEND_DIR / "assets"), filename)


# ── Auth endpoints ──────────────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    data     = request.get_json()
    name     = (data.get("name") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required."}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if email in USERS:
        return jsonify({"error": "An account with this email already exists."}), 409

    USERS[email] = {
        "name":          name,
        "password_hash": generate_password_hash(password),
        "avatar":        None,
        "provider":      "email",
    }
    session["user_email"]  = email
    session["user_name"]   = name
    session["user_avatar"] = None
    return jsonify({"success": True, "name": name, "email": email}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = USERS.get(email)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_email"]  = email
    session["user_name"]   = user["name"]
    session["user_avatar"] = user.get("avatar")
    return jsonify({"success": True, "name": user["name"], "email": email})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me", methods=["GET"])
def me():
    if "user_email" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "email":  session["user_email"],
        "name":   session["user_name"],
        "avatar": session.get("user_avatar"),
    })


# ── Google OAuth routes ─────────────────────────────────────────────────────
@app.route("/api/auth/google")
def google_login():
    if not GOOGLE_CONFIGURED:
        return redirect("/login.html?error=google_not_configured")
    # Hardcode the callback URL to avoid Flask generating wrong host/port
    callback_url = "http://127.0.0.1:5000/api/auth/google/callback"
    return google.authorize_redirect(callback_url)


@app.route("/api/auth/google/callback")
def google_callback():
    if not GOOGLE_CONFIGURED:
        return redirect("/login.html?error=google_not_configured")
    try:
        # Authlib stores redirect_uri in session from authorize_redirect()
        # Do NOT pass it again here — it causes "multiple values" error
        token     = google.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            # Fallback: fetch from userinfo endpoint
            resp      = google.get("https://openidconnect.googleapis.com/v1/userinfo")
            user_info = resp.json()

        email  = user_info.get("email", "").lower()
        name   = user_info.get("name") or user_info.get("given_name") or email.split("@")[0]
        avatar = user_info.get("picture")

        if not email:
            return redirect("/login.html?error=no_email")

        # Create account if first time, otherwise update avatar
        if email not in USERS:
            USERS[email] = {
                "name":          name,
                "password_hash": "",        # no password for Google users
                "avatar":        avatar,
                "provider":      "google",
            }
        else:
            USERS[email]["avatar"]   = avatar
            USERS[email]["provider"] = "google"

        session["user_email"]  = email
        session["user_name"]   = name
        session["user_avatar"] = avatar

        return redirect("/")

    except Exception as e:
        print(f"[Google OAuth Error] {e}")
        return redirect(f"/login.html?error=oauth_failed")


@app.route("/api/health", methods=["GET"])
@login_required
def health():
    return jsonify({
        "status":   "ok",
        "device":   DEVICE,
        "model":    str(YOLO_WEIGHTS),
        "classes":  len(CLASSES),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route("/api/classes", methods=["GET"])
@login_required
def get_classes():
    result = []
    for cls in CLASSES:
        meta = CLASS_META.get(cls, {})
        diag = DIAGNOSIS.get(cls, {})
        result.append({
            "name":          cls,
            "color":         meta.get("color", "#94A3B8"),
            "max_damage":    meta.get("max_damage", 50),
            "icon":          meta.get("icon", "🔍"),
            "what_happened": diag.get("what_happened", ""),
            "impact":        diag.get("impact", []),
            "suggestions":   diag.get("suggestions", []),
        })
    return jsonify({"classes": result})


@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request. Use field name 'image'."}), 400

    file     = request.files["image"]
    conf_thr = float(request.form.get("conf", 0.25))
    conf_thr = max(0.01, min(0.95, conf_thr))

    # Decode image
    img_bytes = file.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image. Ensure it is a valid JPG/PNG."}), 400

    h, w = img_bgr.shape[:2]
    vis  = img_bgr.copy()

    # Write temp file for YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        detections_list = []
        damage_pcts     = []
        primary_class   = None

        # First pass at requested confidence
        results    = yolo_model.predict(source=tmp_path, conf=conf_thr,
                                        imgsz=640, device=DEVICE, verbose=False)
        detections = results[0].boxes

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
                    "color":      CLASS_META.get(class_name, {}).get("color", "#94A3B8"),
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
                    "color":      CLASS_META.get(class_name, {}).get("color", "#94A3B8"),
                    "bbox":       [0, 0, w - 1, h - 1],
                })
    finally:
        os.unlink(tmp_path)

    avg_damage = round(sum(damage_pcts) / len(damage_pcts), 1) if damage_pcts else 0.0
    primary    = primary_class or "Clean"

    # Draw summary bar on annotated image
    if damage_pcts:
        _draw_summary_panel(vis, avg_damage, primary)

    # ── Farm / multi-panel analysis ─────────────────────────────────────
    farm_data = analyse_farm_image(img_bgr, detections_list)

    # Use farm-annotated image (has grid overlay) if panels were found
    if farm_data["total_panels"] > 1:
        farm_vis = farm_data["annotated_farm"].copy()
        # Also draw YOLO boxes on top of grid overlay
        for det in detections_list:
            x1, y1, x2, y2 = det["bbox"]
            color = PALETTE[CLASSES.index(det["class"]) % len(PALETTE)] if det["class"] in CLASSES else (128,128,128)
            draw_box(farm_vis, x1, y1, x2, y2, det["class"], det["confidence"]/100, color)
        if damage_pcts:
            _draw_summary_panel(farm_vis, avg_damage, primary)
        _, buf = cv2.imencode(".jpg", farm_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    img_b64 = base64.b64encode(buf).decode("utf-8")

    # Diagnosis
    diag = DIAGNOSIS.get(primary, DIAGNOSIS["Clean"])

    return jsonify({
        "success":         True,
        "annotated_image": img_b64,
        "image_size":      {"width": w, "height": h},
        "primary_class":   primary,
        "primary_color":   CLASS_META.get(primary, {}).get("color", "#94A3B8"),
        "avg_damage":      avg_damage,
        "severity":        severity_label(avg_damage),
        "detection_count": len(detections_list),
        "detections":      farm_data["detections"],   # enriched with grid_label
        "diagnosis": {
            "what_happened": diag["what_happened"],
            "impact":        diag["impact"],
            "suggestions":   diag["suggestions"],
        },
        # Farm-specific fields
        "farm_mode":       farm_data["farm_mode"],
        "total_panels":    farm_data["total_panels"],
        "affected_panels": farm_data["affected_panels"],
        "grid_map_b64":    farm_data["grid_map_b64"],
        "panel_crops":     farm_data["panel_crops"],
        "grid_panels":     farm_data["grid_panels"],
    })

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
