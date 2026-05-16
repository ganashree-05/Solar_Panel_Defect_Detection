# SolarScan – Solar Panel Defect Detection

AI-powered solar panel defect detection system using **YOLOv8** + **ResNet-50 CNN** with **GradCAM-based precise bounding boxes**.

![Dashboard](https://img.shields.io/badge/Dashboard-Live-7C3AED?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-F59E0B?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge)

---

## Features

- **6 Defect Classes**: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered
- **Precise Bounding Boxes**: GradCAM-guided labels pinpoint exact damage location (not whole-image)
- **Damage Percentage**: Confidence scores mapped to 0–100% damage severity
- **Actionable Suggestions**: Step-by-step repair recommendations per defect type
- **Modern Dashboard**: Purple-themed UI with charts, stats, and real-time analysis
- **GPU Accelerated**: RTX 3050 6GB training in ~2–3 hours

---

## Architecture

### YOLOv8m (Detection)
- **Backbone**: CSPDarknet CNN (pretrained on COCO)
- **Neck**: PANet feature pyramid
- **Head**: Decoupled detection → bbox regression + classification
- **Input**: 640 × 640
- **Output**: Tight bounding boxes with class + confidence

### ResNet-50 (Classification)
- **Pretrained**: ImageNet (IMAGENET1K_V2)
- **Fine-tuned**: layer3, layer4, custom FC head
- **Input**: 224 × 224
- **Output**: 6-class softmax probabilities
- **Used for**: GradCAM label generation (finds exact damage location)

### GradCAM Label Generation
- Computes gradient-weighted activation maps on ResNet-50's last conv layer
- Thresholds the heatmap to isolate high-activation regions
- Fits tight bounding boxes around damage areas
- **Result**: Labels cover 27–49% of image (vs 100% whole-image boxes)

---

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| Train | 929    | 929    |
| Val   | 550    | 550    |
| Test  | 95     | 95     |

**Total**: 1,574 images across 6 classes

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python pillow numpy matplotlib seaborn scikit-learn flask flask-cors
```

### 2. Generate GradCAM labels
```bash
python generate_labels.py
```

### 3. Train models

**YOLOv8 (bounding boxes):**
```bash
python train_yolo.py
```
- 120 epochs, RTX 3050 6GB: ~2–3 hours
- Best weights → `runs/detect/runs/solar_precise/weights/best.pt`

**CNN classifier (optional):**
```bash
python train_cnn.py
```
- 50 epochs: ~30–40 min on GPU
- Best weights → `cnn_best.pth`

### 4. Run the dashboard
```bash
python backend/api.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## Project Structure

```
project/
├── backend/
│   └── api.py              # Flask REST API (JSON endpoints)
│
├── frontend/
│   ├── index.html          # Dashboard UI (purple theme)
│   ├── css/style.css       # All styles
│   └── js/
│       ├── config.js       # API base URL
│       └── app.js          # Charts, navigation, API calls
│
├── predict.py              # Core detection + damage % logic
├── train_yolo.py           # YOLOv8 training script
├── train_cnn.py            # ResNet-50 training script
├── generate_labels.py      # GradCAM-based label generator
├── visualise_labels.py     # Preview GradCAM boxes
├── dataset.yaml            # YOLO dataset config
└── README.md
```

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/health` | Model status, GPU info |
| GET | `/api/classes` | All 6 defect classes + metadata |
| POST | `/api/predict` | Upload image → detection + diagnosis JSON |

---

## Damage Percentage Formula

```
damage% = class_weight × yolo_confidence × 100
```

| Class | Weight | Max Damage |
|-------|--------|------------|
| Clean | 0.00 | 0% |
| Dusty | 0.35 | 35% |
| Bird-drop | 0.60 | 60% |
| Snow-Covered | 0.50 | 50% |
| Physical-Damage | 0.90 | 90% |
| Electrical-damage | 0.95 | 95% |

**Severity Scale:**
- `0%` → No Damage
- `<20%` → Minimal
- `<40%` → Low
- `<60%` → Moderate
- `<80%` → High
- `≥80%` → **Critical**

---

## Output Examples

Each detection includes:
- **Bounding box** around the damaged region (not whole image)
- **Class label** + confidence %
- **Damage percentage** + severity badge
- **Colour-coded severity bar**
- **Bottom summary panel** with overall damage
- **Diagnosis sidebar** (what happened / impact / how to fix)

---

## Training Results

### YOLOv8m (Precise Boxes)
- **mAP50**: ~0.85–0.92 (after 120 epochs)
- **Precision**: ~0.88
- **Recall**: ~0.84
- **Box accuracy**: Tight boxes around damage (27–49% of image area)

### ResNet-50 CNN
- **Test Accuracy**: ~92–95%
- **Per-class F1**: 0.90–0.97
- **Confusion matrix**: `cnn_confusion_matrix.png`

---

## Requirements

- Python 3.8+
- PyTorch 2.11+ with CUDA 12.8 (for GPU)
- Ultralytics 8.4+
- OpenCV, Pillow, NumPy, Matplotlib, Seaborn, scikit-learn
- Flask, flask-cors

---

## License

MIT

---

## Credits

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **ResNet-50**: [torchvision](https://pytorch.org/vision/stable/models.html)
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **Dashboard Design**: Inspired by modern fintech UIs
