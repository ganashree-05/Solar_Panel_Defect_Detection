import csv
from pathlib import Path

models = [
    ("solar_precise3",    "runs/detect/runs/solar_precise3/results.csv"),
    ("solar_precise2",    "runs/detect/runs/solar_precise2/results.csv"),
    ("solar_precise",     "runs/detect/runs/solar_precise/results.csv"),
    ("solar_panel_yolo2", "runs/detect/runs/solar_panel_yolo2/results.csv"),
    ("solar_panel_yolo",  "runs/detect/runs/solar_panel_yolo/results.csv"),
]

print(f"{'Model':<22} {'Epochs':>6} {'mAP50':>7} {'mAP50-95':>9} {'Precision':>10} {'Recall':>8}")
print("-" * 65)
for name, csv_path in models:
    try:
        rows = list(csv.DictReader(open(csv_path)))
        best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"].strip()))
        print(f"{name:<22} {len(rows):>6} "
              f"{float(best['metrics/mAP50(B)'].strip()):>7.4f} "
              f"{float(best['metrics/mAP50-95(B)'].strip()):>9.4f} "
              f"{float(best['metrics/precision(B)'].strip()):>10.4f} "
              f"{float(best['metrics/recall(B)'].strip()):>8.4f}")
    except Exception as e:
        print(f"{name:<22}  error: {e}")
