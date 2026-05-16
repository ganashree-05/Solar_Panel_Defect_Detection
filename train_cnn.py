"""
train_cnn.py
------------
Trains a CNN classifier (ResNet-50 backbone, fine-tuned) on the solar-panel
defect dataset.  Produces:
  • cnn_best.pth          – best checkpoint
  • cnn_training_curves.png
  • cnn_confusion_matrix.png
  • cnn_classification_report.txt

Run:
    python train_cnn.py
"""

import os
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Config ─────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(".")
TRAIN_DIR   = DATA_ROOT / "train"
VAL_DIR     = DATA_ROOT / "val"
TEST_DIR    = DATA_ROOT / "test"

CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]
NUM_CLASSES = len(CLASSES)

IMG_SIZE    = 224
BATCH       = 32              # RTX 3050 6GB can handle 32 at 224px
EPOCHS      = 50
LR          = 1e-4
WEIGHT_DECAY= 1e-4
PATIENCE    = 10          # early-stopping
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "cnn_best.pth"

# ── Transforms ─────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Datasets & Loaders ─────────────────────────────────────────────────────
def build_loaders():
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=val_tf)
    test_ds  = datasets.ImageFolder(str(TEST_DIR),  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Class mapping: {train_ds.class_to_idx}")
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# ── Model ──────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Unfreeze last 2 residual blocks + FC for fine-tuning
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    return model.to(DEVICE)


# ── Training loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += imgs.size(0)
    return running_loss / total, correct / total


# ── Plots ──────────────────────────────────────────────────────────────────
def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    plt.close()
    print("Saved cnn_training_curves.png")


def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved cnn_confusion_matrix.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader, class_to_idx = build_loaders()

    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    best_weights  = None
    no_improve    = 0

    print(f"\nTraining ResNet-50 for {EPOCHS} epochs …\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, SAVE_PATH)
            print(f"  ✓ New best val_acc={best_val_acc:.4f} → saved {SAVE_PATH}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\nLoading best weights for test evaluation …")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Sort class names by index
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(NUM_CLASSES)]

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n── Classification Report (Test) ──")
    print(report)

    with open("cnn_classification_report.txt", "w") as f:
        f.write(report)
    print("Saved cnn_classification_report.txt")

    plot_curves(history)
    plot_confusion(all_labels, all_preds, class_names)

    print(f"\nBest Val Accuracy : {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
