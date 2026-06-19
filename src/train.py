from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

from src.datasets.mri_dataset import MRIDataset
from src.models.cnn_3d import SimpleResNet3D
from src.utils.helpers import load_config, get_device
from src.data_prep.preprocess import build_preprocessing_pipeline
from src.utils.focal_loss import FocalLoss


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg["training"]["device"])

    train_transform = build_preprocessing_pipeline(cfg, train=True)
    val_transform   = build_preprocessing_pipeline(cfg, train=False)

    train_ds = MRIDataset(cfg["data"]["unified_metadata"], split="train", transform=train_transform)
    val_ds   = MRIDataset(cfg["data"]["unified_metadata"], split="val",   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader   = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = SimpleResNet3D(
        in_channels=cfg["model"]["input_channels"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    # Pesos suaves — não tão extremos quanto antes
    class_weights = torch.tensor([1.0, 5.0, 3.0]).to(device)
    print(f"Class weights: CN={class_weights[0]:.3f}  MCI={class_weights[1]:.3f}  DEM={class_weights[2]:.3f}")

    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    class_names = ["CN", "MCI", "DEM"]

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels_b = batch["label"].to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels_b).item() * images.size(0)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels_b.cpu().numpy())

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")
        print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }, checkpoints_dir / "best_model.pth")
            print(f"  ✓ best model saved (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
