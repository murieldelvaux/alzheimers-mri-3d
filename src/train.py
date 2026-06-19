from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets.mri_dataset import MRIDataset
from src.models.cnn_3d import SimpleResNet3D
from src.utils.helpers import load_config, get_device
from src.data_prep.preprocess import build_preprocessing_pipeline


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg["training"]["device"])

    transform = build_preprocessing_pipeline(cfg, train=True)
    train_ds = MRIDataset(cfg["data"]["unified_metadata"], split="train", transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=False,
    )

    model = SimpleResNet3D(
        in_channels=cfg["model"]["input_channels"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    class_counts = torch.tensor([1005.0, 85.0, 285.0])  # CN=0, MCI=1, DEM(AD)=2
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # normaliza
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    best_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) else 0.0
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - loss: {epoch_loss:.4f}")

        # Save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            checkpoints_dir / f"checkpoint_epoch{epoch+1:03d}.pt",
        )

        # Save best model separately (used by inference.py)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                checkpoints_dir / "best_model.pth",
            )
            print(f"  ✓ best model saved (loss={best_loss:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a 3D CNN model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
