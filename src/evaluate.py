from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.mri_dataset import MRIDataset
from src.models.cnn_3d import SimpleResNet3D
from src.utils.helpers import load_config, get_device
from src.data_prep.preprocess import build_preprocessing_pipeline


def evaluate(config_path: str, split: str = "val") -> None:
    cfg = load_config(config_path)
    device = get_device(cfg["training"]["device"])

    transform = build_preprocessing_pipeline(cfg, train=False)
    dataset = MRIDataset(cfg["data"]["unified_metadata"], split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"], pin_memory=False)

    model = SimpleResNet3D(
        in_channels=cfg["model"]["input_channels"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0
    print(f"{split.title()} accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a 3D CNN model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.config, split=args.split)
