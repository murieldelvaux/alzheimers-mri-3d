"""Inference pipeline — loads a trained checkpoint and runs prediction on a NIfTI file.

This module is the bridge between the trained model and the FastAPI backend.
Usage::

    from src.inference import load_model, predict_from_nifti

    model, cfg, device = load_model("checkpoints/best_model.pt")
    result = predict_from_nifti("path/to/scan.nii.gz", model, cfg, device)
    # result = {
    #   "classification": "MCI",
    #   "probabilities": {"CN": 0.12, "MCI": 0.71, "AD": 0.17},
    #   "confidence": 0.71
    # }
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data_prep.preprocess import build_preprocessing_pipeline
from src.models.cnn_3d import SimpleResNet3D
from src.utils.helpers import get_device, load_config

if TYPE_CHECKING:
    pass

CLASS_NAMES: list[str] = ["CN", "MCI", "AD"]
CLASS_LABELS: dict[str, str] = {
    "CN": "Cognitive Normal",
    "MCI": "Mild Cognitive Impairment",
    "AD": "Alzheimer's Disease",
}


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path = "config.yaml",
    force_cpu: bool = True,
) -> tuple[SimpleResNet3D, dict, torch.device]:
    """Load a trained SimpleResNet3D from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint saved during training.
        config_path: Path to config.yaml (same one used during training).
        force_cpu: When True, always loads on CPU (recommended for API serving).

    Returns:
        Tuple of (model, config_dict, device).
    """
    cfg = load_config(config_path)
    device = torch.device("cpu") if force_cpu else get_device(cfg["training"]["device"])

    model = SimpleResNet3D(
        in_channels=cfg["model"]["input_channels"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and full checkpoint dict
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    return model, cfg, device


def predict_from_nifti(
    nifti_path: str | Path,
    model: SimpleResNet3D,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run inference on a single NIfTI MRI file.

    Args:
        nifti_path: Path to the .nii or .nii.gz file.
        model: Loaded and eval()-mode SimpleResNet3D.
        cfg: Config dict from load_config().
        device: torch.device to run inference on.

    Returns:
        dict with keys:
            - classification (str): Predicted class label ("CN", "MCI", or "AD").
            - label_full (str): Human-readable label.
            - probabilities (dict): Per-class softmax probabilities.
            - confidence (float): Probability of the predicted class.
    """
    transform = build_preprocessing_pipeline(cfg, train=False)
    data = transform({"image": str(nifti_path)})
    image: torch.Tensor = data["image"].unsqueeze(0).to(device)  # (1, C, D, H, W)

    with torch.no_grad():
        logits = model(image)  # (1, num_classes)
        probs: np.ndarray = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    return {
        "classification": pred_class,
        "label_full": CLASS_LABELS[pred_class],
        "probabilities": {
            name: round(float(probs[i]), 4)
            for i, name in enumerate(CLASS_NAMES)
        },
        "confidence": round(float(probs[pred_idx]), 4),
    }


def predict_batch(
    nifti_paths: list[str | Path],
    model: SimpleResNet3D,
    cfg: dict,
    device: torch.device,
) -> list[dict]:
    """Convenience wrapper to run predict_from_nifti on multiple files."""
    return [predict_from_nifti(p, model, cfg, device) for p in nifti_paths]
