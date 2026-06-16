from __future__ import annotations

from pathlib import Path
import yaml

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd,
    ToTensord,
)


def build_preprocessing_pipeline(config: dict, train: bool = False):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=tuple(config["preprocessing"]["spacing"]), mode=("bilinear",)),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config["preprocessing"]["intensity_range"][0],
                a_max=config["preprocessing"]["intensity_range"][1],
                b_min=config["preprocessing"]["normalized_range"][0],
                b_max=config["preprocessing"]["normalized_range"][1],
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=tuple(config["preprocessing"]["spatial_size"])),
            ToTensord(keys=["image"]),
        ]
    )


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
