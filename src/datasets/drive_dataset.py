from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def _file_id(path: Path) -> str | None:
    match = re.match(r"^(\d{2})", path.name)
    return match.group(1) if match else None


def _is_missing_path(path: str | Path | None) -> bool:
    return path is None or str(path).strip().lower() in {"", "none", "null"}


def _index_by_id(directory: str | Path | None, required: bool) -> dict[str, Path]:
    if _is_missing_path(directory):
        if required:
            raise FileNotFoundError("Required data directory is not configured.")
        return {}

    directory_path = Path(directory)
    if not directory_path.exists():
        if required:
            raise FileNotFoundError(f"Data directory not found: {directory_path}")
        return {}

    result: dict[str, Path] = {}
    for path in sorted(directory_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        item_id = _file_id(path)
        if item_id is None:
            continue
        result[item_id] = path
    return result


def _load_rgb_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def _load_binary_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return np.expand_dims(array, axis=0)


class DriveDataset(Dataset):
    """DRIVE dataset.

    `vessel_mask_dir` contains vessel ground truth and may be missing for the
    official DRIVE test split. `fov_mask_dir` contains field-of-view masks and
    must never be treated as vessel ground truth.
    """

    def __init__(
        self,
        image_dir: str | Path,
        vessel_mask_dir: str | Path | None,
        fov_mask_dir: str | Path,
        input_size: int | tuple[int, int] = 512,
        require_gt: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.vessel_mask_dir = None if _is_missing_path(vessel_mask_dir) else Path(vessel_mask_dir)  # type: ignore[arg-type]
        self.fov_mask_dir = Path(fov_mask_dir)

        if isinstance(input_size, int):
            self.size = (input_size, input_size)
        else:
            self.size = tuple(input_size)

        images = _index_by_id(self.image_dir, required=True)
        vessel_masks = _index_by_id(self.vessel_mask_dir, required=False)
        fov_masks = _index_by_id(self.fov_mask_dir, required=True)

        missing_fovs = sorted(set(images) - set(fov_masks))
        missing_vessels = sorted(set(images) - set(vessel_masks)) if self.vessel_mask_dir is not None else []

        if not images:
            raise ValueError(f"No DRIVE images found in {self.image_dir}")
        if missing_fovs:
            raise ValueError(
                "Some DRIVE images are missing FOV masks. "
                f"missing_fov_masks={missing_fovs}. FOV masks must be placed under {self.fov_mask_dir}."
            )
        if require_gt and (self.vessel_mask_dir is None or missing_vessels):
            raise ValueError(
                "Vessel ground truth is required for this split, but it is missing. "
                f"vessel_mask_dir={self.vessel_mask_dir}, missing_vessel_masks={missing_vessels}"
            )

        self.samples: list[dict[str, Any]] = []
        for item_id in sorted(images):
            vessel_path = vessel_masks.get(item_id)
            self.samples.append(
                {
                    "id": item_id,
                    "image": images[item_id],
                    "vessel_mask": vessel_path,
                    "fov_mask": fov_masks[item_id],
                    "has_gt": vessel_path is not None,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def vessel_gt_count(self) -> int:
        return sum(1 for sample in self.samples if sample["has_gt"])

    @property
    def fov_mask_count(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if torch is None:
            raise ModuleNotFoundError("torch is required to use DriveDataset with a PyTorch DataLoader.")

        sample = self.samples[index]
        image = torch.from_numpy(_load_rgb_array(sample["image"], self.size))
        vessel_mask = (
            torch.from_numpy(_load_binary_array(sample["vessel_mask"], self.size))
            if sample["vessel_mask"] is not None
            else None
        )
        fov_mask = torch.from_numpy(_load_binary_array(sample["fov_mask"], self.size))
        return image, vessel_mask, fov_mask, sample["image"].name, bool(sample["has_gt"])

    def load_numpy_sample(self, index: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str, bool]:
        sample = self.samples[index]
        image = _load_rgb_array(sample["image"], self.size)
        vessel_mask = (
            _load_binary_array(sample["vessel_mask"], self.size)
            if sample["vessel_mask"] is not None
            else None
        )
        fov_mask = _load_binary_array(sample["fov_mask"], self.size)
        return image, vessel_mask, fov_mask, sample["image"].name, bool(sample["has_gt"])


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        if yaml is not None:
            config = yaml.safe_load(f)
        else:
            config = _load_simple_yaml(f.read())
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def _load_simple_yaml(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("PyYAML is not installed and fallback parser only supports simple key: value lines.")
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"null", "none"}:
            parsed_value: Any = None
        elif value.lower() in {"true", "false"}:
            parsed_value = value.lower() == "true"
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value.strip("\"'")
        config[key.strip()] = parsed_value
    return config


def build_dataset_from_config(config: dict[str, Any], split: str, require_gt: bool = False) -> DriveDataset:
    if split == "training":
        return DriveDataset(
            image_dir=config["train_image_dir"],
            vessel_mask_dir=config.get("train_vessel_mask_dir"),
            fov_mask_dir=config["train_fov_mask_dir"],
            input_size=config["input_size"],
            require_gt=require_gt,
        )
    if split == "test":
        return DriveDataset(
            image_dir=config["test_image_dir"],
            vessel_mask_dir=config.get("test_vessel_mask_dir"),
            fov_mask_dir=config["test_fov_mask_dir"],
            input_size=config["input_size"],
            require_gt=require_gt,
        )
    raise ValueError(f"Unsupported split: {split}")


def save_dataset_check_figure(
    image: np.ndarray,
    vessel_mask: np.ndarray | None,
    fov_mask: np.ndarray,
    filename: str,
    split: str,
    output_dir: str | Path = "outputs/visualizations/dataset_check",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_hwc = np.transpose(image, (1, 2, 0))
    image_uint8 = np.clip(image_hwc * 255.0, 0, 255).astype(np.uint8)
    panels = [Image.fromarray(image_uint8)]
    labels = ["image"]

    if vessel_mask is not None:
        vessel_uint8 = (vessel_mask[0] * 255).astype(np.uint8)
        panels.append(Image.fromarray(vessel_uint8, mode="L").convert("RGB"))
        labels.append("vessel GT")

    fov_uint8 = (fov_mask[0] * 255).astype(np.uint8)
    panels.append(Image.fromarray(fov_uint8, mode="L").convert("RGB"))
    labels.append("FOV mask")

    width, height = panels[0].size
    label_height = 24
    canvas = Image.new("RGB", (width * len(panels), height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        x = idx * width
        canvas.paste(panel, (x, label_height))
        draw.text((x + 8, 5), label, fill=(0, 0, 0))

    save_path = output_path / f"{split}_{Path(filename).stem}_check.png"
    canvas.save(save_path)
    return save_path


def _array_info(name: str, array: np.ndarray | None) -> str:
    if array is None:
        return f"{name}: no vessel GT"
    unique_values = np.unique(array).astype(int).tolist()
    return (
        f"{name}: shape={array.shape}, dtype={array.dtype}, "
        f"min={array.min():.4f}, max={array.max():.4f}, unique={unique_values}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DRIVE dataset loading and mask pairing.")
    parser.add_argument("--split", choices=["training", "test"], required=True, help="Dataset split to inspect.")
    parser.add_argument("--config", default="configs/unet_drive.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = build_dataset_from_config(config, args.split, require_gt=False)

    image, vessel_mask, fov_mask, filename, has_gt = dataset.load_numpy_sample(0)
    check_path = save_dataset_check_figure(image, vessel_mask, fov_mask, filename, args.split)

    vessel_gt_text = str(dataset.vessel_gt_count) if dataset.vessel_gt_count else "missing"
    print(f"split: {args.split}")
    print(f"image count: {len(dataset)}")
    print(f"vessel GT count: {vessel_gt_text}")
    print(f"FOV mask count: {dataset.fov_mask_count}")
    print(f"first filename: {filename}")
    print(f"has_gt: {has_gt}")
    print(
        f"image: shape={image.shape}, dtype={image.dtype}, "
        f"min={image.min():.4f}, max={image.max():.4f}"
    )
    if vessel_mask is None:
        print("vessel mask: no vessel GT")
    else:
        print(_array_info("vessel mask", vessel_mask))
    print(_array_info("fov mask", fov_mask))
    print(f"saved visualization: {check_path}")


if __name__ == "__main__":
    main()
