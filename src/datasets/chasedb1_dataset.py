from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


VALID_SPLITS = {"train_pool", "train", "val", "test"}


def _normalize_size(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return (input_size, input_size)
    return tuple(input_size)


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


def _load_fov_array(path: Path, size: tuple[int, int], threshold: int) -> np.ndarray:
    gray = Image.open(path).convert("L")
    array = (np.asarray(gray, dtype=np.uint8) > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(array, mode="L")
    mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    fov = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return np.expand_dims(fov, axis=0)


def _split_indices(total: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to split CHASEDB1 train/val samples.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    val_size = max(1, int(round(total * val_ratio)))
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Check train_count and val_ratio.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    return indices[:train_size], indices[train_size:]


class CHASEDB1Dataset(Dataset):
    """CHASEDB1 retinal vessel dataset.

    Images are sorted by filename. The first `train_count` images form the
    train pool, and the remaining images form the final test split.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        input_size: int | tuple[int, int] = 512,
        annotation: str = "1stHO",
        train_count: int = 20,
        val_ratio: float = 0.2,
        seed: int = 42,
        fov_threshold: int = 10,
        require_gt: bool = True,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported CHASEDB1 split: {split}. Expected one of {sorted(VALID_SPLITS)}.")

        self.root = Path(root)
        self.split = split
        self.annotation = annotation
        self.train_count = int(train_count)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.fov_threshold = int(fov_threshold)
        self.size = _normalize_size(input_size)

        if not self.root.exists():
            raise FileNotFoundError(f"CHASEDB1 root not found: {self.root}")

        images = sorted(path for path in self.root.glob("*.jpg") if path.is_file())
        if len(images) <= self.train_count:
            raise ValueError(
                f"CHASEDB1 needs more images than train_count. image_count={len(images)}, train_count={self.train_count}"
            )

        train_pool = images[: self.train_count]
        test_images = images[self.train_count :]
        if split == "train_pool":
            selected = train_pool
        elif split == "test":
            selected = test_images
        else:
            train_indices, val_indices = _split_indices(len(train_pool), self.val_ratio, self.seed)
            indices = train_indices if split == "train" else val_indices
            selected = [train_pool[index] for index in indices]

        self.samples: list[dict[str, Any]] = []
        missing_vessels: list[str] = []
        for image_path in selected:
            vessel_path = image_path.with_name(f"{image_path.stem}_{self.annotation}.png")
            has_gt = vessel_path.exists()
            if require_gt and not has_gt:
                missing_vessels.append(vessel_path.name)
            self.samples.append(
                {
                    "id": image_path.stem,
                    "image": image_path,
                    "vessel_mask": vessel_path if has_gt else None,
                    "has_gt": has_gt,
                }
            )

        if missing_vessels:
            raise ValueError(
                "Some CHASEDB1 images are missing vessel ground truth. "
                f"annotation={self.annotation}, missing_vessel_masks={missing_vessels}"
            )
        if not self.samples:
            raise ValueError(f"No CHASEDB1 samples found for split={split}")

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
            raise ModuleNotFoundError("torch is required to use CHASEDB1Dataset with a PyTorch DataLoader.")

        sample = self.samples[index]
        image = torch.from_numpy(_load_rgb_array(sample["image"], self.size))
        vessel_mask = (
            torch.from_numpy(_load_binary_array(sample["vessel_mask"], self.size))
            if sample["vessel_mask"] is not None
            else None
        )
        fov_mask = torch.from_numpy(_load_fov_array(sample["image"], self.size, self.fov_threshold))
        return image, vessel_mask, fov_mask, sample["image"].name, bool(sample["has_gt"])

    def load_numpy_sample(self, index: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str, bool]:
        sample = self.samples[index]
        image = _load_rgb_array(sample["image"], self.size)
        vessel_mask = (
            _load_binary_array(sample["vessel_mask"], self.size)
            if sample["vessel_mask"] is not None
            else None
        )
        fov_mask = _load_fov_array(sample["image"], self.size, self.fov_threshold)
        return image, vessel_mask, fov_mask, sample["image"].name, bool(sample["has_gt"])


def build_dataset_from_config(config: dict[str, Any], split: str) -> CHASEDB1Dataset:
    return CHASEDB1Dataset(
        root=config["chasedb1_root"],
        split=split,
        input_size=config["input_size"],
        annotation=str(config.get("annotation", "1stHO")),
        train_count=int(config.get("train_count", 20)),
        val_ratio=float(config.get("val_ratio", 0.2)),
        seed=int(config.get("seed", 42)),
        fov_threshold=int(config.get("fov_threshold", 10)),
        require_gt=True,
    )


def save_dataset_check_figure(
    image: np.ndarray,
    vessel_mask: np.ndarray | None,
    fov_mask: np.ndarray,
    filename: str,
    split: str,
    output_dir: str | Path = "outputs/visualizations/chasedb1_dataset_check",
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
    parser = argparse.ArgumentParser(description="Check CHASEDB1 dataset loading and mask pairing.")
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), required=True, help="Dataset split to inspect.")
    parser.add_argument("--config", default="configs/unet_chasedb1.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    from src.utils import load_config

    args = parse_args()
    config = load_config(args.config)
    dataset = build_dataset_from_config(config, args.split)

    image, vessel_mask, fov_mask, filename, has_gt = dataset.load_numpy_sample(0)
    check_path = save_dataset_check_figure(image, vessel_mask, fov_mask, filename, args.split)

    print(f"split: {args.split}")
    print(f"image count: {len(dataset)}")
    print(f"vessel GT count: {dataset.vessel_gt_count}")
    print(f"FOV mask count: {dataset.fov_mask_count}")
    print(f"first filename: {filename}")
    print(f"has_gt: {has_gt}")
    print(
        f"image: shape={image.shape}, dtype={image.dtype}, "
        f"min={image.min():.4f}, max={image.max():.4f}"
    )
    print(_array_info("vessel mask", vessel_mask))
    print(_array_info("fov mask", fov_mask))
    print(f"saved visualization: {check_path}")


if __name__ == "__main__":
    main()
