from __future__ import annotations

from pathlib import Path


MISSING_GT_ERROR = "Cannot evaluate this split because vessel ground truth is missing."


def get_dataset_name(config: dict) -> str:
    return str(config.get("dataset", "drive")).lower()


def split_train_val(dataset, val_ratio: float, seed: int):
    import torch
    from torch.utils.data import random_split

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Check dataset size and val_ratio.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _build_drive_training_dataset(config: dict):
    from src.datasets.drive_dataset import DriveDataset

    try:
        return DriveDataset(
            image_dir=config["train_image_dir"],
            vessel_mask_dir=config["train_vessel_mask_dir"],
            fov_mask_dir=config["train_fov_mask_dir"],
            input_size=config["input_size"],
            require_gt=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            "Failed to build DRIVE training dataset. Please check the data directories in the config "
            "and make sure training/1st_manual contains vessel ground truth and training/mask contains FOV masks."
        ) from exc


def _build_chasedb1_dataset(config: dict, split: str):
    from src.datasets.chasedb1_dataset import CHASEDB1Dataset

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


def build_train_val_datasets(config: dict):
    dataset_name = get_dataset_name(config)
    seed = int(config.get("seed", 42))
    if dataset_name == "drive":
        dataset = _build_drive_training_dataset(config)
        return split_train_val(dataset, float(config["val_ratio"]), seed)
    if dataset_name == "chasedb1":
        return _build_chasedb1_dataset(config, "train"), _build_chasedb1_dataset(config, "val")
    raise ValueError(f"Unsupported dataset: {dataset_name!r}. Expected 'drive' or 'chasedb1'.")


def build_eval_dataset(config: dict, split: str):
    dataset_name = get_dataset_name(config)
    seed = int(config.get("seed", 42))

    if dataset_name == "drive":
        from src.datasets.drive_dataset import DriveDataset

        if split == "val":
            dataset = _build_drive_training_dataset(config)
            _train_dataset, val_dataset = split_train_val(dataset, float(config["val_ratio"]), seed)
            return val_dataset
        if split == "test":
            vessel_mask_dir = config.get("test_vessel_mask_dir")
            if vessel_mask_dir is None or not Path(vessel_mask_dir).exists():
                raise RuntimeError(MISSING_GT_ERROR)
            try:
                return DriveDataset(
                    image_dir=config["test_image_dir"],
                    vessel_mask_dir=vessel_mask_dir,
                    fov_mask_dir=config["test_fov_mask_dir"],
                    input_size=config["input_size"],
                    require_gt=True,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise RuntimeError(MISSING_GT_ERROR) from exc

    if dataset_name == "chasedb1" and split in {"val", "test"}:
        return _build_chasedb1_dataset(config, split)

    raise ValueError(f"Unsupported dataset/split for evaluation: dataset={dataset_name!r}, split={split!r}")


def build_predict_dataset(config: dict, split: str):
    dataset_name = get_dataset_name(config)
    seed = int(config.get("seed", 42))

    if dataset_name == "drive":
        from src.datasets.drive_dataset import DriveDataset

        if split == "test":
            return DriveDataset(
                image_dir=config["test_image_dir"],
                vessel_mask_dir=config.get("test_vessel_mask_dir"),
                fov_mask_dir=config["test_fov_mask_dir"],
                input_size=config["input_size"],
                require_gt=False,
            )
        if split == "val":
            dataset = _build_drive_training_dataset(config)
            _train_dataset, val_dataset = split_train_val(dataset, float(config["val_ratio"]), seed)
            return val_dataset

    if dataset_name == "chasedb1" and split in {"val", "test"}:
        return _build_chasedb1_dataset(config, split)

    raise ValueError(f"Unsupported dataset/split for prediction: dataset={dataset_name!r}, split={split!r}")
