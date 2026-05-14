from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on the DRIVE retinal vessel dataset.")
    parser.add_argument("--config", default="configs/unet_drive.yaml", help="Path to YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override Adam weight decay.")
    parser.add_argument("--input_size", type=int, default=None, help="Override square input image size.")
    parser.add_argument("--threshold", type=float, default=None, help="Override probability threshold for metrics.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", default=None, help="Override device, for example cpu, cuda, cuda:0, or mps.")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    override_map = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "input_size": args.input_size,
        "threshold": args.threshold,
        "output_dir": args.output_dir,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
    }
    for key, value in override_map.items():
        if value is not None:
            config[key] = value

    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    config.setdefault("num_workers", 0)
    return config


def print_effective_config(config: dict[str, Any]) -> None:
    keys = [
        "model",
        "input_size",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "val_ratio",
        "seed",
        "device",
        "output_dir",
    ]
    print("Effective training config:")
    for key in keys:
        print(f"  {key}: {config.get(key)}")


def save_config_used(config: dict[str, Any], path: Path) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def get_default_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_train_dataset(config: dict):
    from src.datasets import DriveDataset

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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    threshold: float,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    from src.metrics import compute_binary_metrics

    model.train()
    running_loss = 0.0

    for images, masks, _fovs, _filenames, has_gt in tqdm(loader, desc="train", leave=False):
        if hasattr(has_gt, "all"):
            has_all_gt = bool(has_gt.all().item())
        else:
            has_all_gt = all(has_gt)
        if not has_all_gt:
            raise RuntimeError("Training batch contains a sample without vessel ground truth.")
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

    num_samples = len(loader.dataset)
    return {"loss": running_loss / num_samples}


def validate_one_epoch(model, loader, device, threshold: float) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    from src.metrics import compute_binary_metrics

    model.eval()
    metric_sums: dict[str, float] = {}
    with torch.no_grad():
        for images, masks, fovs, _filenames, has_gt in tqdm(loader, desc="val", leave=False):
            if hasattr(has_gt, "all"):
                has_all_gt = bool(has_gt.all().item())
            else:
                has_all_gt = all(has_gt)
            if not has_all_gt:
                raise RuntimeError("Validation batch contains a sample without vessel ground truth.")
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            batch_metrics = compute_binary_metrics(probs, masks, threshold=threshold, fov_mask=fovs)
            batch_size = images.size(0)
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size

    num_samples = len(loader.dataset)
    return {key: value / num_samples for key, value in metric_sums.items()}


def main() -> None:
    args = parse_args()

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.losses import BCEDiceLoss
    from src.models import UNet
    from src.utils import ensure_dir, load_config, save_checkpoint, set_seed

    config = apply_cli_overrides(load_config(args.config), args)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    if config.get("model") != "unet":
        raise ValueError(f"Only model: unet is supported in this stage, got {config.get('model')!r}")

    dataset = build_train_dataset(config)
    train_dataset, val_dataset = split_train_val(dataset, float(config["val_ratio"]), seed)
    print(f"Train/val split: {len(train_dataset)}/{len(val_dataset)}")

    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    output_dir = Path(config["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "visualizations")

    device_name = config.get("device") or get_default_device_name()
    config["device"] = device_name
    device = torch.device(device_name)
    print_effective_config(config)
    save_config_used(config, output_dir / "config_used.yaml")

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config["loss_bce_weight"]),
        dice_weight=float(config["loss_dice_weight"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_dice = -1.0
    rows = []
    epochs = int(config["epochs"])
    threshold = float(config["threshold"])

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, threshold)
        val_metrics = validate_one_epoch(model, val_loader, device, threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)

        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "config": config,
                },
                checkpoint_dir / "best.pth",
            )

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": config,
            },
            checkpoint_dir / "last.pth",
        )

        pd.DataFrame(rows).to_csv(log_dir / "history.csv", index=False)
        print(
            f"epoch={epoch:03d}/{epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )


if __name__ == "__main__":
    main()
