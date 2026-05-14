from __future__ import annotations

import argparse
import json
from pathlib import Path


MISSING_GT_ERROR = "Cannot evaluate this split because vessel ground truth is missing."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U-Net checkpoint on DRIVE val or test split.")
    parser.add_argument("--config", default="configs/unet_drive.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Split to evaluate. Default uses the validation subset from DRIVE training images.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for metrics.csv and metrics.json. Defaults to <config output_dir>/eval.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override probability threshold for metrics.")
    parser.add_argument("--device", default=None, help="Override device, for example cpu, cuda, cuda:0, or mps.")
    return parser.parse_args()


def build_validation_dataset(config: dict, seed: int):
    from src.datasets.drive_dataset import DriveDataset
    from src.train import split_train_val

    dataset = DriveDataset(
        image_dir=config["train_image_dir"],
        vessel_mask_dir=config["train_vessel_mask_dir"],
        fov_mask_dir=config["train_fov_mask_dir"],
        input_size=config["input_size"],
        require_gt=True,
    )
    _train_dataset, val_dataset = split_train_val(dataset, float(config["val_ratio"]), seed)
    return val_dataset


def build_test_dataset(config: dict):
    from src.datasets.drive_dataset import DriveDataset

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


def get_default_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _has_all_gt(has_gt) -> bool:
    if hasattr(has_gt, "all"):
        return bool(has_gt.all().item())
    return all(has_gt)


def main() -> None:
    args = parse_args()

    from src.utils import ensure_dir, load_config

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    threshold = float(args.threshold if args.threshold is not None else config["threshold"])

    if args.split == "test":
        vessel_mask_dir = config.get("test_vessel_mask_dir")
        if vessel_mask_dir is None or not Path(vessel_mask_dir).exists():
            raise RuntimeError(MISSING_GT_ERROR)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first or pass a valid --checkpoint path."
        )

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.metrics import compute_binary_metrics
    from src.models import build_model

    if args.split == "val":
        dataset = build_validation_dataset(config, seed)
    else:
        dataset = build_test_dataset(config)

    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)

    device_name = args.device or get_default_device_name()
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, checkpoint=checkpoint, in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for images, masks, fovs, filenames, has_gt in tqdm(loader, desc=f"evaluate-{args.split}"):
            if not _has_all_gt(has_gt):
                raise RuntimeError(MISSING_GT_ERROR)
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            for idx, filename in enumerate(filenames):
                metrics = compute_binary_metrics(
                    probs[idx : idx + 1],
                    masks[idx : idx + 1],
                    threshold=threshold,
                    fov_mask=fovs[idx : idx + 1],
                )
                rows.append({"filename": filename, **metrics})

    report_dir = ensure_dir(args.output_dir or (Path(config["output_dir"]) / "eval"))
    metrics_path = report_dir / "metrics.csv"
    metrics_json_path = report_dir / "metrics.json"
    metrics_df = pd.DataFrame(rows)
    mean_metrics = metrics_df.drop(columns=["filename"]).mean(numeric_only=True).to_dict()
    metrics_with_mean = pd.concat(
        [metrics_df, pd.DataFrame([{"filename": "mean", **mean_metrics}])],
        ignore_index=True,
    )
    metrics_with_mean.to_csv(metrics_path, index=False)

    payload = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "config": str(args.config),
        "num_samples": len(rows),
        "threshold": threshold,
        "device": device_name,
        "mean_metrics": mean_metrics,
        "per_image": rows,
    }
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved metrics JSON to {metrics_json_path}")
    print("Mean metrics:")
    display_names = {
        "dice": "Dice",
        "iou": "IoU",
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1": "F1",
    }
    for key in ["dice", "iou", "accuracy", "sensitivity", "specificity", "precision", "f1"]:
        print(f"{display_names[key]}: {mean_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
