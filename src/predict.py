from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate U-Net predictions for DRIVE images.")
    parser.add_argument("--config", default="configs/unet_drive.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Split to predict. Default is DRIVE test, which may not have vessel ground truth.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Base output directory. Defaults to config output_dir.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override probability threshold.")
    parser.add_argument("--device", default=None, help="Override device, for example cpu, cuda, cuda:0, or mps.")
    return parser.parse_args()


def get_default_device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_dataset(config: dict, split: str):
    from src.datasets.drive_dataset import DriveDataset
    from src.train import split_train_val

    if split == "test":
        return DriveDataset(
            image_dir=config["test_image_dir"],
            vessel_mask_dir=config.get("test_vessel_mask_dir"),
            fov_mask_dir=config["test_fov_mask_dir"],
            input_size=config["input_size"],
            require_gt=False,
        )

    dataset = DriveDataset(
        image_dir=config["train_image_dir"],
        vessel_mask_dir=config["train_vessel_mask_dir"],
        fov_mask_dir=config["train_fov_mask_dir"],
        input_size=config["input_size"],
        require_gt=True,
    )
    _train_dataset, val_dataset = split_train_val(
        dataset,
        float(config["val_ratio"]),
        int(config.get("seed", 42)),
    )
    return val_dataset


def collate_prediction_batch(batch):
    import torch

    images, vessel_masks, fov_masks, filenames, has_gt = zip(*batch)
    stacked_images = torch.stack(list(images), dim=0)
    stacked_fovs = torch.stack(list(fov_masks), dim=0)
    if all(mask is not None for mask in vessel_masks):
        stacked_vessels = torch.stack(list(vessel_masks), dim=0)
    else:
        stacked_vessels = None
    return stacked_images, stacked_vessels, stacked_fovs, list(filenames), list(has_gt)


def tensor_image_to_uint8(image):
    import numpy as np

    image_hwc = image.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image_hwc * 255.0, 0, 255).astype(np.uint8)


def tensor_mask_to_uint8(mask):
    import numpy as np

    array = mask.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def make_overlay(image_uint8, pred_binary):
    import numpy as np

    overlay = image_uint8.copy()
    pred = pred_binary > 0
    overlay[pred] = (0.65 * overlay[pred] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
    return overlay


def make_error_map(pred_binary, gt_binary):
    import numpy as np

    pred = pred_binary > 0
    gt = gt_binary > 0
    error = np.zeros((*pred.shape, 3), dtype=np.uint8)
    error[pred & gt] = [255, 255, 255]
    error[pred & ~gt] = [255, 0, 0]
    error[~pred & gt] = [0, 180, 255]
    return error


def save_visualization(
    image_uint8,
    fov_uint8,
    pred_uint8,
    overlay_uint8,
    save_path: Path,
    gt_uint8=None,
    error_uint8=None,
) -> None:
    from PIL import Image, ImageDraw

    panels = [
        ("original image", Image.fromarray(image_uint8)),
        ("FOV mask", Image.fromarray(fov_uint8, mode="L").convert("RGB")),
        ("prediction mask", Image.fromarray(pred_uint8, mode="L").convert("RGB")),
        ("overlay", Image.fromarray(overlay_uint8)),
    ]
    if gt_uint8 is not None and error_uint8 is not None:
        panels.extend(
            [
                ("vessel GT", Image.fromarray(gt_uint8, mode="L").convert("RGB")),
                ("error map", Image.fromarray(error_uint8)),
            ]
        )

    width, height = panels[0][1].size
    label_height = 26
    canvas = Image.new("RGB", (width * len(panels), height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * width
        canvas.paste(panel, (x, label_height))
        draw.text((x + 8, 6), label, fill=(0, 0, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.models import build_model
    from src.utils import ensure_dir, load_config

    config = load_config(args.config)
    threshold = float(args.threshold if args.threshold is not None else config["threshold"])
    output_root = Path(args.output_dir or config["output_dir"])
    prediction_dir = ensure_dir(output_root / "predictions" / args.split)
    visualization_dir = ensure_dir(output_root / "visualizations" / args.split)
    probability_dir = ensure_dir(prediction_dir / "probability")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first or pass a valid --checkpoint path."
        )

    dataset = build_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_prediction_batch,
    )

    device_name = args.device or get_default_device_name()
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, checkpoint=checkpoint, in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_count = 0
    visualization_count = 0
    with torch.no_grad():
        for images, vessel_masks, fov_masks, filenames, has_gt in tqdm(loader, desc=f"predict-{args.split}"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            for idx, filename in enumerate(filenames):
                stem = Path(filename).stem
                prob = probs[idx, 0]
                fov = fov_masks[idx, 0]
                pred = ((prob >= threshold).float() * fov).cpu()

                prob_uint8 = tensor_mask_to_uint8(prob)
                pred_uint8 = tensor_mask_to_uint8(pred)
                fov_uint8 = tensor_mask_to_uint8(fov)
                image_uint8 = tensor_image_to_uint8(images[idx].cpu())
                overlay_uint8 = make_overlay(image_uint8, pred_uint8)

                Image.fromarray(pred_uint8, mode="L").save(prediction_dir / f"{stem}_pred.png")
                Image.fromarray(prob_uint8, mode="L").save(probability_dir / f"{stem}_prob.png")
                prediction_count += 1

                gt_uint8 = None
                error_uint8 = None
                if vessel_masks is not None and has_gt[idx]:
                    gt_uint8 = tensor_mask_to_uint8(vessel_masks[idx, 0])
                    error_uint8 = make_error_map(pred_uint8, gt_uint8)

                save_visualization(
                    image_uint8=image_uint8,
                    fov_uint8=fov_uint8,
                    pred_uint8=pred_uint8,
                    overlay_uint8=overlay_uint8,
                    gt_uint8=gt_uint8,
                    error_uint8=error_uint8,
                    save_path=visualization_dir / f"{stem}_vis.png",
                )
                visualization_count += 1

    print(f"Saved prediction masks: {prediction_count}")
    print(f"Saved probability maps: {prediction_count}")
    print(f"Saved visualizations: {visualization_count}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Visualization directory: {visualization_dir}")


if __name__ == "__main__":
    main()
