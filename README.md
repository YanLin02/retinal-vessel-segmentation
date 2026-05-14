# Retinal Vessel Segmentation

This project implements a PyTorch baseline for retinal vessel segmentation on the DRIVE dataset. The current version supports:

- DRIVE dataset loading
- U-Net baseline
- BCE + Dice loss
- Metrics for segmentation evaluation
- Checkpoint and CSV log output

Attention U-Net and CHASEDB1 are intentionally not included in this initial skeleton.

## Dataset Layout

Download the DRIVE dataset and place files under `data/DRIVE` using this layout:

```text
data/DRIVE/
├── training/
│   ├── images/
│   │   ├── 21_training.tif
│   │   └── ...
│   ├── 1st_manual/
│   │   ├── 21_manual1.gif
│   │   └── ...
│   └── mask/
│       ├── 21_training_mask.gif
│       └── ...
└── test/
    ├── images/
    │   ├── 01_test.tif
    │   └── ...
    └── mask/
        ├── 01_test_mask.gif
        └── ...
```

Files are matched by the first two digits in the filename, for example:

- `21_training.tif`
- `21_manual1.gif`
- `21_training_mask.gif`

Important: `training/1st_manual` is the vessel ground truth. `training/mask` and `test/mask` are FOV masks, not vessel labels. The DRIVE test split in this layout has no vessel ground truth, so quantitative metrics are computed from a train/validation split of the 20 training images.

## Dataset Check

```bash
python -m src.datasets.drive_dataset --split training --config configs/unet_drive.yaml
python -m src.datasets.drive_dataset --split test --config configs/unet_drive.yaml
```

Dataset check figures are saved to `outputs/visualizations/dataset_check/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python -m src.train --config configs/unet_drive.yaml
```

or:

```bash
bash scripts/train_unet.sh
```

## Evaluation

```bash
python -m src.evaluate \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive/checkpoints/best.pth
```

By default, evaluation uses the validation subset split from `data/DRIVE/training`. The DRIVE test split in this layout has no vessel ground truth and cannot be used for Dice/IoU/F1 evaluation.

## Outputs

Default outputs are written to `outputs/unet_drive`:

```text
outputs/unet_drive/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── logs/
│   └── history.csv
├── predictions/
└── visualizations/
```

The initial training script writes checkpoints and a training CSV log. Prediction masks and visualization figures are reserved for later stages.
