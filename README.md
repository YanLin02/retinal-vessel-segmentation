# Retinal Vessel Segmentation

This project implements a PyTorch baseline for retinal vessel segmentation on the DRIVE dataset. The current version supports:

- DRIVE dataset loading
- U-Net baseline training
- BCE + Dice loss
- validation evaluation with Dice, IoU, Accuracy, Sensitivity, Specificity, Precision, and F1
- DRIVE test prediction and visualization
- checkpoint, CSV, JSON, prediction mask, probability map, and visualization output

Attention U-Net and CHASEDB1 are intentionally not included in the current implementation.

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

Common debug override:

```bash
python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 20 \
  --batch_size 1 \
  --output_dir outputs/unet_drive
```

## Evaluation

```bash
python -m src.evaluate \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive/checkpoints/best.pth \
  --split val \
  --output_dir outputs/unet_drive/eval
```

By default, evaluation uses the validation subset split from `data/DRIVE/training`. The DRIVE test split in this layout has no vessel ground truth and cannot be used for Dice/IoU/F1 evaluation.

## Prediction

Generate masks and visualizations for the DRIVE test split:

```bash
python -m src.predict \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive/checkpoints/best.pth \
  --split test \
  --output_dir outputs/unet_drive
```

The test split can be predicted without vessel ground truth. FOV masks are used for the visible field area and are never used as vessel labels.

## Example Result

Example 20 epoch debug result on the DRIVE training split internal 16/4 validation split:

| Metric | Value |
| --- | ---: |
| Dice | 0.6372 |
| IoU | 0.4691 |
| Accuracy | 0.9196 |
| Sensitivity | 0.6106 |
| Specificity | 0.9603 |
| Precision | 0.7465 |
| F1 | 0.6372 |

These numbers are from the validation subset created from the 20 labeled DRIVE training images. They are not official DRIVE test set metrics because the local DRIVE test split has no vessel ground truth.

## Outputs

Default outputs are written to `outputs/unet_drive`:

```text
outputs/unet_drive/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── logs/
│   └── history.csv
├── eval/
│   ├── metrics.csv
│   └── metrics.json
├── predictions/
│   └── test/
│       ├── 01_test_pred.png
│       └── probability/
└── visualizations/
    └── test/
        └── 01_test_vis.png
```

Training writes `best.pth`, `last.pth`, `config_used.yaml`, and `logs/history.csv`. Evaluation writes `eval/metrics.csv` and `eval/metrics.json`. Prediction writes binary masks, probability maps, and visualization PNGs.

## Git Hygiene

Do not commit local datasets, generated outputs, or model weights:

- `data/`
- `outputs/`
- `*.pth`
