# Retinal Vessel Segmentation

This project implements a PyTorch baseline for retinal vessel segmentation on the DRIVE dataset. The current version supports:

- DRIVE dataset loading
- U-Net baseline training
- Attention U-Net training
- BCE + Dice loss
- validation evaluation with Dice, IoU, Accuracy, Sensitivity, Specificity, Precision, and F1
- DRIVE test prediction and visualization
- checkpoint, CSV, JSON, prediction mask, probability map, and visualization output

The current implementation supports both `unet` and `attention_unet` model configs. CHASEDB1 is not connected yet.

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

U-Net baseline:

```bash
python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 80 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/unet_drive_cuda_80e
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

Attention U-Net:

```bash
python -m src.train \
  --config configs/attention_unet_drive.yaml \
  --epochs 80 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/attention_unet_cuda_80e
```

## Evaluation

U-Net baseline:

```bash
python -m src.evaluate \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive_cuda_80e/checkpoints/best.pth \
  --split val \
  --output_dir outputs/unet_drive_cuda_80e/eval
```

Attention U-Net:

```bash
python -m src.evaluate \
  --config configs/attention_unet_drive.yaml \
  --checkpoint outputs/attention_unet_cuda_80e/checkpoints/best.pth \
  --split val \
  --output_dir outputs/attention_unet_cuda_80e/eval
```

By default, evaluation uses the validation subset split from `data/DRIVE/training`. The DRIVE test split in this layout has no vessel ground truth and cannot be used for Dice/IoU/F1 evaluation.

## Prediction

Generate masks and visualizations for the DRIVE test split.

U-Net baseline:

```bash
python -m src.predict \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/unet_drive_cuda_80e
```

Attention U-Net:

```bash
python -m src.predict \
  --config configs/attention_unet_drive.yaml \
  --checkpoint outputs/attention_unet_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/attention_unet_cuda_80e
```

The test split can be predicted without vessel ground truth. FOV masks are used for the visible field area and are never used as vessel labels.

## Experiment Results

Formal validation results on the DRIVE training split internal 16/4 validation split, using seed `42` and `val_ratio=0.2`:

| Model | Dice | IoU | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7507 | 0.6013 | 0.9401 | 0.7671 | 0.9632 | 0.7415 | 0.7507 |
| Attention U-Net | 0.7341 | 0.5808 | 0.9398 | 0.7136 | 0.9700 | 0.7748 | 0.7341 |

These numbers are from the validation subset created from the 20 labeled DRIVE training images. They are not official DRIVE test set metrics because the local DRIVE test split has no vessel ground truth. DRIVE test outputs are for prediction visualization only.

## Outputs

Example U-Net baseline outputs are written to `outputs/unet_drive_cuda_80e`:

```text
outputs/unet_drive_cuda_80e/
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

Attention U-Net outputs use the same layout under `outputs/attention_unet_cuda_80e`.

## Git Hygiene

Do not commit local datasets, generated outputs, or model weights:

- `data/`
- `outputs/`
- `*.pth`
