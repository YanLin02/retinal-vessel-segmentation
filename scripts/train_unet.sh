#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/unet_drive.yaml
