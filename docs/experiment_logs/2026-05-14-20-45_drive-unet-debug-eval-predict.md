# 实验记录：DRIVE U-Net Debug Training、Validation Evaluation 与 Test Prediction

## 1. 任务目标

本次任务围绕医学图像分析课程作业“眼底血管分割”建立一个可复现实验流程，实际完成内容包括：

- PyTorch 项目骨架与依赖文件搭建
- DRIVE Dataset 读取与数据语义校验
- U-Net baseline 训练入口与 2 epoch smoke test
- validation split evaluation
- DRIVE test split prediction 与 visualization
- README / GitHub 项目说明更新

本次未进行正式 80 epoch 训练，未实现 Attention U-Net，未接入 CHASE_DB1。

## 2. 当前项目状态

任务开始前，项目目录 `/Users/lin/Documents/Code/Homework/Med` 已逐步完成了最小 PyTorch 项目骨架，并已有 DRIVE 数据集。后续真实执行中完成并验证了：

- `src/datasets/drive_dataset.py`：支持 DRIVE training/test 读取，明确区分 vessel ground truth 与 FOV mask。
- `src/models/unet.py`：实现 U-Net baseline，输出 logits。
- `src/losses.py`：实现 DiceLoss 与 BCE + Dice Loss。
- `src/metrics.py`：实现 Dice、IoU、Accuracy、Sensitivity、Specificity、Precision、F1。
- `src/train.py`：支持 CLI 覆盖参数，支持 16/4 train/val 划分与 checkpoint 保存。
- `src/evaluate.py`：支持 validation evaluation，拒绝在缺少 vessel GT 的 test split 上计算指标。
- `src/predict.py`：支持 DRIVE test prediction、probability map 与 visualization 输出。
- `README.md`：补充数据说明、训练/评估/预测命令、debug validation 示例结果和 Git hygiene。

尚未完成：

- Attention U-Net
- CHASE_DB1 Dataset 接入
- CUDA/GPU 正式长训练
- U-Net 与 Attention U-Net 对比实验
- 课程实验报告最终整理

## 3. 实验环境

| 项目 | 记录 |
|---|---|
| hostname | `Lins-MacBook-Air.local` |
| username | `lin` |
| working directory | `/Users/lin/Documents/Code/Homework/Med` |
| default Python | `Python 3.13.5` |
| virtual environment | `/Users/lin/Documents/Code/Homework/Med/.venv` |
| venv Python | `Python 3.11.15` |
| torch version | `2.12.0` |
| CUDA available | `False` |
| torch.version.cuda | `None` |
| GPU / display chipset | `Apple M4` |
| Metal | `Supported` |
| nvidia-smi | command not found |
| MPS check | 当前 `.venv` 中 `torch.backends.mps.is_available()` 返回 `False` |

实际运行情况说明：

- 默认 `python` 指向 `/opt/miniconda3/bin/python`，缺少 `torch`，因此训练、评估、预测的实际执行使用 `.venv/bin/python`。
- 本地为 Mac 环境，无 NVIDIA CUDA。当前 torch 探测结果显示 CUDA 不可用，MPS 也不可用。
- `outputs/debug_unet/config_used.yaml` 中记录过 `device: mps`，但当前环境复查显示 MPS 不可用；本记录以实际命令输出和日志文件为准，不把该 debug 结果作为正式 GPU 实验。

## 4. 数据集状态

本次检查到的本地数据数量：

| 路径 | 数量 | 语义 |
|---|---:|---|
| `data/DRIVE/training/images` | 20 | DRIVE training image |
| `data/DRIVE/training/1st_manual` | 20 | vessel ground truth |
| `data/DRIVE/training/mask` | 20 | FOV mask |
| `data/DRIVE/test/images` | 20 | DRIVE test image |
| `data/DRIVE/test/mask` | 20 | FOV mask |
| `data/DRIVE/test/1st_manual` | 不存在 | test split 无 vessel GT |
| `data/CHASE_DB1` | 存在，约 85 个文件 | 本次未接入、未使用 |

数据语义必须保持清楚：

- `DRIVE/training/1st_manual` 是血管分割 ground truth。
- `DRIVE/training/mask` 和 `DRIVE/test/mask` 是 FOV mask，不是血管 GT。
- 当前 `DRIVE/test` 没有 vessel GT，因此只能用于 prediction / visualization，不能用于 Dice、IoU、F1 等定量评估。
- 定量评估来自 `DRIVE/training` 内部按 `seed=42`、`val_ratio=0.2` 划分得到的 16/4 train/val split。

## 5. 执行命令

本次任务中实际运行过的关键命令如下。

环境与数据检查：

```bash
hostname
whoami
pwd
python --version
.venv/bin/python --version
.venv/bin/python -c "import torch, platform; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('mps_available', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('platform', platform.platform())"
system_profiler SPDisplaysDataType | rg -n "Chipset Model|Metal|VRAM|Vendor"
nvidia-smi
```

Dataset 检查：

```bash
python -m src.datasets.drive_dataset --split training --config configs/unet_drive.yaml
python -m src.datasets.drive_dataset --split test --config configs/unet_drive.yaml
```

编译与帮助信息：

```bash
python -m compileall src
python -m src.train --help
python -m src.evaluate --help
python -m src.predict --help
```

2 epoch debug training：

```bash
python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 2 \
  --batch_size 1 \
  --output_dir outputs/debug_unet
```

该命令在默认 Python 环境失败，原因是缺少 `torch`。随后使用 `.venv` 实际完成：

```bash
.venv/bin/python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 2 \
  --batch_size 1 \
  --output_dir outputs/debug_unet
```

Validation evaluation：

```bash
.venv/bin/python -m src.evaluate \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/debug_unet/checkpoints/best.pth \
  --split val \
  --output_dir outputs/debug_unet/eval
```

Test split 缺 GT 防护验证：

```bash
python -m src.evaluate \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/debug_unet/checkpoints/best.pth \
  --split test \
  --output_dir outputs/debug_unet/eval_test
```

DRIVE test prediction：

```bash
python -m src.predict \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/debug_unet/checkpoints/best.pth \
  --split test \
  --output_dir outputs/debug_unet
```

该命令在默认 Python 环境失败，原因是缺少 `torch`。随后使用 `.venv` 实际完成：

```bash
.venv/bin/python -m src.predict \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/debug_unet/checkpoints/best.pth \
  --split test \
  --output_dir outputs/debug_unet
```

Git 状态检查：

```bash
git status --short --branch
git log -1 --oneline
cat .gitignore
```

## 6. 代码修改

| 文件 | 修改内容 | 目的 |
|---|---|---|
| `configs/unet_drive.yaml` | 使用 `train_vessel_mask_dir`、`train_fov_mask_dir`、`test_vessel_mask_dir: null`、`test_fov_mask_dir`；新增 `val_ratio` | 适配真实 DRIVE 目录，避免把 FOV mask 当 vessel GT |
| `src/datasets/drive_dataset.py` | 支持无 vessel GT 的 test split；按文件名前两位编号匹配 image、vessel GT、FOV mask；新增 dataset CLI 检查和可视化 | 验证数据配对与 mask 语义 |
| `src/models/unet.py` | 实现标准 2D U-Net，输入 `[B,3,H,W]`，输出 logits `[B,1,H,W]` | U-Net baseline |
| `src/losses.py` | 实现 DiceLoss 与 BCEDiceLoss | 二分类血管分割训练损失 |
| `src/metrics.py` | 实现 Dice、IoU、Accuracy、Sensitivity、Specificity、Precision、F1，支持 FOV 内统计 | validation 定量评估 |
| `src/train.py` | 增加 CLI 覆盖参数；按 `seed` 和 `val_ratio` 划分 16/4；按 val Dice 保存 `best.pth`；保存 `last.pth`、`history.csv` 和 `config_used.yaml` | 支持 debug training 与可复现实验配置 |
| `src/evaluate.py` | 支持 `--split val/test`、`--output_dir`、`--threshold`、`--device`；默认 val；test 无 GT 时报错；保存 `metrics.csv` 和 `metrics.json` | validation evaluation |
| `src/predict.py` | 新增 prediction CLI；支持 test 无 GT；输出 binary mask、probability map、visualization；有 GT 时显示 GT 和 error map | DRIVE test prediction / visualization |
| `README.md` | 更新项目能力、数据语义、训练/评估/预测命令、20 epoch debug result 示例、Git hygiene | 便于课程报告和 GitHub 展示 |
| `.gitignore` | 忽略 `.venv/`、`data/`、`outputs/`、`*.pth` 等 | 避免提交数据、模型和输出文件 |

## 7. 问题与解决

### 问题 1：默认 Python 环境缺少 torch

- 现象：运行训练或预测时，默认 `python` 报错。
- 报错信息：

```text
ModuleNotFoundError: No module named 'torch'
```

- 原因分析：默认 `python` 为 `/opt/miniconda3/bin/python`，未安装 PyTorch；项目 `.venv` 中已安装 `torch 2.12.0`。
- 解决方法：实际训练、评估、预测改用 `.venv/bin/python` 执行。
- 验证结果：`.venv/bin/python` 成功完成 2 epoch debug training、validation evaluation 和 test prediction。

### 问题 2：最初 `train.py` 不支持 CLI 覆盖参数

- 现象：运行带 `--epochs` 和 `--batch_size` 的训练命令失败。
- 报错信息：

```text
train.py: error: unrecognized arguments: --epochs 2 --batch_size 1
```

- 原因分析：早期 `train.py` 只定义了 `--config` 参数。
- 解决方法：新增 `--epochs`、`--batch_size`、`--lr`、`--learning_rate`、`--weight_decay`、`--input_size`、`--threshold`、`--output_dir`、`--num_workers`、`--seed`、`--device`，并实现“先读 YAML，再用非空 CLI 参数覆盖”的逻辑。
- 验证结果：`python -m src.train --help` 显示新增参数；`.venv/bin/python -m src.train --epochs 2 --batch_size 1 ...` 成功完成。

### 问题 3：DRIVE test split 无 vessel GT，不能定量评估

- 现象：本地 `data/DRIVE/test` 只有 `images` 和 `mask`，没有 `1st_manual`。
- 报错信息：在显式评估 test split 时，脚本返回：

```text
RuntimeError: Cannot evaluate this split because vessel ground truth is missing.
```

- 原因分析：`test/mask` 是 FOV mask，不是 vessel GT。如果将其当作标签，会得到错误指标。
- 解决方法：`evaluate.py` 默认评估 training 内部 val split；当 `--split test` 且 `test_vessel_mask_dir` 为 `null` 或不存在时直接报错。`predict.py` 支持 test 无 GT，只做 prediction / visualization。
- 验证结果：test evaluation 被正确拒绝；test prediction 成功输出 20 张预测和可视化。

### 问题 4：DataLoader 默认 collate 不支持 test split 的 `None` vessel mask

- 现象：test split 没有 vessel GT，Dataset 返回 `vessel_mask=None`。
- 报错信息：该问题在实现前被识别，未让预测流程进入崩溃状态。
- 原因分析：PyTorch 默认 `DataLoader` collate 无法堆叠 `None`。
- 解决方法：在 `src/predict.py` 中实现 `collate_prediction_batch`，允许 vessel masks 整批为 `None`。
- 验证结果：`.venv/bin/python -m src.predict --split test ...` 成功处理 20 张 test image。

## 8. 运行结果

### Dataset 检查结果

training split：

- image count: 20
- vessel GT count: 20
- FOV mask count: 20
- first filename: `21_training.tif`
- image shape: `(3, 512, 512)`
- vessel mask shape: `(1, 512, 512)`，unique values: `[0, 1]`
- fov mask shape: `(1, 512, 512)`，unique values: `[0, 1]`
- visualization: `outputs/visualizations/dataset_check/training_21_training_check.png`

test split：

- image count: 20
- vessel GT count: missing
- FOV mask count: 20
- first filename: `01_test.tif`
- image shape: `(3, 512, 512)`
- vessel mask: no vessel GT
- fov mask shape: `(1, 512, 512)`，unique values: `[0, 1]`
- visualization: `outputs/visualizations/dataset_check/test_01_test_check.png`

### 2 epoch debug training

本次训练是 smoke test，不是正式实验结果。

| 项目 | 记录 |
|---|---|
| train/val split | 16/4 |
| epochs | 2 |
| batch size | 1 |
| learning rate | `1e-4` |
| weight decay | `1e-5` |
| checkpoint | `outputs/debug_unet/checkpoints/best.pth` |
| last checkpoint | `outputs/debug_unet/checkpoints/last.pth` |
| history | `outputs/debug_unet/logs/history.csv` |

训练日志：

| epoch | train loss | val Dice | val IoU | val Accuracy | val Sensitivity | val Specificity | val Precision | val F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6960 | 0.0000 | 0.0000 | 0.8813 | 0.0000 | 0.9999 | 0.0000 | 0.0000 |
| 2 | 0.5865 | 0.0032 | 0.0016 | 0.8816 | 0.0016 | 1.0000 | 0.4596 | 0.0032 |

### Validation evaluation

| 项目 | 记录 |
|---|---|
| checkpoint | `outputs/debug_unet/checkpoints/best.pth` |
| split | `val`，来自 DRIVE training 内部 16/4 split |
| samples | 4 |
| threshold | 0.5 |
| metrics.csv | `outputs/debug_unet/eval/metrics.csv` |
| metrics.json | `outputs/debug_unet/eval/metrics.json` |

Mean metrics：

| Metric | Value |
|---|---:|
| Dice | 0.0032 |
| IoU | 0.0016 |
| Accuracy | 0.8816 |
| Sensitivity | 0.0016 |
| Specificity | 1.0000 |
| Precision | 0.4596 |
| F1 | 0.0032 |

说明：这是 2 epoch smoke test 的 validation evaluation，不能作为正式实验结论。

### DRIVE test prediction

| 项目 | 数量 / 路径 |
|---|---|
| prediction binary masks | 20 |
| probability maps | 20 |
| visualization PNGs | 20 |
| prediction dir | `outputs/debug_unet/predictions/test` |
| probability dir | `outputs/debug_unet/predictions/test/probability` |
| visualization dir | `outputs/debug_unet/visualizations/test` |

test split 无 vessel GT，因此只生成 prediction / visualization，不计算 Dice、IoU、F1。

## 9. 结果分析

本次 2 epoch debug training 的主要目标是验证工程链路，而不是获得可报告的最终性能。训练 loss 从约 0.6960 降到 0.5865，说明前向、反向传播、loss 计算和参数更新链路可以正常运行。

validation 指标中 Accuracy 约 0.8816，但 Dice 和 IoU 极低，Sensitivity 也接近 0。这说明模型在 2 epoch 内几乎没有学到有效血管召回，Accuracy 主要受非血管背景占比较大影响，不能作为主要效果指标。Specificity 接近 1.0 表明模型大多预测为背景，这在训练初期是常见现象。

因此，本次结果只能作为 debug / smoke test，不能作为课程报告中的正式模型性能结论。正式报告应使用更长训练轮数，例如 80 epoch U-Net baseline，并在相同 validation split 上报告 Dice、IoU、Sensitivity、Specificity 等指标。

README 中记录的 20 epoch debug result 示例为：Dice 0.6372、IoU 0.4691、Accuracy 0.9196、Sensitivity 0.6106、Specificity 0.9603、Precision 0.7465、F1 0.6372。该结果被明确标注为 DRIVE training 内部 16/4 validation，不是 official test set。由于本次没有重新运行该 20 epoch 训练，本实验记录不将其作为本次执行得到的新实验结果。

## 10. 生成文件

本次流程中已生成或确认的重要文件：

```text
outputs/visualizations/dataset_check/training_21_training_check.png
outputs/visualizations/dataset_check/test_01_test_check.png

outputs/debug_unet/config_used.yaml
outputs/debug_unet/checkpoints/best.pth
outputs/debug_unet/checkpoints/last.pth
outputs/debug_unet/logs/history.csv

outputs/debug_unet/eval/metrics.csv
outputs/debug_unet/eval/metrics.json

outputs/debug_unet/predictions/test/
outputs/debug_unet/predictions/test/probability/
outputs/debug_unet/visualizations/test/

docs/experiment_logs/2026-05-14_drive-unet-debug-eval-predict.md
```

## 11. Git 状态

Git 状态检查结果：

```text
## main...origin/main
```

`git status --short` 输出为空，说明记录创建前工作区是干净的。最近一次提交：

```text
4dd51ae Add DRIVE prediction and update README
```

本次创建实验记录文件后，工作区会新增 `docs/experiment_logs/2026-05-14_drive-unet-debug-eval-predict.md`，尚未 commit。

`.gitignore` 已包含：

- `data/`
- `outputs/`
- `*.pth`
- `.venv/`
- `*.zip`
- `requirements.lock.txt`

因此本地数据、输出结果和模型权重不会提交到 GitHub。

## 12. 后续任务

1. 在可用 GPU 或确认可用 MPS 的环境中跑 80 epoch U-Net baseline。
2. 使用固定 `seed=42`、`val_ratio=0.2` 评估 validation split，导出正式 `metrics.csv` / `metrics.json`。
3. 用最佳 checkpoint 生成 DRIVE test prediction visualizations，作为报告可视化结果。
4. 实现 Attention U-Net，保持相同数据划分和训练超参数。
5. 做 U-Net vs Attention U-Net 对比，重点比较 Dice、IoU、Sensitivity、Specificity。
6. 整理课程实验报告表格、训练曲线、预测对比图和误差图。

报告可用总结：本项目已完成 DRIVE 数据读取、U-Net baseline 训练、validation evaluation 和 test prediction visualization 的完整工程链路。当前 test split 无 vessel GT，因此定量指标只来自 training 内部 16/4 validation。

当前结论：2 epoch debug training 证明代码链路可运行，但性能指标不能作为正式实验结论；下一步应进行更长轮数训练并实现 Attention U-Net 对比实验。
