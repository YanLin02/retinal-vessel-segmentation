# 实验记录：Attention U-Net 对比实验与 README 更新

## 1. 任务目标

本次任务目标是在已有 U-Net baseline 基础上实现并训练 Attention U-Net 作为对比模型，同时更新 README 以反映当前项目状态。具体包括：

- 新增 Attention U-Net 模型实现。
- 保持 DRIVE Dataset 逻辑和 train/val 划分逻辑不变。
- 让 `train.py`、`evaluate.py`、`predict.py` 支持 `unet` 与 `attention_unet` 两种模型。
- 新增 Attention U-Net 配置文件。
- 运行 shape test、2 epoch CUDA smoke test、80 epoch 正式训练。
- 使用 validation split 评估 Attention U-Net。
- 生成 DRIVE test prediction/visualization 和 validation visualization。
- 更新 README，加入 Attention U-Net 命令和 U-Net vs Attention U-Net 结果表。

本次没有修改 U-Net baseline 训练结果，没有修改 DRIVE Dataset 语义，没有接入 CHASE_DB1，也没有将 FOV mask 当作 vessel GT。

## 2. 当前项目状态

任务开始前项目已完成：

- DRIVE dataset loading。
- U-Net baseline。
- BCE + Dice Loss。
- validation evaluation。
- test prediction and visualization。
- U-Net baseline 在 CUDA 上完成 80 epoch 正式训练。
- U-Net baseline 结果可作为对比基线：
  - output_dir: `outputs/unet_drive_cuda_80e`
  - best epoch: 64
  - Dice: 0.7507
  - IoU: 0.6013
  - Accuracy: 0.9401
  - Sensitivity: 0.7671
  - Specificity: 0.9632
  - Precision: 0.7415
  - F1: 0.7507

任务开始时尚未完成：

- Attention U-Net 模型实现。
- Attention U-Net 配置、训练、评估和预测。
- README 对 Attention U-Net 支持状态和正式结果表的更新。

## 3. 实验环境

- hostname: `ucas-ai-14`
- username: `sunyl25`
- working directory: `/data/sunyl25/Temp_work/retinal-vessel-segmentation`
- Python: `Python 3.10.12`
- virtual environment: `/data/sunyl25/Temp_work/retinal-vessel-segmentation/.venv`
- torch: `2.6.0+cu124`
- `torch.version.cuda`: `12.4`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: 8

PyTorch 可见 GPU：

- 0 `NVIDIA L40S`
- 1 `NVIDIA L40S`
- 2 `NVIDIA L40S`
- 3 `NVIDIA GeForce RTX 4090`
- 4 `NVIDIA GeForce RTX 4090`
- 5 `NVIDIA GeForce RTX 4090`
- 6 `NVIDIA GeForce RTX 4090`
- 7 `NVIDIA GeForce RTX 4090`

本次 Attention U-Net 正式训练使用：

- device: `cuda`
- `CUDA_VISIBLE_DEVICES=4`
- 物理 GPU: `NVIDIA GeForce RTX 4090`

训练前曾尝试使用 `CUDA_VISIBLE_DEVICES=1`，但该物理 GPU 被其他进程占用，smoke test 出现 OOM，随后改用空闲 GPU 4。

## 4. 数据集状态

DRIVE 数据集文件数量：

- `data/DRIVE/training/images`: 20
- `data/DRIVE/training/1st_manual`: 20
- `data/DRIVE/training/mask`: 20
- `data/DRIVE/test/images`: 20
- `data/DRIVE/test/mask`: 20
- `data/DRIVE/test/1st_manual`: 不存在

CHASE_DB1 状态：

- `data/CHASE_DB1` 目录存在。
- 本次没有接入 CHASE_DB1，没有训练或评估 CHASE_DB1。

数据语义：

- `training/1st_manual` 是 vessel ground truth。
- `training/mask` 和 `test/mask` 是 FOV mask，不是 vessel label。
- 当前 DRIVE/test 无 vessel GT，因此 test split 只用于 prediction/visualization，不用于 Dice、IoU、F1 等定量评估。
- 定量指标均来自 DRIVE training split 内部固定 16/4 validation split，`seed=42`，`val_ratio=0.2`。

## 5. 执行命令

代码检查和 shape test：

```bash
.venv/bin/python -m compileall src
.venv/bin/python -m src.models.attention_unet
```

shape test 输出：

```text
input shape: [2, 3, 512, 512]
output shape: [2, 1, 512, 512]
```

首次 2 epoch smoke test，使用物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.train \
  --config configs/attention_unet_drive.yaml \
  --epochs 2 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/attention_unet_cuda_smoke
```

该命令因 GPU 1 显存被其他进程占满而 OOM。

重新运行 2 epoch smoke test，使用物理 GPU 4：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.train \
  --config configs/attention_unet_drive.yaml \
  --epochs 2 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/attention_unet_cuda_smoke
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.train \
  --config configs/attention_unet_drive.yaml \
  --epochs 80 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/attention_unet_cuda_80e
```

validation evaluation：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.evaluate \
  --config configs/attention_unet_drive.yaml \
  --checkpoint outputs/attention_unet_cuda_80e/checkpoints/best.pth \
  --split val \
  --output_dir outputs/attention_unet_cuda_80e/eval
```

DRIVE test prediction：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.predict \
  --config configs/attention_unet_drive.yaml \
  --checkpoint outputs/attention_unet_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/attention_unet_cuda_80e
```

validation visualization：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.predict \
  --config configs/attention_unet_drive.yaml \
  --checkpoint outputs/attention_unet_cuda_80e/checkpoints/best.pth \
  --split val \
  --output_dir outputs/attention_unet_cuda_80e
```

README 检查：

```bash
rg -n "Attention U-Net|CHASE|Experiment Results|attention_unet|unet_drive_cuda_80e|data/|outputs/|\\.pth" README.md
```

## 6. 代码修改

| 文件 | 修改内容 | 目的 |
|---|---|---|
| `src/models/attention_unet.py` | 新增 `AttentionGate`、`AttentionUp`、`AttentionUNet`，并包含 `__main__` shape test | 实现 Attention U-Net 对比模型 |
| `src/models/__init__.py` | 导出 `AttentionUNet`，新增 `get_model_name()` 和 `build_model()` | 统一根据 config/checkpoint 构建 `unet` 或 `attention_unet` |
| `src/train.py` | 将模型支持从仅 `unet` 扩展为 `unet` / `attention_unet`，使用 `build_model()` | 让训练脚本支持 Attention U-Net，同时不改变 Dataset 和 split 逻辑 |
| `src/evaluate.py` | 使用 `build_model(config, checkpoint=checkpoint)` 构建模型 | 让评估脚本能读取 Attention U-Net checkpoint |
| `src/predict.py` | 使用 `build_model(config, checkpoint=checkpoint)` 构建模型 | 让预测/可视化脚本支持 Attention U-Net |
| `configs/attention_unet_drive.yaml` | 新增 Attention U-Net DRIVE 配置 | 固定模型名、超参数、数据路径、输出目录 |
| `README.md` | 更新项目状态、Attention U-Net 命令、正式结果表和 Git hygiene 说明 | 让 README 反映当前项目真实状态 |

注意：本次没有修改 DRIVE Dataset 逻辑，没有修改 train/val split 逻辑，没有修改 U-Net baseline checkpoint。

## 7. 问题与解决

### 问题 1：Attention U-Net smoke test 在 GPU 1 上 OOM

- 现象：首次使用 `CUDA_VISIBLE_DEVICES=1` 运行 2 epoch smoke test 时，训练第一步前向传播失败。
- 报错信息：

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 128.00 MiB.
GPU 0 has a total capacity of 44.52 GiB of which 12.50 MiB is free.
Process 3880772 has 38.92 GiB memory in use.
```

- 原因分析：`CUDA_VISIBLE_DEVICES=1` 后进程内 GPU 0 对应物理 GPU 1。该 GPU 当时已有其他 Python 进程占用约 39.9 GiB，剩余显存不足。不是模型 shape 或 Dataset 问题。
- 解决方法：运行 `nvidia-smi` 查看显存占用，选择空闲的物理 GPU 4，使用 `CUDA_VISIBLE_DEVICES=4` 重跑同一 smoke test。
- 验证结果：2 epoch smoke test 在 GPU 4 上通过，输出：

```text
epoch=001/2 train_loss=0.7507 val_dice=0.2130 val_iou=0.1192
epoch=002/2 train_loss=0.7002 val_dice=0.0000 val_iou=0.0000
```

### 问题 2：训练日志 best 指标与 evaluate.py mean 指标略有差异

- 现象：`history.csv` 中 best epoch 59 的 `val_dice=0.737835`，而独立 `evaluate.py` 输出 mean Dice 为 `0.7341`。
- 报错信息：无报错。
- 原因分析：训练时 validation 以一个 batch 汇总计算指标；独立评估时逐图计算后再求 mean，因此数值略有差异。
- 解决方法：最终报告中的 validation metrics 采用 `evaluate.py` 生成的 `metrics.csv` / `metrics.json`。
- 验证结果：`outputs/attention_unet_cuda_80e/eval/metrics.csv` 和 `metrics.json` 已生成。

## 8. 运行结果

### Shape test

- 命令：`.venv/bin/python -m src.models.attention_unet`
- 结果：通过
- 输入 shape: `[2, 3, 512, 512]`
- 输出 shape: `[2, 1, 512, 512]`

### CUDA smoke test

smoke test 只用于流程验证，不能作为正式实验结论。

- output_dir: `outputs/attention_unet_cuda_smoke`
- model: `attention_unet`
- train/val split: 16/4
- epochs: 2
- batch size: 4
- learning rate: `1e-4`
- device: `cuda`
- GPU: 物理 GPU 4 `NVIDIA GeForce RTX 4090`
- epoch 2 train_loss: 0.7002
- epoch 2 val_dice: 0.0000
- epoch 2 val_iou: 0.0000

### CUDA 80 epoch 正式训练

- output_dir: `outputs/attention_unet_cuda_80e`
- model: `attention_unet`
- train/val split: 16/4
- seed: 42
- epochs: 80
- batch size: 4
- learning rate: `1e-4`
- weight decay: `1e-5`
- loss: BCE + Dice Loss
- device: `cuda`
- GPU: 物理 GPU 4 `NVIDIA GeForce RTX 4090`
- best epoch: 59
- best checkpoint: `outputs/attention_unet_cuda_80e/checkpoints/best.pth`
- last checkpoint: `outputs/attention_unet_cuda_80e/checkpoints/last.pth`
- history: `outputs/attention_unet_cuda_80e/logs/history.csv`

Best epoch 59 from `history.csv`:

- train_loss: 0.364668
- val_dice: 0.737835
- val_iou: 0.584579
- val_accuracy: 0.939757
- val_sensitivity: 0.715019
- val_specificity: 0.969986
- val_precision: 0.762156
- val_f1: 0.737835

Final epoch 80 from `history.csv`:

- train_loss: 0.311846
- val_dice: 0.730386
- val_iou: 0.575282

### Validation evaluation

- checkpoint: `outputs/attention_unet_cuda_80e/checkpoints/best.pth`
- split: `val`
- metrics.csv: `outputs/attention_unet_cuda_80e/eval/metrics.csv`
- metrics.json: `outputs/attention_unet_cuda_80e/eval/metrics.json`

Mean metrics from `evaluate.py`:

- Dice: 0.7341
- IoU: 0.5808
- Accuracy: 0.9398
- Sensitivity: 0.7136
- Specificity: 0.9700
- Precision: 0.7748
- F1: 0.7341

### Prediction and visualization

DRIVE test split:

- checkpoint: `outputs/attention_unet_cuda_80e/checkpoints/best.pth`
- prediction masks: 20
- probability maps: 20
- visualizations: 20
- prediction directory: `outputs/attention_unet_cuda_80e/predictions/test`
- visualization directory: `outputs/attention_unet_cuda_80e/visualizations/test`
- test split 无 vessel GT，未计算 Dice/IoU/F1。

Validation split:

- checkpoint: `outputs/attention_unet_cuda_80e/checkpoints/best.pth`
- prediction masks: 4
- probability maps: 4
- visualizations: 4
- prediction directory: `outputs/attention_unet_cuda_80e/predictions/val`
- visualization directory: `outputs/attention_unet_cuda_80e/visualizations/val`

### U-Net vs Attention U-Net

| Model | Dice | IoU | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7507 | 0.6013 | 0.9401 | 0.7671 | 0.9632 | 0.7415 | 0.7507 |
| Attention U-Net | 0.7341 | 0.5808 | 0.9398 | 0.7136 | 0.9700 | 0.7748 | 0.7341 |

## 9. 结果分析

Attention U-Net 的训练过程正常，loss 从前期约 0.7507 下降到第 80 轮的 0.3118，说明模型能够有效优化。validation Dice 在前 20-30 轮快速上升，并在 50-70 轮附近进入平台期。best epoch 为第 59 轮，最后一轮 train loss 继续下降但 val Dice 从 best 附近回落到 0.7304，存在一定 validation 波动，也可能有轻微过拟合。

独立 validation evaluation 的 Dice 为 0.7341，IoU 为 0.5808，整体指标正常。Sensitivity 为 0.7136，低于 U-Net baseline 的 0.7671，说明 Attention U-Net 在本次设置下检出血管像素更少；Specificity 为 0.9700、Precision 为 0.7748，高于 U-Net baseline，说明预测更保守、假阳性更少。

与 U-Net baseline 相比，Attention U-Net 本次 Dice、IoU 和 F1 略低，但 Precision 和 Specificity 更高。该结果可以作为课程报告中的对比实验结果，但应说明这是 DRIVE training 内部 16/4 validation 结果，不是 DRIVE test 官方评估结果。2 epoch smoke test 只用于流程验证，不能作为最终实验结论。

## 10. 生成文件

本次生成或更新的重要文件：

```text
src/models/attention_unet.py
configs/attention_unet_drive.yaml
README.md

outputs/attention_unet_cuda_smoke/checkpoints/best.pth
outputs/attention_unet_cuda_smoke/checkpoints/last.pth
outputs/attention_unet_cuda_smoke/logs/history.csv
outputs/attention_unet_cuda_smoke/config_used.yaml

outputs/attention_unet_cuda_80e/checkpoints/best.pth
outputs/attention_unet_cuda_80e/checkpoints/last.pth
outputs/attention_unet_cuda_80e/logs/history.csv
outputs/attention_unet_cuda_80e/config_used.yaml
outputs/attention_unet_cuda_80e/eval/metrics.csv
outputs/attention_unet_cuda_80e/eval/metrics.json
outputs/attention_unet_cuda_80e/predictions/test
outputs/attention_unet_cuda_80e/predictions/val
outputs/attention_unet_cuda_80e/visualizations/test
outputs/attention_unet_cuda_80e/visualizations/val

docs/experiment_logs/2026-05-14-21-46_attention-unet-comparison-readme.md
```

注意：`data/`、`outputs/`、`*.pth` 不应提交到 GitHub。

## 11. Git 状态

- 最近提交中已有：`9f01a2d Add Attention U-Net comparison model`
- 当前未提交变更：

```text
 M README.md
```

- 本次生成实验记录后，新增日志文件也会出现在 Git 状态中。
- 本次未 push 到 GitHub。
- 当前没有提交 `data/`、`outputs/`、`*.pth`。

## 12. 后续任务

1. 将 U-Net 与 Attention U-Net 的 validation 指标整理进课程报告表格。
2. 选择 `outputs/unet_drive_cuda_80e/visualizations/val` 和 `outputs/attention_unet_cuda_80e/visualizations/val` 中的样例做定性对比。
3. 绘制 U-Net 与 Attention U-Net 的 loss / Dice 曲线，分析收敛和波动。
4. 检查 README 与实验记录是否需要加入更多图示说明。
5. 如果课程要求更多数据集，再评估是否接入 CHASE_DB1。
6. 提交文档和代码时确认 `data/`、`outputs/`、`*.pth` 未进入 Git。

## 报告可用总结

本次实现 Attention U-Net 并完成 DRIVE 16/4 validation 对比实验。Attention U-Net best checkpoint 的 Dice 为 0.7341、IoU 为 0.5808，略低于 U-Net baseline，但 Precision 和 Specificity 更高，说明其预测更保守。

## 当前结论

Attention U-Net 流程已经跑通，可作为 U-Net baseline 的正式对比模型。本次结果显示 Attention U-Net 未超过 U-Net baseline 的 Dice/IoU，但降低了假阳性倾向。DRIVE test split 无 vessel GT，因此 test 输出仅用于可视化展示。
