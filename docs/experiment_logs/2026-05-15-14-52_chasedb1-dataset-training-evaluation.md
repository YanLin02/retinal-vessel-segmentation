# 实验记录：CHASEDB1 数据集接入、训练与评估

## 1. 任务目标

本次任务是在已有 DRIVE + U-Net + Attention U-Net 实验基础上接入 CHASEDB1 数据集，并完成第二数据集上的训练、评估和预测流程。核心目标如下：

- 在不破坏 DRIVE 逻辑的前提下新增 CHASEDB1 Dataset。
- 在 CHASEDB1 上分别训练 U-Net 和 Attention U-Net。
- 使用 CHASEDB1 后 8 张 test 图像做最终定量评价。
- 生成 test prediction、probability map 和 visualization。
- 为最终医学图像分析课程实验报告补齐第二数据集结果。

## 2. 当前项目状态

任务开始前，项目已经完成 DRIVE Dataset 接入，且 U-Net 和 Attention U-Net 已在 DRIVE 上完成训练、评估和预测。已有 DRIVE internal validation 16/4 split 结果如下：

| Dataset | Split | Model | Dice | IoU | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DRIVE | internal val | U-Net | 0.7507 | 0.6013 | 0.9401 | 0.7671 | 0.9632 | 0.7415 | 0.7507 |
| DRIVE | internal val | Attention U-Net | 0.7341 | 0.5808 | 0.9398 | 0.7136 | 0.9700 | 0.7748 | 0.7341 |

当时 CHASEDB1 尚未接入，因此本次任务补齐了 CHASEDB1 Dataset、训练配置、评估流程和预测可视化流程。

## 3. 实验环境

服务器环境：

- hostname: `ucas-ai-14`
- username: `sunyl25`
- working directory: `/data/sunyl25/Temp_work/retinal-vessel-segmentation`
- virtual environment: `/data/sunyl25/Temp_work/retinal-vessel-segmentation/.venv`

Python / PyTorch：

- Python: `.venv/bin/python`
- torch: `2.6.0+cu124`
- torch.version.cuda: `12.4`

训练使用的 CUDA 环境由用户在 GPU 可用会话中确认：

- `nvidia-smi` 可用
- `torch.cuda.is_available(): True`
- `torch.cuda.device_count(): 8`

GPU 列表：

- GPU 0: NVIDIA L40S
- GPU 1: NVIDIA L40S
- GPU 2: NVIDIA L40S
- GPU 3: NVIDIA GeForce RTX 4090
- GPU 4: NVIDIA GeForce RTX 4090
- GPU 5: NVIDIA GeForce RTX 4090
- GPU 6: NVIDIA GeForce RTX 4090
- GPU 7: NVIDIA GeForce RTX 4090

实际训练使用：

- U-Net CHASEDB1: `CUDA_VISIBLE_DEVICES=0`，程序内 `cuda:0`，物理 GPU 0，NVIDIA L40S。
- Attention U-Net CHASEDB1: `CUDA_VISIBLE_DEVICES=4`，程序内 `cuda:0`，物理 GPU 4，NVIDIA GeForce RTX 4090。

补充说明：任务早期曾处于 GPU 不可用的沙盒/会话中，表现为 `nvidia-smi` 无法连接 NVIDIA driver、`/dev/nvidia*` 不存在、`torch.cuda.is_available()==False`。因此当时只完成代码实现和 dataset check，没有启动 smoke 或正式训练。后续用户切换到 CUDA 可用会话后，手动完成了正式训练、评估和预测。本实验记录生成时再次检查当前会话，仍无法访问 GPU，但训练输出文件、history、metrics 和预测文件均已存在并完成核对。

## 4. 数据集状态

### DRIVE

Dataset check 结果：

- DRIVE `training/images`: 20
- DRIVE `training/1st_manual`: 20
- DRIVE `training/mask`: 20
- DRIVE `test/images`: 20
- DRIVE `test/mask`: 20
- DRIVE `test/1st_manual`: 不存在

语义说明：

- `DRIVE/training/1st_manual` 是 vessel ground truth。
- `DRIVE/training/mask` 和 `DRIVE/test/mask` 是 FOV mask，不是 vessel GT。
- 当前 DRIVE test 没有 vessel GT，因此 DRIVE test 只用于 prediction / visualization，不用于 Dice / IoU / F1 定量评估。

### CHASEDB1

数据目录：

- `data/CHASE_DB1`

文件形式：

```text
Image_01L.jpg
Image_01L_1stHO.png
Image_01L_2ndHO.png
Image_01R.jpg
Image_01R_1stHO.png
Image_01R_2ndHO.png
...
Image_14L.jpg
Image_14L_1stHO.png
Image_14L_2ndHO.png
Image_14R.jpg
Image_14R_1stHO.png
Image_14R_2ndHO.png
```

CHASEDB1 使用规则：

- 只读取 `*.jpg` 作为图像。
- 按文件名排序。
- 前 20 张作为 `train_pool`。
- `train_pool` 内部按 `seed=42` 和 `val_ratio=0.2` 划分为 train/val = 16/4。
- 后 8 张作为 final test。
- 使用 `*_1stHO.png` 作为 vessel GT。
- `*_2ndHO.png` 未进入主实验。
- FOV mask 由 RGB 图像转灰度后自动生成：`gray > 10`。
- CHASEDB1 test split 有 GT，因此可以计算 Dice / IoU / F1 等定量指标。

CHASEDB1 dataset check 结果：

- train: 16 images, 16 vessel GT, 16 FOV masks, `has_gt=True`
- val: 4 images, 4 vessel GT, 4 FOV masks, `has_gt=True`
- test: 8 images, 8 vessel GT, 8 FOV masks, `has_gt=True`
- image shape: `(3, 512, 512)`
- vessel mask shape: `(1, 512, 512)`, unique=`[0, 1]`
- fov mask shape: `(1, 512, 512)`, unique=`[0, 1]`

## 5. 执行命令

### 环境检查

```bash
cd /data/sunyl25/Temp_work/retinal-vessel-segmentation
source .venv/bin/activate

nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

### 代码和 Dataset 检查

```bash
.venv/bin/python -m compileall src

.venv/bin/python -m src.datasets.drive_dataset \
  --split training \
  --config configs/unet_drive.yaml

.venv/bin/python -m src.datasets.drive_dataset \
  --split test \
  --config configs/unet_drive.yaml

.venv/bin/python -m src.datasets.chasedb1_dataset \
  --split train \
  --config configs/unet_chasedb1.yaml

.venv/bin/python -m src.datasets.chasedb1_dataset \
  --split val \
  --config configs/unet_chasedb1.yaml

.venv/bin/python -m src.datasets.chasedb1_dataset \
  --split test \
  --config configs/unet_chasedb1.yaml
```

### 正式训练

U-Net:

```bash
mkdir -p outputs/logs

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m src.train \
  --config configs/unet_chasedb1.yaml \
  --epochs 80 \
  --batch_size 2 \
  --device cuda \
  --output_dir outputs/unet_chasedb1_cuda_80e \
  > outputs/logs/unet_chasedb1_cuda_80e.log 2>&1 &
```

Attention U-Net:

```bash
CUDA_VISIBLE_DEVICES=4 nohup .venv/bin/python -m src.train \
  --config configs/attention_unet_chasedb1.yaml \
  --epochs 80 \
  --batch_size 2 \
  --device cuda \
  --output_dir outputs/attention_unet_chasedb1_cuda_80e \
  > outputs/logs/attention_unet_chasedb1_cuda_80e.log 2>&1 &
```

### 查看训练日志

```bash
tail -f outputs/logs/unet_chasedb1_cuda_80e.log
tail -f outputs/logs/attention_unet_chasedb1_cuda_80e.log
```

### 测试集评估

U-Net:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.evaluate \
  --config configs/unet_chasedb1.yaml \
  --checkpoint outputs/unet_chasedb1_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/unet_chasedb1_cuda_80e/eval_test
```

Attention U-Net:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.evaluate \
  --config configs/attention_unet_chasedb1.yaml \
  --checkpoint outputs/attention_unet_chasedb1_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/attention_unet_chasedb1_cuda_80e/eval_test
```

### 测试集预测和可视化

U-Net:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.predict \
  --config configs/unet_chasedb1.yaml \
  --checkpoint outputs/unet_chasedb1_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/unet_chasedb1_cuda_80e
```

Attention U-Net:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m src.predict \
  --config configs/attention_unet_chasedb1.yaml \
  --checkpoint outputs/attention_unet_chasedb1_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/attention_unet_chasedb1_cuda_80e
```

### 提取 best epoch

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

runs = {
    "U-Net CHASEDB1": "outputs/unet_chasedb1_cuda_80e/logs/history.csv",
    "Attention U-Net CHASEDB1": "outputs/attention_unet_chasedb1_cuda_80e/logs/history.csv",
}

for name, path in runs.items():
    df = pd.read_csv(path)
    best = df.loc[df["val_dice"].idxmax()]
    print(name)
    print("best epoch:", int(best["epoch"]))
    print("best val dice:", round(float(best["val_dice"]), 4))
    print("best val iou:", round(float(best["val_iou"]), 4))
    print()
PY
```

## 6. 代码修改

| 文件 | 修改内容 | 目的 |
|---|---|---|
| `src/datasets/chasedb1_dataset.py` | 新增 `CHASEDB1Dataset`，支持 `train_pool/train/val/test`，读取 jpg 和 1stHO，自动生成 FOV | 接入 CHASEDB1 |
| `src/datasets/factory.py` | 新增 dataset factory，根据 `config["dataset"]` 构建 drive 或 chasedb1 数据集 | 统一 train/evaluate/predict 的数据集构建逻辑 |
| `src/datasets/__init__.py` | 导出 `CHASEDB1Dataset` 和 factory 函数 | 统一数据集接口 |
| `configs/unet_chasedb1.yaml` | 新增 U-Net 在 CHASEDB1 上的训练配置 | 支持 CHASEDB1 U-Net 实验 |
| `configs/attention_unet_chasedb1.yaml` | 新增 Attention U-Net 在 CHASEDB1 上的训练配置 | 支持 CHASEDB1 Attention U-Net 实验 |
| `src/train.py` | 改为通过 factory 构建 train/val dataset，保留 `best.pth` 按 val Dice 保存 | 支持多数据集训练 |
| `src/evaluate.py` | 改为通过 factory 构建 eval dataset，支持 CHASEDB1 test 有 GT 评估 | 支持 CHASEDB1 test 定量评价 |
| `src/predict.py` | 改为通过 factory 构建 predict dataset，支持 CHASEDB1 test 可视化包含 GT/error map | 支持 CHASEDB1 预测和可视化 |
| `README.md` | 补充 CHASEDB1 布局、split、1stHO/2ndHO、FOV 生成和命令 | 更新文档 |

说明：

- 未修改 U-Net 和 Attention U-Net 模型结构。
- 未破坏 DRIVE Dataset 逻辑。
- 旧 DRIVE 配置没有 `dataset` 字段时默认 `dataset=drive`。

## 7. 问题与解决

### 问题 1：初始会话 GPU 不可用

现象：

- `nvidia-smi` 无法连接 NVIDIA driver。
- `/dev/nvidia*` 不存在。
- `torch.cuda.is_available() == False`。

原因分析：

- 当前会话 / 沙盒未暴露 GPU 设备，不能进行 CUDA 训练。

解决方法：

- 没有强行 CPU 训练。
- 等待/切换到 GPU 可用会话。

验证结果：

- 后续训练会话中 `nvidia-smi` 正常，`torch.cuda.is_available() == True`，`device count = 8`。
- 两个模型均完成 80 epoch 训练，并生成 checkpoint、history、test metrics 和 prediction visualizations。

### 问题 2：正式训练前需要确认 CHASEDB1 FOV mask

现象：

- CHASEDB1 没有 DRIVE 那样的独立 FOV mask 目录。

原因分析：

- 需要自动生成 FOV mask，否则指标统计区域不明确。

解决方法：

- 使用 RGB 转灰度后 `gray > 10` 自动生成 FOV mask。
- 不引入 opencv/scipy 等新依赖。

验证结果：

- CHASEDB1 train/val/test dataset check 中 FOV mask unique=`[0, 1]`。
- Dataset check 可视化已保存到 `outputs/visualizations/chasedb1_dataset_check/`。

## 8. 运行结果

### 8.1 CHASEDB1 U-Net 训练

- dataset: CHASEDB1
- model: U-Net
- train/val split: 16/4
- test split: 8
- epochs: 80
- batch_size: 2
- learning_rate: `1e-4`
- loss: BCE + Dice
- device: `cuda`
- physical GPU: NVIDIA L40S, `CUDA_VISIBLE_DEVICES=0`
- output_dir: `outputs/unet_chasedb1_cuda_80e`
- best checkpoint: `outputs/unet_chasedb1_cuda_80e/checkpoints/best.pth`
- last checkpoint: `outputs/unet_chasedb1_cuda_80e/checkpoints/last.pth`
- history: `outputs/unet_chasedb1_cuda_80e/logs/history.csv`
- best epoch: 78
- best val Dice: 0.7913
- best val IoU: 0.6548

训练趋势：

- `train_loss` 从 0.7400 降至 0.2098。
- `val_dice` 从 0.0000 上升到最高约 0.7913。
- 后期 validation Dice 在 0.78 到 0.79 附近波动，说明模型已经基本收敛。

### 8.2 CHASEDB1 Attention U-Net 训练

- dataset: CHASEDB1
- model: Attention U-Net
- train/val split: 16/4
- test split: 8
- epochs: 80
- batch_size: 2
- learning_rate: `1e-4`
- loss: BCE + Dice
- device: `cuda`
- physical GPU: NVIDIA GeForce RTX 4090, `CUDA_VISIBLE_DEVICES=4`
- output_dir: `outputs/attention_unet_chasedb1_cuda_80e`
- best checkpoint: `outputs/attention_unet_chasedb1_cuda_80e/checkpoints/best.pth`
- last checkpoint: `outputs/attention_unet_chasedb1_cuda_80e/checkpoints/last.pth`
- history: `outputs/attention_unet_chasedb1_cuda_80e/logs/history.csv`
- best epoch: 69
- best val Dice: 0.7855
- best val IoU: 0.6471

训练趋势：

- `train_loss` 从 0.7356 降至 0.2046。
- `val_dice` 从 0.0000 上升到最高约 0.7855。
- 后期 validation Dice 稳定在约 0.78 附近。

### 8.3 CHASEDB1 test evaluation

U-Net checkpoint:

- `outputs/unet_chasedb1_cuda_80e/checkpoints/best.pth`

U-Net test split metrics:

| Dataset | Split | Epoch | Batch size | Device | Checkpoint | Output dir | Dice | IoU | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CHASEDB1 | test | 78 | 2 | cuda | `outputs/unet_chasedb1_cuda_80e/checkpoints/best.pth` | `outputs/unet_chasedb1_cuda_80e/eval_test` | 0.7586 | 0.6121 | 0.9568 | 0.7533 | 0.9772 | 0.7711 | 0.7586 |

输出：

- `outputs/unet_chasedb1_cuda_80e/eval_test/metrics.csv`
- `outputs/unet_chasedb1_cuda_80e/eval_test/metrics.json`

Attention U-Net checkpoint:

- `outputs/attention_unet_chasedb1_cuda_80e/checkpoints/best.pth`

Attention U-Net test split metrics:

| Dataset | Split | Epoch | Batch size | Device | Checkpoint | Output dir | Dice | IoU | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CHASEDB1 | test | 69 | 2 | cuda | `outputs/attention_unet_chasedb1_cuda_80e/checkpoints/best.pth` | `outputs/attention_unet_chasedb1_cuda_80e/eval_test` | 0.7574 | 0.6103 | 0.9562 | 0.7567 | 0.9761 | 0.7622 | 0.7574 |

输出：

- `outputs/attention_unet_chasedb1_cuda_80e/eval_test/metrics.csv`
- `outputs/attention_unet_chasedb1_cuda_80e/eval_test/metrics.json`

### 8.4 CHASEDB1 test prediction

U-Net:

- prediction masks: 8
- probability maps: 8
- visualizations: 8
- prediction directory: `outputs/unet_chasedb1_cuda_80e/predictions/test`
- visualization directory: `outputs/unet_chasedb1_cuda_80e/visualizations/test`

Attention U-Net:

- prediction masks: 8
- probability maps: 8
- visualizations: 8
- prediction directory: `outputs/attention_unet_chasedb1_cuda_80e/predictions/test`
- visualization directory: `outputs/attention_unet_chasedb1_cuda_80e/visualizations/test`

## 9. 结果分析

CHASEDB1 上两个模型的 test 性能非常接近。U-Net 在 Dice、IoU、Accuracy、Specificity、Precision 和 F1 上略高，其中 Dice 为 0.7586，高于 Attention U-Net 的 0.7574；IoU 为 0.6121，高于 Attention U-Net 的 0.6103。Attention U-Net 在 Sensitivity 上略高，为 0.7567，高于 U-Net 的 0.7533。

这说明 Attention U-Net 在 CHASEDB1 上略微提高了血管召回，但没有带来 Dice/IoU 的明显提升。两个模型差距非常小，不应夸大优劣。由于 CHASEDB1 test split 有 `1stHO` vessel GT，因此这些是正式 test 指标；这点不同于当前 DRIVE test，后者没有 vessel GT，只能用于 prediction / visualization。

结合 DRIVE 结果看，DRIVE internal validation 上 U-Net 综合指标略优，Attention U-Net 的 precision/specificity 更高；CHASEDB1 上两者几乎持平。总体上，Attention Gate 在本实验中没有稳定提升 Dice/IoU，但会影响 precision-recall trade-off。Accuracy 仍不能作为唯一指标，因为背景像素占比大；实验报告中应重点分析 Dice、IoU、Sensitivity、Precision 和 F1。当前 CHASEDB1 结果可以作为正式报告结果。

## 10. 生成文件

代码/配置：

- `src/datasets/chasedb1_dataset.py`
- `src/datasets/factory.py`
- `configs/unet_chasedb1.yaml`
- `configs/attention_unet_chasedb1.yaml`

训练输出：

- `outputs/unet_chasedb1_cuda_80e/checkpoints/best.pth`
- `outputs/unet_chasedb1_cuda_80e/checkpoints/last.pth`
- `outputs/unet_chasedb1_cuda_80e/logs/history.csv`
- `outputs/attention_unet_chasedb1_cuda_80e/checkpoints/best.pth`
- `outputs/attention_unet_chasedb1_cuda_80e/checkpoints/last.pth`
- `outputs/attention_unet_chasedb1_cuda_80e/logs/history.csv`

评估输出：

- `outputs/unet_chasedb1_cuda_80e/eval_test/metrics.csv`
- `outputs/unet_chasedb1_cuda_80e/eval_test/metrics.json`
- `outputs/attention_unet_chasedb1_cuda_80e/eval_test/metrics.csv`
- `outputs/attention_unet_chasedb1_cuda_80e/eval_test/metrics.json`

预测输出：

- `outputs/unet_chasedb1_cuda_80e/predictions/test/`
- `outputs/unet_chasedb1_cuda_80e/visualizations/test/`
- `outputs/attention_unet_chasedb1_cuda_80e/predictions/test/`
- `outputs/attention_unet_chasedb1_cuda_80e/visualizations/test/`

dataset check：

- `outputs/visualizations/chasedb1_dataset_check/`

文档：

- `docs/experiment_logs/2026-05-15-14-52_chasedb1-dataset-training-evaluation.md`

## 11. Git 状态

执行命令：

```bash
git status --short
```

记录结果：

```text

```

当前 `git status --short` 为空，说明工作区干净。`data/`、`outputs/`、`*.pth` 未提交到 GitHub。最近一次 commit message 为：

```text
添加 CHASEDB1 数据集支持，包括数据集加载、训练、验证和测试功能，并更新相关文档和配置文件。
```

## 12. 后续任务

1. 将 CHASEDB1 输出拉回本地。
2. 压缩 CHASEDB1 visualization，生成 report_assets。
3. 更新 README 的最终结果表，加入 CHASEDB1 test 指标。
4. 更新课程实验报告：数据集说明、模型方法、训练设置、DRIVE 与 CHASEDB1 总结果表、可视化结果、讨论与局限性。
5. 检查全文，确保 DRIVE test 没有被写成定量测试集，CHASEDB1 test 是有 GT 的正式测试集，FOV mask 没有被写成 vessel GT。
6. 不再继续调参，除非报告明确需要额外消融实验。

## 报告可用总结

本次在保持 DRIVE 逻辑不变的基础上接入 CHASEDB1，完成 U-Net 与 Attention U-Net 的 80 epoch 训练、test 评估和预测可视化。CHASEDB1 test Dice 分别为 0.7586 和 0.7574，可作为正式报告结果。

## 当前结论

CHASEDB1 上 U-Net 与 Attention U-Net 表现接近，U-Net 在 Dice/IoU/F1 上略高，Attention U-Net 在 Sensitivity 上略高。结合 DRIVE 结果，Attention Gate 未在本实验中稳定提升 Dice/IoU，但会影响 precision-recall trade-off。
