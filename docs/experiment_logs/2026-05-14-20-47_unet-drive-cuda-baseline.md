# 实验记录：U-Net DRIVE CUDA Baseline 训练与评估

## 1. 任务目标

本次任务是在服务器上继续完成医学图像分析课程作业中的眼底血管分割实验，重点包括：

- 确认服务器仓库、环境和数据集状态。
- 诊断并修复 PyTorch CUDA 不可用问题。
- 检查 DRIVE dataset 读取逻辑，确认 vessel ground truth 与 FOV mask 没有混用。
- 运行 U-Net baseline 的 CUDA smoke test。
- 在 CUDA 可用环境中正式训练 U-Net baseline 80 epoch。
- 使用 validation split 进行定量评估。
- 对 DRIVE test split 生成 prediction、probability map 和 visualization。

本次没有实现 Attention U-Net，没有接入 CHASE_DB1，没有修改模型结构，也没有使用 DRIVE test split 做定量评估。

## 2. 当前项目状态

任务开始前项目已经具备：

- DRIVE 数据集读取代码。
- U-Net baseline。
- BCE + Dice Loss。
- validation evaluation，指标包括 Dice、IoU、Accuracy、Sensitivity、Specificity、Precision、F1。
- DRIVE test prediction 和 visualization。
- 已明确区分 vessel GT 与 FOV mask：
  - `data/DRIVE/training/1st_manual` 是 vessel ground truth。
  - `data/DRIVE/training/mask` 是 FOV mask。
  - `data/DRIVE/test/mask` 是 FOV mask。
  - 当前 `data/DRIVE/test` 无 vessel GT，因此 test split 只用于 prediction 和 visualization。
- 固定 seed=42，按 `val_ratio=0.2` 将 DRIVE training 的 20 张有标注图像划分为 train/val = 16/4。

任务开始时服务器会话中 `.venv` 已存在，但 PyTorch 安装为 `torch 2.12.0+cu130`，与服务器 driver / CUDA 12.4 不兼容，导致 `torch.cuda.is_available()` 为 `False`。

## 3. 实验环境

基本环境：

- hostname: `ucas-ai-14`
- username: `sunyl25`
- working directory: `/data/sunyl25/Temp_work/retinal-vessel-segmentation`
- Python: `Python 3.10.12`
- virtual environment: `/data/sunyl25/Temp_work/retinal-vessel-segmentation/.venv`

修复前 PyTorch CUDA 状态：

- `torch: 2.12.0+cu130`
- `torch.version.cuda: 13.0`
- `torch.cuda.is_available(): False`
- 报错/警告：`The NVIDIA driver on your system is too old (found version 12040).`

服务器 GPU/driver 诊断：

- `nvidia-smi` 可正常运行。
- Driver Version: `550.163.01`
- nvidia-smi 显示 CUDA Version: `12.4`
- `/dev/nvidia0` 到 `/dev/nvidia7` 存在。
- GPU 包括：
  - GPU 0-2: `NVIDIA L40S`
  - GPU 3-7: `NVIDIA GeForce RTX 4090`

修复后 PyTorch CUDA 状态：

- `torch: 2.6.0+cu124`
- `torch.version.cuda: 12.4`
- `torch.cuda.is_available(): True`
- `torch.cuda.device_count(): 8`
- PyTorch 可见 GPU：
  - 0 `NVIDIA L40S`
  - 1 `NVIDIA L40S`
  - 2 `NVIDIA L40S`
  - 3 `NVIDIA GeForce RTX 4090`
  - 4 `NVIDIA GeForce RTX 4090`
  - 5 `NVIDIA GeForce RTX 4090`
  - 6 `NVIDIA GeForce RTX 4090`
  - 7 `NVIDIA GeForce RTX 4090`

训练时 device 设置：

- 使用 `device=cuda`
- 设置 `CUDA_VISIBLE_DEVICES=1`
- 实际使用物理 GPU 1：`NVIDIA L40S`
- 进程内对应 `cuda:0`

## 4. 数据集状态

DRIVE 数据文件数量：

- `data/DRIVE/training/images`: 20
- `data/DRIVE/training/1st_manual`: 20
- `data/DRIVE/training/mask`: 20
- `data/DRIVE/test/images`: 20
- `data/DRIVE/test/mask`: 20
- `data/DRIVE/test/1st_manual`: 不存在

Dataset check 输出摘要：

- training:
  - image count: 20
  - vessel GT count: 20
  - FOV mask count: 20
  - first filename: `21_training.tif`
  - `has_gt: True`
- test:
  - image count: 20
  - vessel GT count: missing
  - FOV mask count: 20
  - first filename: `01_test.tif`
  - `has_gt: False`

数据语义说明：

- `1st_manual` 是 vessel ground truth。
- `mask` 是 FOV mask，不是 vessel ground truth。
- 当前 DRIVE test split 没有 vessel GT，因此不能计算 Dice、IoU、F1 等定量指标，只能用于 prediction 和 visualization。
- `data/CHASE_DB1` 目录存在，但本次实验没有接入 CHASE_DB1，也没有修改任何 CHASE_DB1 相关逻辑。

## 5. 执行命令

仓库确认：

```bash
git pull
```

环境诊断与修复：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

由于网络和 PyTorch CUDA 版本问题，后续卸载 cu130 版本并安装 cu124 版本：

```bash
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

PyTorch CUDA 检查：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

硬件诊断：

```bash
nvidia-smi || true
ls -l /dev/nvidia* || true
lspci | grep -i nvidia || true
lsmod | grep nvidia || true
cat /proc/1/cgroup || true
env | grep -E "CUDA|NVIDIA" || true
```

数据目录检查：

```bash
find data/DRIVE/training/images -type f | wc -l
find data/DRIVE/training/1st_manual -type f | wc -l
find data/DRIVE/training/mask -type f | wc -l
find data/DRIVE/test/images -type f | wc -l
find data/DRIVE/test/mask -type f | wc -l
```

代码和 dataset 检查：

```bash
.venv/bin/python -m compileall src
.venv/bin/python -m src.datasets.drive_dataset \
  --split training \
  --config configs/unet_drive.yaml
.venv/bin/python -m src.datasets.drive_dataset \
  --split test \
  --config configs/unet_drive.yaml
```

CPU debug training：

```bash
.venv/bin/python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 5 \
  --batch_size 1 \
  --device cpu \
  --output_dir outputs/unet_drive_cpu_debug
```

CUDA smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 2 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/unet_drive_cuda_smoke
```

CUDA 80 epoch 正式训练：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.train \
  --config configs/unet_drive.yaml \
  --epochs 80 \
  --batch_size 4 \
  --device cuda \
  --output_dir outputs/unet_drive_cuda_80e
```

validation split 评估：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.evaluate \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive_cuda_80e/checkpoints/best.pth \
  --split val \
  --output_dir outputs/unet_drive_cuda_80e/eval
```

DRIVE test prediction：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src.predict \
  --config configs/unet_drive.yaml \
  --checkpoint outputs/unet_drive_cuda_80e/checkpoints/best.pth \
  --split test \
  --output_dir outputs/unet_drive_cuda_80e
```

## 6. 代码修改

本次未修改项目代码。

| 文件 | 修改内容 | 目的 |
|---|---|---|
| 无 | 本次未修改项目代码 | 保持现有 U-Net baseline、Dataset 逻辑和配置语义不变 |

本次只修改了 Python 虚拟环境中的依赖版本，将 PyTorch 从不兼容的 `2.12.0+cu130` 替换为与服务器 driver 匹配的 `2.6.0+cu124`。

## 7. 问题与解决

### 问题 1：首次 `git pull` 受沙箱权限影响失败

- 现象：执行 `git pull` 时失败。
- 报错信息：`error: 不能打开 .git/FETCH_HEAD: 只读文件系统`
- 原因分析：当时命令运行环境对 `.git/FETCH_HEAD` 写入受限。
- 解决方法：在允许写入的执行环境中重新运行 `git pull`。
- 验证结果：`git pull` 输出 `已经是最新的。`

### 问题 2：pip 联网安装依赖被网络/代理限制影响

- 现象：首次安装依赖时连接 PyPI 失败。
- 报错信息：`ProxyError('Cannot connect to proxy.' ... [Errno 1] 不允许的操作)`
- 原因分析：默认 shell 中网络代理不可用或未开启。
- 解决方法：使用交互式 bash 函数 `proxyon`，代理端口为 `127.0.0.1:7897`，并重新运行 pip 安装命令。
- 验证结果：依赖可以开始下载并安装。

### 问题 3：下载 PyTorch wheel 时出现 hash mismatch

- 现象：下载 `torch-2.12.0` 时 pip 报 hash mismatch。
- 报错信息：

```text
ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.
Expected sha256 ...
Got ...
```

- 原因分析：下载过程中 wheel 内容不完整或缓存损坏。
- 解决方法：启用 `proxyon` 后使用 `--no-cache-dir` 重新安装依赖，避免复用损坏缓存。
- 验证结果：后续依赖安装继续推进。

### 问题 4：PyTorch cu130 与服务器 driver / CUDA 12.4 不兼容

- 现象：服务器 `nvidia-smi` 正常，GPU 和 `/dev/nvidia*` 可见，但 PyTorch CUDA 不可用。
- 报错信息：

```text
torch: 2.12.0+cu130
torch.version.cuda: 13.0
torch.cuda.is_available(): False
The NVIDIA driver on your system is too old (found version 12040).
```

- 原因分析：服务器 NVIDIA driver 为 `550.163.01`，nvidia-smi 显示 CUDA Version `12.4`。当前 `.venv` 中安装的是 CUDA 13.0 编译的 PyTorch wheel，与 driver 不兼容。
- 解决方法：卸载 `torch 2.12.0+cu130` 和 `torchvision 0.27.0`，安装 CUDA 12.4 对应版本：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

- 验证结果：

```text
torch: 2.6.0+cu124
torch cuda: 12.4
cuda available: True
cuda device count: 8
```

### 问题 5：CUDA 12.4 PyTorch 安装过程中大包下载卡住

- 现象：安装 `torch==2.6.0+cu124` 时，pip 长时间停在大包下载阶段。
- 报错信息：没有立即退出的错误；曾出现 `Connection timed out while downloading`，pip 尝试断点续传。
- 原因分析：网络不稳定，大体积 CUDA wheel 下载容易卡住。
- 解决方法：停止卡住的 pip 进程，由用户在终端使用 `proxyon` 后继续安装。
- 验证结果：用户完成安装并反馈：

```text
torch: 2.6.0+cu124
torch cuda: 12.4
cuda available: True
cuda device count: 8
```

### 问题 6：CUDA smoke test 第 2 轮结束后短暂等待

- 现象：2 epoch smoke test 已完成两轮训练/验证主要计算，但进程短时间没有返回。
- 报错信息：无报错。
- 原因分析：可能是 checkpoint 写入或 CUDA 同步。
- 解决方法：等待进程自然退出。
- 验证结果：进程正常退出，最后输出：

```text
epoch=002/2 train_loss=0.7144 val_dice=0.2087 val_iou=0.1165
```

## 8. 运行结果

### CPU debug training

CPU debug 只用于流程验证，不能作为正式实验结果。

- output directory: `outputs/unet_drive_cpu_debug`
- device: `cpu`
- epoch: 5
- batch size: 1
- train/val split: 16/4
- best epoch: 5
- best validation Dice from history: 0.4213
- generated files:
  - `outputs/unet_drive_cpu_debug/checkpoints/best.pth`
  - `outputs/unet_drive_cpu_debug/checkpoints/last.pth`
  - `outputs/unet_drive_cpu_debug/logs/history.csv`
  - `outputs/unet_drive_cpu_debug/eval/metrics.csv`
  - `outputs/unet_drive_cpu_debug/eval/metrics.json`

### CUDA smoke test

CUDA smoke test 只用于验证 CUDA 训练流程，不能作为正式实验结果。

- output directory: `outputs/unet_drive_cuda_smoke`
- device: `cuda`
- physical GPU: GPU 1 `NVIDIA L40S`
- epoch: 2
- batch size: 4
- train/val split: 16/4
- final smoke test output:
  - epoch 2 train_loss: 0.7144
  - epoch 2 val_dice: 0.2087
  - epoch 2 val_iou: 0.1165
- generated files:
  - `outputs/unet_drive_cuda_smoke/checkpoints/best.pth`
  - `outputs/unet_drive_cuda_smoke/checkpoints/last.pth`
  - `outputs/unet_drive_cuda_smoke/logs/history.csv`
  - `outputs/unet_drive_cuda_smoke/config_used.yaml`

### CUDA 80 epoch 正式训练

- output directory: `outputs/unet_drive_cuda_80e`
- model: U-Net baseline
- loss: BCE + Dice Loss
- train/val split: 16/4
- epoch: 80
- batch size: 4
- learning rate: `1e-4`
- weight decay: `1e-5`
- device: `cuda`
- physical GPU: GPU 1 `NVIDIA L40S`
- best epoch: 64
- best checkpoint: `outputs/unet_drive_cuda_80e/checkpoints/best.pth`
- last checkpoint: `outputs/unet_drive_cuda_80e/checkpoints/last.pth`
- history: `outputs/unet_drive_cuda_80e/logs/history.csv`

Best epoch 64 from `history.csv`:

- train_loss: 0.390168
- val_dice: 0.752437
- val_iou: 0.603126
- val_accuracy: 0.940058
- val_sensitivity: 0.768316
- val_specificity: 0.963159
- val_precision: 0.737201
- val_f1: 0.752437

Final epoch 80 from `history.csv`:

- train_loss: 0.350380
- val_dice: 0.726330
- val_iou: 0.570265

### Validation evaluation

独立评估命令使用 `best.pth`，评估 split 为 validation split。

- checkpoint: `outputs/unet_drive_cuda_80e/checkpoints/best.pth`
- split: `val`
- output directory: `outputs/unet_drive_cuda_80e/eval`
- metrics.csv: `outputs/unet_drive_cuda_80e/eval/metrics.csv`
- metrics.json: `outputs/unet_drive_cuda_80e/eval/metrics.json`

Mean metrics from `metrics.csv`:

- Dice: 0.7507
- IoU: 0.6013
- Accuracy: 0.9401
- Sensitivity: 0.7671
- Specificity: 0.9632
- Precision: 0.7415
- F1: 0.7507

Per-image validation files in evaluation:

- `40_training.tif`
- `26_training.tif`
- `37_training.tif`
- `30_training.tif`

### DRIVE test prediction

DRIVE test split 无 vessel GT，因此本步骤只生成预测和可视化，不计算 Dice、IoU、F1。

- checkpoint: `outputs/unet_drive_cuda_80e/checkpoints/best.pth`
- split: `test`
- output directory: `outputs/unet_drive_cuda_80e`
- binary prediction masks: 20
- probability maps: 20
- visualizations: 20
- prediction directory: `outputs/unet_drive_cuda_80e/predictions/test`
- visualization directory: `outputs/unet_drive_cuda_80e/visualizations/test`

## 9. 结果分析

本次 80 epoch CUDA 训练结果正常，可作为 U-Net baseline 的正式报告结果。训练 loss 从第 1 轮的 0.7551 下降到第 80 轮的 0.3504，说明优化过程有效。

validation Dice 在前 20 轮快速上升，从接近 0 提升到约 0.6352；第 64 轮达到本次最佳 Dice 0.7524，IoU 0.6031。第 80 轮 train loss 继续下降，但 val Dice 回落到 0.7263，说明后期存在一定波动，可能有轻微过拟合或 validation split 小样本导致的指标波动。因此报告中应使用按 validation Dice 保存的 `best.pth` 对应评估结果，而不是最后一轮结果。

Sensitivity 为 0.7671，说明模型能够检出较多血管像素；Specificity 为 0.9632，说明背景区域识别较稳定。Precision 为 0.7415，表明预测血管中仍有一定假阳性。Dice/F1 为 0.7507，IoU 为 0.6013，作为 DRIVE 小样本 U-Net baseline 的课程实验结果是合理的。

CPU 5 epoch 和 CUDA 2 epoch smoke test 仅用于流程验证，不能作为最终实验结论。DRIVE test split 因无 vessel GT，只能展示 prediction 和 visualization，不能汇报 test Dice/IoU/F1。

## 10. 生成文件

本次生成或使用的重要文件包括：

```text
outputs/unet_drive_cpu_debug/checkpoints/best.pth
outputs/unet_drive_cpu_debug/checkpoints/last.pth
outputs/unet_drive_cpu_debug/logs/history.csv
outputs/unet_drive_cpu_debug/eval/metrics.csv
outputs/unet_drive_cpu_debug/eval/metrics.json

outputs/unet_drive_cuda_smoke/checkpoints/best.pth
outputs/unet_drive_cuda_smoke/checkpoints/last.pth
outputs/unet_drive_cuda_smoke/logs/history.csv
outputs/unet_drive_cuda_smoke/config_used.yaml

outputs/unet_drive_cuda_80e/checkpoints/best.pth
outputs/unet_drive_cuda_80e/checkpoints/last.pth
outputs/unet_drive_cuda_80e/logs/history.csv
outputs/unet_drive_cuda_80e/config_used.yaml
outputs/unet_drive_cuda_80e/eval/metrics.csv
outputs/unet_drive_cuda_80e/eval/metrics.json
outputs/unet_drive_cuda_80e/predictions/test
outputs/unet_drive_cuda_80e/visualizations/test

outputs/visualizations/dataset_check/training_21_training_check.png
outputs/visualizations/dataset_check/test_01_test_check.png

docs/experiment_logs/2026-05-14-20-47_unet-drive-cuda-baseline.md
```

注意：`outputs/`、`data/`、`*.pth` 不应提交到 GitHub。

## 11. Git 状态

- 当前 `git status --short` 在写入本日志前为空，说明训练和预测输出没有进入 Git 跟踪状态。
- 本次未 commit。
- 本次未 push 到 GitHub。
- 本次没有提交 `data/`、`outputs/`、`*.pth`。
- 本日志文件创建后，`docs/experiment_logs/2026-05-14-20-47_unet-drive-cuda-baseline.md` 会成为新的未提交文件。

## 12. 后续任务

1. 将本次 U-Net baseline 的 validation 指标整理进课程实验报告表格。
2. 从 `outputs/unet_drive_cuda_80e/visualizations/test` 选择代表性 test visualization 用于报告展示。
3. 从 validation split 中选择若干带 GT 的样例，整理 input / GT / prediction 对比图。
4. 在保持 DRIVE 数据语义正确的前提下，实现 Attention U-Net。
5. 训练 Attention U-Net 并与本次 U-Net baseline 在相同 train/val split 上对比。
6. 整理 loss 曲线和 Dice/IoU 曲线，分析 best epoch 与最后 epoch 的差异。
7. 确认 `.gitignore` 覆盖 `data/`、`outputs/`、`*.pth`，避免误提交大文件或数据集。

## 报告可用总结

本次在服务器修复 PyTorch CUDA 环境后，使用 L40S 完成 U-Net baseline 在 DRIVE training split 上的 80 epoch 训练。best checkpoint 在 validation split 上取得 Dice 0.7507、IoU 0.6013，并生成 20 张 DRIVE test prediction 与 visualization。

## 当前结论

U-Net baseline 的 CUDA 正式训练流程已经跑通，数据语义正确，validation 定量评估结果可作为课程实验报告中的 baseline 结果。DRIVE test split 无 vessel GT，因此只能用于可视化展示，不能作为定量评估依据。
