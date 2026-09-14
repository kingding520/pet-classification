# ResNet-18 GPU 实验

本目录是阶段三的独立实验实现；不修改 MobileNetV2 CPU 基线、现有 Android 模型或 ShuffleNetV2 历史实验。

## GPU 正式训练前提

- 使用 PyTorch CUDA 12.8 在本机 RTX 5060 上训练；MindSpore Lite 继续负责后续 `.ms` 转换。
- 使用 PetImages 中既有的 `train/`、`eval/` 划分，类别索引保持 ImageFolderDataset 的字典序：`Cat=0`、`Dog=1`。
- 固定随机种子 1、批大小 32、训练 30 epoch，完整保留环境和参数记录。

GPU 命令：

```bash
python train_resnet18_torch.py \
  --device GPU \
  --dataset_path /path/to/PetImages \
  --output_dir outputs/gpu_30ep \
  --epochs 30 --batch_size 32 --seed 1
```

训练脚本在 `--device GPU` 且 CUDA 不可用时会直接失败，绝不会隐式回退到 CPU。

原 `train_resnet18.py` 保留为 MindSpore 训练实现和 CPU 代码链路验证；由于本机 MindSpore GPU 与 RTX 5060/CUDA 13.2 不兼容，不用于本机正式 GPU 训练。

## MindSpore Lite `.ms` 转换

GPU 训练的最佳 `.pth` 将先导出为 ONNX，再使用 MindSpore Lite Converter 转为 `.ms`：

```powershell
converter_lite.exe --fmk=ONNX --modelFile=resnet18_pet.onnx --outputFile=resnet18_pet
```

转换成功需出现 `CONVERT RESULT SUCCESS:0`，并校验生成的 `resnet18_pet.ms`。

## 已弃用的 MindSpore GPU 冒烟命令

```bash
python train_resnet18.py \
  --platform GPU \
  --dataset_path /path/to/PetImages \
  --output_dir outputs/gpu_smoke \
  --epochs 1 --batch_size 32 --seed 1 --max_train_steps 1 --max_eval_steps 1
```

该命令仅适用于具备兼容 MindSpore GPU 环境的主机，当前本机不执行。

## CPU 同架构对照

GPU 训练完成后，再用完全相同的参数运行：

```bash
python train_resnet18_torch.py \
  --device CPU \
  --dataset_path /path/to/PetImages \
  --output_dir outputs/cpu_30ep \
  --epochs 30 --batch_size 32 --seed 1
```

对照 `summary.json` 与 `metrics.jsonl` 中的最高验证准确率、总耗时和每 epoch 耗时。既有 MobileNetV2 CPU 结果仅作历史参考，不用于计算 ResNet-18 的 GPU 加速倍数。
