# 训练指南 - 优化版基线

## 🎯 快速开始

### 1. 安装依赖（首次运行）

```powershell
cd qw-advice
python -m pip install -r requirements.txt
```

### 2. 检查环境

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 3. 开始训练

#### 标准配置（推荐，需要 8GB+ 显存）
```powershell
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/opt-v1
```

这将使用：
- ResNet101 backbone
- 768×768 裁剪
- Batch size 4
- 类别权重
- 50 epochs + 早停

#### 轻量配置（4-6GB 显存）
```powershell
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/opt-light --backbone resnet50 --crop-size 512 --batch-size 4
```

#### 高性能配置（12GB+ 显存）
```powershell
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/opt-hires --crop-size 1024 --batch-size 2
```

### 4. 监控训练

训练过程中会显示：
- 每轮的训练损失
- 验证 mIoU
- 各类别 IoU
- 是否刷新最佳模型

示例输出：
```
Train 1/50: 100%|████████| 1234/1234 [05:23<00:00, 3.82it/s]
Validate: 100%|████████| 137/137 [00:45<00:00, 3.01it/s]
Epoch 1: loss=1.2345, mIoU=0.6234
  Per-class IoU: ['0.850', '0.720', '0.680', '0.590', '0.710', '0.650', '0.580', '0.420']
  ✓ New best mIoU: 0.6234
  Saved best model to runs/opt-v1/best.pt
```

### 5. 恢复训练

如果训练中断，可以从最近的 checkpoint 恢复：

```powershell
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/opt-v1 --resume runs/opt-v1/last.pt --epochs 60
```

注意：`--epochs` 是总轮数，不是额外轮数。

### 6. 推理与提交

```powershell
# 基础推理（带水平翻转TTA）
python predict.py --images "test/images" --checkpoint runs/opt-v1/best.pt --output submission/masks --tta

# 多尺度推理（更强但更慢，约3倍时间）
python predict.py --images "test/images" --checkpoint runs/opt-v1/best.pt --output submission/masks --tta --multi-scale

# 验证提交格式
python validate_submission.py --images "test/images" --masks submission/masks --zip submission/submission.zip
```

## ⚙️ 参数调优指南

### 显存不足时

按优先级降低配置：

1. **减小 batch size**
   ```powershell
   --batch-size 2
   ```

2. **减小 crop size**
   ```powershell
   --crop-size 512
   ```

3. **换用 ResNet50**
   ```powershell
   --backbone resnet50
   ```

4. **减少 workers**
   ```powershell
   --workers 2
   ```

### 提升性能时

1. **增大 crop size**（最有效）
   ```powershell
   --crop-size 1024
   ```

2. **增加训练轮数**
   ```powershell
   --epochs 80
   ```

3. **调整学习率**
   ```powershell
   --lr 5e-4  # 或 1e-4
   ```

4. **禁用早停**（让模型充分训练）
   ```powershell
   --patience 0
   ```

### 验证集划分

默认 10% 数据作为验证集。如果验证 mIoU 波动大，可以增大：

```powershell
--val-ratio 0.15  # 15% 验证集
```

## 📊 预期训练时间

基于 NVIDIA RTX 3060 (12GB)：

| 配置 | 每轮时间 | 50轮总时间 |
|------|----------|------------|
| ResNet50, 512×512, BS=4 | ~3分钟 | ~2.5小时 |
| ResNet101, 768×768, BS=4 | ~6分钟 | ~5小时 |
| ResNet101, 1024×1024, BS=2 | ~10分钟 | ~8小时 |

## 🔍 调试技巧

### 查看训练日志

所有关键信息都会打印到控制台。如需保存日志：

```powershell
python train.py ... 2>&1 | Tee-Object -FilePath training.log
```

### 检查数据加载

```powershell
# 测试数据加载是否正常
python -c "from uavseg.data import SegmentationDataset; from pathlib import Path; ds = SegmentationDataset(Path('data/images'), Path('data/masks'), ['0000.png'], 512, False); print(ds[0][0].shape, ds[0][1].shape)"
```

### 验证模型输出

```powershell
# 测试单张图像推理
python -c "from uavseg.core import *; import torch; model = build_model(False).cuda(); x = torch.randn(1, 3, 768, 768).cuda(); y = model(x)['out']; print(y.shape)"
```

## 🎓 进阶技巧

### 1. 学习率调整

如果发现训练不稳定：
- 降低初始 lr：`--lr 1e-4`
- 增加 warmup：修改代码中的 `warmup_epochs`

### 2. 数据增强调整

在 `uavseg/data.py` 的 `_augment` 方法中调整：
- 缩放范围：`random.uniform(0.75, 1.35)` → `(0.8, 1.2)`
- 颜色扰动强度：`(0.8, 1.2)` → `(0.9, 1.1)`

### 3. 损失函数权重

在 `train.py` 中调整：
```python
main_loss = main_ce + 0.5 * main_dice  # Dice 权重
loss = main_loss + 0.4 * aux_loss      # 辅助损失权重
```

### 4. TTA 策略组合

在推理时尝试不同组合：
- 仅水平翻转：`--tta`
- 多尺度：`--multi-scale`
- 两者结合：`--tta --multi-scale`（最强）

## ❓ 常见问题

**Q: 训练一开始 loss 就很大怎么办？**  
A: 检查数据路径是否正确，标签值是否在 [0,8] 范围内。

**Q: 验证 mIoU 一直不提升？**  
A: 可能是学习率太大，尝试 `--lr 1e-4`；或者增加 patience。

**Q: CUDA out of memory？**  
A: 按"显存不足时"章节的建议逐步降低配置。

**Q: 推理速度太慢？**  
A: 使用 `--batch-size 16` 增大滑窗批处理大小；或关闭 multi-scale。

**Q: 如何提高小目标（Vehicle）的检测？**  
A: 
1. 确保使用了类别权重（默认启用）
2. 增大 crop-size 到 1024
3. 使用多尺度推理

## 📈 性能基准

在验证集上的预期表现：

| Epoch | mIoU | 说明 |
|-------|------|------|
| 5 | 0.55-0.60 | 初期收敛 |
| 10 | 0.62-0.68 | 快速提升 |
| 20 | 0.68-0.73 | 稳定增长 |
| 30 | 0.72-0.76 | 接近饱和 |
| 40+ | 0.74-0.78 | 微调阶段 |

实际表现取决于数据分布和硬件配置。

---

祝训练顺利！如有问题请查看 README.md 或检查训练日志。
