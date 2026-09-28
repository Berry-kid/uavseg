# ⚡ 快速参考卡

## 🎯 一行命令速查

### 训练
```powershell
# 标准训练（推荐）
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/opt-v1

# 轻量训练（显存不足）
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/light --backbone resnet50 --crop-size 512

# 高分辨率训练（追求极致）
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/hires --crop-size 1024 --batch-size 2
```

### 推理
```powershell
# 基础推理
python predict.py --images test/images --checkpoint runs/opt-v1/best.pt --output submission/masks --tta

# 最强推理
python predict.py --images test/images --checkpoint runs/opt-v1/best.pt --output submission/masks --tta --multi-scale
```

### 验证
```powershell
python validate_submission.py --images test/images --masks submission/masks --zip submission/submission.zip
```

---

## 🔧 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--crop-size` | 768 | 512/768/1024 |
| `--batch-size` | 4 | 显存不足改2 |
| `--backbone` | resnet101 | resnet50/resnet101 |
| `--epochs` | 50 | 总轮数 |
| `--lr` | 3e-4 | 学习率 |
| `--patience` | 10 | 早停耐心值 |
| `--use-class-weights` | True | 类别权重 |

---

## 💡 故障排除

**CUDA out of memory**
```powershell
--batch-size 2 --crop-size 512
```

**训练太慢**
```powershell
--workers 8
```

**验证 mIoU 波动大**
```powershell
--val-ratio 0.15
```

**恢复训练**
```powershell
--resume runs/opt-v1/last.pt --epochs 60
```

---

## 📊 预期时间

| 任务 | RTX 3060 | RTX 4090 |
|------|----------|----------|
| 训练（50轮，768） | ~5小时 | ~2小时 |
| 推理（500张，TTA） | ~15分钟 | ~5分钟 |
| 推理（多尺度+TTA） | ~45分钟 | ~15分钟 |

---

## 🎓 关键提示

1. ✅ **首次运行**会下载预训练权重（~170MB）
2. ✅ **类别权重**会自动计算并打印
3. ✅ **最佳模型**保存在 `best.pt`
4. ✅ **最新模型**保存在 `last.pt`
5. ✅ **提交文件**是 `submission/submission.zip`

---

## 🏆 目标成绩

- **基线 mIoU**: 65-75%
- **优化后 mIoU**: 72-80%
- **目标排名**: 前 10-20%

---

**Good Luck! 🚀**
