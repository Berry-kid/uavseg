# 已完成50轮后的追加训练

旧 best.pt 的优化器学习率已经为 0。直接 --resume 会恢复这个值，命令行 --lr 不会覆盖它。
本版新增 --restart-optimizer：保留模型、原 split.json 和轮次编号，重启 AdamW、
AMP scaler、早停计数及学习率调度。该选项要求一个新的输出目录。

以旧权重位于 runs/deeplabv3-opt、数据根目录为 datasets/train/train 为例：

```bash
python -u -B prove/training_monitor/train_monitored.py \
  --data-root datasets/train/train \
  --resume runs/deeplabv3-opt/best.pt \
  --restart-optimizer \
  --output runs/continue-v1 \
  --epochs 70 --lr 3e-5 \
  --batch-size 4 --crop-size 768 --backbone resnet101 --workers 4 \
  --seed 2026 --patience 0 --keep-best 3 --tta-batch-size 2 \
  --train-eval-images 256 --train-eval-every 1 --log-every 50
```

这是追加20轮（51–70）的试验配置，不保证比旧权重更好。基础 LR 为3e-5，
新阶段前5轮 warmup 从3e-6开始，然后按剩余轮数多项式衰减到0。
旧权重进入同一普通val候选池，最终TTA从保留候选中选择。

数据要求：本入口仍使用单个带标签数据根目录，其 images/ 和 masks/ 必须覆盖原
split.json 的全部训练与验证样本。原 split.json 必须与旧 best.pt 放在同一目录。
独立 train/ 和 val/ 目录尚不受此入口直接支持，不能把无标签test当作val。

若这一新阶段中断，恢复时使用同一数据/训练参数以及 --epochs 70，但去掉
--restart-optimizer，将 --resume 改为 runs/continue-v1/last.pt。
否则会拒绝在已有输出目录重新启动优化器。正常断点恢复不允许悄悄改变目标总轮数。
