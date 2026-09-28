# 类别误报实验：只调整 CE 类别权重

这一轮检验更温和的类别权重是否能减少误报。正式收益需要训练后验证；本目录内的 smoke 测试只证明代码行为，不代表比赛分数。

## 已确认的基准问题

当前第62轮最佳模型，完整699张val + 固定TTA，mIoU为0.76812083；线上test2总分由用户提供为65.6259，没有线上分类成绩。

| 类别 | 精确率 | 召回率 | 主要误报来源 |
|---|---:|---:|---|
| 荒地 | 61.55% | 75.66% | 背景占该类误报65.40% |
| 车辆 | 75.40% | 97.49% | 背景54.71%、道路24.80%、森林/植被18.33% |

这些来自本地val混淆矩阵，不等于已查明test2的分类掉分原因。详见 reference/baseline.json 和 reference/BASELINE.md。

## 这次改变什么

默认 `--class-weighting inverse_sqrt`：
- 原来是 `1 / frequency`，这次是 `1 / sqrt(frequency)`，有效类别1–8均值归一为1。
- 车辆相对背景：36.057倍 → 6.005倍；荒地相对背景：15.339倍 → 3.916倍。
- 主分支CE和辅助分支CE一起换权重。
- Dice、BN、数据增强、模型、划分、70轮预算、学习率3e-5、batch4、crop768和最终TTA全部沿用本次70轮基准。
- 不使用上一轮BN校准副本。新实验默认使用原来的ImageNet预训练骨干初始化。
- 忽略类0的loss权重为0，输出logit0和预测argmax的行为不变。
- 权重只统计固定train名单里的mask，原split原始字节及顺序保留。

还支持 `inverse` 对照组和 `none` 未加权CE。当前先跑一次 `inverse_sqrt`，不要同时混入其他策略。

## 远程使用

将 `qw-advice-class-fp.zip` 解压到一个新的项目目录，再把已有数据放到该目录，或用FP_DATA_ROOT指向原数据目录。包内有代码、固定划分和基准统计，不含数据、训练权重或BN校准权重。

在具有CUDA的PyTorch环境、项目根目录下执行：

```bash
bash prove/class_fp/run_ablation.sh inverse_sqrt
```

脚本自动寻找以下候选目录中包含images/和masks/的目录：
- `2026-低空图像语义分割赛道-训练集/train`
- `2026-低空图像语义分割赛道-训练集/train/train`
- `datasets/train/train`

如果数据在别处：

```bash
FP_DATA_ROOT="/实际路径/包含images和masks的目录" bash prove/class_fp/run_ablation.sh inverse_sqrt
```

需要指定Python时使用 `FP_PYTHON=/实际路径/python`。默认结果目录为 `prove/class_fp/runs/inverse_sqrt-v1`；每个新实验必须使用新的空输出目录，可通过 `FP_RUN` 指定。

脚本会依次检查环境、训练70轮、选出前三个普通val候选并进行完整val+TTA选优、生成与原基准的误报对照报告。不会自动跑test2或提交比赛。

等效训练指令（data-root请填包含images/masks的那层）：

```bash
python -u -B prove/class_fp/train_fp.py \
  --data-root "2026-低空图像语义分割赛道-训练集/train" \
  --split-file prove/class_fp/reference/split.json \
  --output prove/class_fp/runs/inverse_sqrt-v1 \
  --class-weighting inverse_sqrt \
  --epochs 70 --lr 3e-5 --batch-size 4 --crop-size 768 \
  --backbone resnet101 --workers 4 --seed 2026 --patience 0 \
  --keep-best 3 --tta-batch-size 2 \
  --train-eval-images 256 --train-eval-every 1
```

手动生成最终对照：

```bash
python -B prove/class_fp/compare_runs.py --run prove/class_fp/runs/inverse_sqrt-v1
```

## 看哪些结果

- `class_weights.json`：训练像素计数、权重模式和实际归一权重。
- `monitor/dashboard.html`：原训练曲线，以及precision/recall曲线。
- `monitor/epochs.jsonl`：每轮各类别precision、recall、TP/FP/FN及支持像素数。
- `monitor/false_positives/epoch_NNNN.json`：普通val及固定训练子集的逐类误报来源。
- `monitor/final_tta_false_positives.json`：最终选中模型的完整val+TTA误报诊断。
- `fp_comparison/COMPARISON.md`：与现有基准的mIoU、IoU、precision/recall、FP/FN对照。
- `best_tta.pt`：此次模型选优结果；需要以后预测时再使用。

优先看完整val+TTA的mIoU和类别IoU；精确率提升但召回率大幅下降、总mIoU下降，不算成功。不同权重方案的加权CE loss标尺不同，不能因为新loss更低就认定模型更好。局部val改善不保证test2同幅度涨分。

## 中断恢复

只支持恢复当前同一实验。保持所有训练参数、权重模式和输出目录不变：

```bash
FP_RESUME="prove/class_fp/runs/inverse_sqrt-v1/last.pt" bash prove/class_fp/run_ablation.sh inverse_sqrt
```

恢复时核对模式、划分、训练计数、归一权重、checkpoint元数据、学习率和其他配置。此处不承诺与完全不中断训练逐位一致。不要从其他实验或BN校准权重恢复这次实验。

## 验证范围

- 10项误报诊断测试：GT0忽略、预测0计漏检、矩阵方向、无定义比率、JSON/CSV/TensorBoard/页面及最终TTA记录。
- 9项训练/权重测试：旧inverse控制组与原入口训练3轮逐张量一致；平方根权重确实改变梯度与更新；只读取train标签；恢复一致性及拒绝错配；实际TinyModel多尺度翻转TTA/候选选优。
- 真实6297张训练mask计数重现原有效类权重。
- 原始训练、推理代码和已有权重均未修改。

