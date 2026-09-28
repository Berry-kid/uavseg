# 普通验证筛选候选，B-TTA 选择最终模型

## 默认流程

1. 每轮训练完成后，沿用固定验证集的中心裁剪评估，不开 TTA。这个普通 val mIoU 用于早停和候选排名。
2. 默认保留普通 val 分数最高的 3 个轮次，不要求每个轮次都刷新历史最好分数。分数相同优先保留更早轮次。超过数量时只淘汰本流程管理的候选文件。
3. 正常训练结束或早停后，对保留的全部候选执行完整验证图像上的 B-TTA。
4. 按 TTA val mIoU 选择一份最终权重，保存为 best_tta.pt。它是单一模型，不融合候选权重。
5. test 读取 best_tta.pt 内保存的推理参数，并调用与最终验证相同的推理入口。

普通 val 仍是快速中心裁剪指标。最终 TTA val 才对应完整图像上的提交推理方式，两个指标单独记录，不能把差值全部解释为 TTA 收益。

## B 方案参数

| 项目 | 默认值 |
|---|---|
| 尺度 | 0.75、1.0、1.25 |
| 水平翻转 | 开启，各尺度均使用 |
| 滑窗 | 768 |
| 重叠 | 192 |
| 窗口 batch size | 2 |
| 精度 | CUDA float16 AMP 前向，float32 累加 |
| 输入 | 完整原始图像，保留原尺寸 |
| mIoU | 累计全验证集混淆矩阵，忽略真实标签 0，平均类别 1—8 |

验证与 test 共享 uavseg/inference.py 和原有滑窗函数。最终权重已保存配置时，若命令行要求不同的尺度、翻转、窗口、批次或 AMP，预测入口会明确报错，防止静默改变评估方案。

## 开始新的训练

先进入项目目录，并使用装有本项目 PyTorch 依赖的环境。新训练请使用新的输出目录，避免覆盖已完成的实验。

~~~powershell
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/b-tta-v1 --epochs 50 --keep-best 3
~~~

默认每轮不开 TTA，最后自动进行候选 B 验证。候选数量可用 --keep-best 调整。最终验证的窗口 batch size 可用 --tta-batch-size 调整，该数值会同时保存给 test 使用。

本机已有 PyTorch 环境，也可以通过 PowerShell 入口运行：

~~~powershell
.\run_training.ps1 -PythonExe "D:\anaconda\envs\pytorch\python.exe"
~~~

这个入口默认使用新的 runs/b-tta-v1 输出目录。它不会自动启动测试集预测。

## 测试集预测

训练和最终选择都成功完成后：

~~~powershell
python predict.py --images images --checkpoint runs/b-tta-v1/best_tta.pt --output submission_b/masks
~~~

无需重复写 --tta、--multi-scale 或 --amp，参数自动从最终权重读取。压缩包默认位于 submission_b/submission.zip。

输出仍为原尺寸、同名、类别 ID 0—8 的单通道 PNG。请选择新的结果目录，避免将旧 A 方案的结果与新 B 方案混合。

对于不含最终选择配置的旧权重，predict.py 现在默认使用 B。旧权重可以显式使用 --no-multi-scale 或 --no-tta 做对照；最终选出的 best_tta.pt 会检查参数一致性。旧权重在 CPU 上运行时会使用 FP32；以 CUDA AMP 选出的 best_tta.pt 需要同样的 GPU 精度配置。

## 中断与恢复

继续中断的训练，优先使用同一实验的 last.pt，保留原来的数据、骨干和裁剪尺寸：

~~~powershell
python train.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/b-tta-v1 --epochs 50 --keep-best 3 --resume runs/b-tta-v1/last.pt
~~~

验证集划分保持固定；候选名单与普通 val 历史会继续使用。候选或最终权重是仅供推理的精简文件，不能替代 last.pt 恢复优化器。

需要暂时推迟最后的 TTA 验证时，训练增加 --skip-final-tta。之后单独运行：

~~~powershell
python select_tta.py --run-dir runs/b-tta-v1 --data-root "2026-低空图像语义分割赛道-训练集/train/train"
~~~

最终验证中断后也可重新执行同一命令。完整评估过的候选，只有在权重、配置、划分、数据文件状态及推理代码均未改变时才复用结果；未完成的候选会重新评估。若训练时调整了 --tta-batch-size，手动选择时也应传入相同的 --batch-size。

历史训练只保留 best.pt 和 last.pt，无法找回已经丢失的其他轮次。恢复旧实验时，只能从仍存在的权重和之后的新轮次积累候选。

## 输出文件

| 文件 | 用途 |
|---|---|
| last.pt | 最近一轮完整训练状态，含优化器、调度器和早停计数 |
| best.pt | 普通 val 最好的完整训练状态 |
| val_history.json | 每轮普通 val 分数、各类 IoU 和训练 loss |
| candidates/index.json | 按普通 val 排序的候选名单及验证口径 |
| candidates/epoch_NNNN.pt | 候选模型参数，不含优化器，以减少占用 |
| tta_eval/epoch_NNNN.json | 每个候选的完整 B 验证结果 |
| tta_selection.json | 全部候选的普通/TTA分数与最终选择 |
| final_inference.json | 最终推理配置的可读副本 |
| best_tta.pt | 最终提交用的单一权重，含普通分数、TTA 分数及推理配置 |

字段 val_miou / best_val_miou 表示普通验证；tta_miou 表示完整 B 验证。旧字段 miou / best_miou 保留普通验证的含义，不会被 TTA 分数覆盖。

## 验证范围

流程测试覆盖候选排名和淘汰、断点恢复、早停后自动选择、TTA 改变最佳轮次、缓存失效、分数分离以及 test 配置一致性。另外使用现有真实权重和两张实际验证图片核对推理入口；这是代码回归检查，不是重新训练或重新报告完整验证集成绩。

如需运行流程测试：

~~~powershell
python -B -m unittest discover -s tests -p test_tta_workflow.py -v
~~~

最后评估的总时间随候选数量增加，约相当于候选数量次完整 B 验证。保留普通 val 的前 3 名是速度与选优覆盖范围的折中，不保证包含所有历史轮次中 TTA 得分最高的权重。

