# 训练监控使用说明

此目录提供独立训练入口 `train_monitored.py`。原来的 `train.py`、`run_training.ps1`、数据和已有权重没有修改。继续运行原入口不会启用这些监控；下一次训练请使用本目录的入口。

## 启动

在项目根目录执行。下面是通用命令，请使用安装了 PyTorch 的环境，并保留你实际实验所用的训练参数：

```bash
python -B prove/training_monitor/train_monitored.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/monitored-v1 --epochs 50 --batch-size 4 --crop-size 768 --backbone resnet101 --keep-best 3
```

本机 Windows 的 PyTorch 环境是 `D:\anaconda\envs\pytorch\python.exe`。PowerShell 示例使用 batch size 2、workers 0；是否能容纳该配置取决于运行时可用显存：

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' -B prove/training_monitor/train_monitored.py --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output runs/monitored-v1 --epochs 50 --batch-size 2 --crop-size 768 --backbone resnet101 --workers 0 --keep-best 3
```

每次新实验使用新的 `--output` 目录。训练入口保留当前版本训练代码的 top-K 候选权重与最终 TTA 选择流程；默认训练后继续做最终 TTA 评估。可传 `--skip-final-tta` 暂缓这一步，再使用项目原有 `select_tta.py`。

这是当前训练器的独立副本，之后如果修改原 `train.py`，需要同步检查本入口，不能假定它自动获得原文件的更新。

## 记录的内容

| 数据 | 用途 |
| --- | --- |
| 每轮训练/验证总 loss、主分支 CE、Dice、辅助分支 CE | 查看优化过程和损失组成 |
| 普通 val mIoU、类别 1–8 的 IoU、9×9 混淆矩阵 | 确定哪些类别出现退化或混淆 |
| 固定训练子集的 mIoU、loss、类别 IoU、混淆矩阵 | 在同一评估方式下比较 train 与 val |
| 本轮学习率、下一轮学习率、最佳 val 分数、早停计数 | 关联学习率和最佳权重选择 |
| 每轮耗时、训练/验证/子集评估耗时、GPU 峰值显存 | 查看成本和资源需求 |
| 抽样 batch loss、AMP scale、优化器跳步 | 排查数值异常 |
| 实际参数、数据路径、固定划分协议、源码哈希、软件版本、GPU | 复现实验 |
| 恢复训练、训练状态、异常堆栈、最终 TTA 结果 | 保留实验过程记录 |

沿用当前项目指标实现：真实标签 0 不参与评估，mIoU 对类别 1–8 求平均。类别顺序为背景、建筑、道路、水体、裸地、植被、农田、车辆。

默认从训练划分中固定抽取 256 张样本，使用与普通 val 相同的中心裁剪、无数据增强、FP32、无 TTA 评估。它是训练子集分数，不能当作完整训练集分数。该评估不做反向传播，并使用独立随机生成器，以免改变训练抽样顺序。

训练时的 loss 含随机增强且使用 AMP，和评估 loss 的条件不同。判断泛化差距时优先比较 `train_eval_miou` 与 `val_miou` 的走势；两者存在差距本身不能证明 test 分数下降的原因。

总损失公式沿用原代码：`CE + 0.5 × Dice + 0.4 × auxiliary CE`。每轮 loss 为按样本数加权的 batch 均值，不是另行计算的全数据集像素加权损失。

## 日志位置与查看方式

全部监控输出位于 `<output>/monitor/`：

- `epochs.csv`：每轮汇总，可用于后续分析。
- `epochs.jsonl`：当前有效轮次的主记录。
- `epochs/epoch_NNNN.json`：单轮详情，包括混淆矩阵。
- `steps.jsonl`、`progress.json`：抽样 batch 记录和最近一次记录。
- `dashboard.html`：离线曲线页面，直接用浏览器打开；每 15 秒刷新，曲线每完成一轮更新。
- `tensorboard/`：TensorBoard 事件。
- `run_info.json`、`sessions/`：首次配置及各次启动配置。
- `train_eval_names.json`：固定训练评估子集。
- `status.json`、`events.jsonl`、`failure_*.json`：状态、事件及发生异常时的诊断。
- `final_tta.json`：完成最终 TTA 后生成，独立于普通 val 曲线。

TensorBoard 为可选依赖，缺少它时仍会输出 CSV、JSON 和离线页面。使用当前 Python 环境启动：

```bash
python -m tensorboard.main --logdir runs/monitored-v1/monitor/tensorboard --port 6006
```

然后在该机器浏览器打开 `http://localhost:6006`。远程服务器可使用平台端口映射或 SSH 转发。本功能不会自动启动服务。若当前环境未安装 TensorBoard，可自行执行 `python -m pip install tensorboard`，或传 `--no-tensorboard`。

## 控制额外开销

- `--train-eval-images 256`：默认每轮评估固定的最多 256 张训练图。
- `--train-eval-every 5`：每 5 轮评估一次；到设定的最终轮次也会评估。
- `--train-eval-images -1`：评估完整训练集，耗时明显增加。
- `--train-eval-images 0`：关闭额外训练子集评估，保留训练 loss 和 val 监控。
- `--log-every 50`：默认每 50 个 batch 记录一次；首尾 batch 也会记录。

## 已完成旧训练后的追加训练

请先阅读 [追加训练说明](README_CONTINUE.md)。已完成50轮的旧权重学习率为0，需要显式重启优化器和调度器。

## 断点恢复

使用相同训练参数和本监控入口，加上：

```text
--resume runs/monitored-v1/last.pt
```

`--epochs` 表示目标总轮数，不是额外训练轮数。优先从 `last.pt` 恢复，并保持原总轮数与学习率调度配置。固定训练子集与数据划分会检查一致性。

恢复时只保留早于恢复起点的有效轮次，重复轮次不会混入汇总；回退时旧汇总归档到 `archives/`。batch 与事件日志保留各会话记录，按 `session` 区分。回退后尚未覆盖的旧单轮详情文件可能仍存在，分析时以 `epochs.jsonl` / `epochs.csv` 为准。

恢复到新的输出目录时会读取原监控汇总及固定子集信息，但不会复制所有旧 batch、事件、单轮详情。既有训练器没有保存完整随机状态，因此此处的恢复测试保证日志轮次一致，不承诺与不中断训练逐位相同。

可以从旧训练的权重继续训练并开始记录，但监控无法补回旧轮次未保存的 loss、显存、耗时等数据。

## 之后提供哪些结果

训练结束后，将本次 `monitor` 文件夹以及同一输出目录的 `val_history.json`、`tta_selection.json`（如已生成）一起提供。再附上实际提交 ZIP 对应的权重、推理配置和线上成绩，就能对照曲线、类别误差和评估方式决定下一步修改。

## 已完成的验证

`test_results.json` 记录 6 项合成数据测试全部通过：加入监控后的模型参数更新与原版逐张量一致；loss、类别 IoU、CSV、TensorBoard 记录；中断/恢复/回退；最终 TTA 独立记录；非有限 loss 异常诊断；零学习率权重追加训练与新调度的断点恢复；显式重启参数校验。

`smoke_output/` 中全部是小型合成数据及测试结果，与比赛数据、正式分数无关。没有启动正式 50 轮训练。

`baseline_train.py` 是原训练器快照；`build_monitored_entry.py` 是构建工具；日常训练只使用 `train_monitored.py`，同目录保留 `monitoring.py`，并依赖项目的 `uavseg` 包。
