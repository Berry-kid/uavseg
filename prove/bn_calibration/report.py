"""Create the human-readable result of the fixed BN calibration experiment."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("experiment", type=Path)
args = p.parse_args()
folder = args.experiment.resolve()
result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
if result.get("complete") is not True:
    raise ValueError("Experiment is incomplete")
before, after = result["tta"]["before"], result["tta"]["after"]
delta = result["tta"]["delta_percentage_points"]
verdict = ("完整 val 的 TTA mIoU 提高；这仍需线上评测确认能否迁移到 test2。"
           if delta > 0 else
           "本次 BN 校准未提高完整 val 的 TTA mIoU，建议继续使用原权重。")
lines = [
    "# BN 统计校准对照结果", "",
    verdict, "",
    "本实验只检验一次固定的训练集 BN 统计重估方法。没有训练优化器步骤，没有改变类别权重、数据增强或 TTA 配置。", "",
    "| 评估项目 | 原始 best | BN 校准副本 | 变化 |",
    "|---|---:|---:|---:|",
]
for label, key, metric in [
        ("完整 val + TTA mIoU", "tta", "miou"),
        ("中心裁剪 val mIoU", "val", "miou"),
        ("中心裁剪 val loss", "val", "loss"),
        ("固定训练子集 mIoU", "train_subset", "miou"),
        ("固定训练子集 loss", "train_subset", "loss")]:
    if key == "tta":
        a, b = before[metric], after[metric]
    else:
        a, b = result["before_center"][key][metric], result["after_center"][key][metric]
    change = f"{(b-a)*100:+.4f} 个百分点" if metric == "miou" else f"{b-a:+.6f}"
    lines.append(f"| {label} | {a:.6f} | {b:.6f} | {change} |")
lines += ["", "## 完整 val + TTA 的类别结果", "",
          "| ID | 类别 | 原始 IoU | 校准 IoU | 变化（百分点） | 原始精确率 | 校准精确率 | 原始召回率 | 校准召回率 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for i, name in enumerate(before["class_names"]):
    vals = [before["per_class_iou"][i], after["per_class_iou"][i],
            (after["per_class_iou"][i]-before["per_class_iou"][i])*100,
            before["precision"][i], after["precision"][i],
            before["recall"][i], after["recall"][i]]
    lines.append(f"| {i+1} | {name} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:+.4f} | {vals[3]:.6f} | {vals[4]:.6f} | {vals[5]:.6f} | {vals[6]:.6f} |")
lines += [
    "", "## 实验条件与检查", "",
    f"- 来源：第 {config['epoch']} 轮 best_tta.pt。",
    f"- 来源权重 SHA256：{config['checkpoint_sha256']}。",
    f"- 校准读取原 split 的全部 {result['calibration']['images']} 张训练图像，共 {result['calibration']['batches']} 批。",
    "- 无随机增强，中心裁剪 768，FP32，无梯度；Dropout 关闭，仅 BatchNorm 使用训练模式。",
    "- batch size 4，最后 5 张拆成 3 + 2；每张图像恰好参与一次。",
    "- 重置并重新估计各层运行均值/方差；使用按图像数加权的批内统计平均，不是精确总体方差合并。",
    f"- 全部 {result['calibration']['bn_layers']} 个 BN 层均完成校准，修改 {result['invariants']['changed_buffer_count']} 个统计 buffer。",
    "- 所有可学习参数（包括 BN 的缩放与偏置）和非 BN 统计 buffer 逐元素完全不变。",
    "- 原始 checkpoint、源码和 split 的 SHA256 未变，所读数据文件的大小与修改时间未变。",
    "- 两个模型在同一机器上重新评估全部 699 张 val；NumPy 与项目混淆矩阵逐图核对一致。",
    f"- TTA：{json.dumps(config['inference_config'], ensure_ascii=False)}。",
    f"- 环境：{config['gpu']}，PyTorch {config['torch']}，CUDA {config['cuda']}。",
    "- 没有预测 test2、提交比赛，或执行类别权重等后续实验。",
    "", "## 结果的边界", "",
    "本次只验证一个预先固定的 BN 校准方案。校准变差不能排除所有 BN 相关问题；校准变好也不能证明 test2 会同幅度提高。单个 checkpoint 的前后对照不能独自解释整个训练期间 loss 波动的原因。",
    "",
    "详细数据：results.json；逐图混淆矩阵：tta_partial.npz；独立实验权重：bn_calibrated.pt。",
]
(folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(folder / "REPORT.md")
print(json.dumps({"before_tta_miou": before["miou"], "after_tta_miou": after["miou"], "delta_pp": delta}, ensure_ascii=False))

