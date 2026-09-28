"""Compare A/B TTA with one frozen checkpoint and the saved validation split.
Writes only to --output. Existing project source and checkpoints remain read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
from PIL import Image
import torch

LABELS = ["Ignore", "Background", "Building", "Road", "Water",
          "Barren", "Vegetation", "Agricultural", "Vehicle"]
ZH = ["忽略", "背景", "建筑", "道路", "水体", "裸地", "植被", "农田", "车辆"]


def dump(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(cm):
    tp = np.diag(cm).astype(np.float64)
    gt, pred = cm.sum(1), cm.sum(0)
    union = gt + pred - tp
    ious = tp[1:] / np.maximum(union[1:], 1)
    return {"miou": float(ious.mean()), "valid_pixels": int(cm.sum()),
            "per_class": [{"id": c, "name": LABELS[c], "name_zh": ZH[c],
                           "iou": float(ious[c-1]), "intersection": int(tp[c]),
                           "union": int(union[c]), "ground_truth_pixels": int(gt[c])}
                          for c in range(1, 9)],
            "confusion_matrix": cm.tolist()}


def independent_cm(pred, target):
    valid = target != 0
    ids = target[valid].astype(np.int64) * 9 + pred[valid].astype(np.int64)
    return np.bincount(ids, minlength=81).reshape(9, 9)


def save_state(out, names, done, cms_a, cms_b, times_a, times_b):
    tmp = out / "partial.npz.tmp"
    with tmp.open("wb") as f:
        np.savez_compressed(f, names=np.asarray(names), processed=np.asarray(done),
                            cm_a=cms_a, cm_b=cms_b, seconds_a=times_a, seconds_b=times_b)
    tmp.replace(out / "partial.npz")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    root, out = args.project.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(out / "cuda_cache")
    os.environ["TORCH_HOME"] = str(out / "torch_cache")
    sys.path.insert(0, str(root))
    import torchvision
    from uavseg.core import load_checkpoint, image_tensor, sliding_logits, confusion_matrix, mean_iou, seed_everything

    cp = root / "runs/deeplabv3-opt/best.pt"
    sp = cp.parent / "split.json"
    data = root / "2026-低空图像语义分割赛道-训练集/train/train"
    protected = [cp, sp, root / "train.py", root / "predict.py",
                 root / "uavseg/core.py", root / "uavseg/data.py"]
    hashes_before = {str(path): sha256(path) for path in protected}
    split = json.loads(sp.read_text(encoding="utf-8"))
    train, names = split["train"], split["val"]
    all_names = sorted(path.name for path in (data / "images").glob("*.png"))
    assert len(set(train)) == len(train) and len(set(names)) == len(names)
    assert not set(train) & set(names)
    assert set(train) | set(names) == set(all_names)
    assert all(Path(n).name == n for n in names)
    input_stats = {}
    ignored = 0
    for name in names:
        ip, mp = data / "images" / name, data / "masks" / name
        with Image.open(ip) as image, Image.open(mp) as mask:
            assert image.size == mask.size == (1024, 1024), name
            assert mask.mode == "L", (name, mask.mode)
            target = np.asarray(mask)
            assert target.dtype == np.uint8 and target.max() <= 8, name
            ignored += int(np.count_nonzero(target == 0))
        for path in (ip, mp):
            stat = path.stat()
            input_stats[str(path)] = [stat.st_size, stat.st_mtime_ns]

    assert torch.cuda.is_available(), "CUDA required"
    torch.set_num_threads(4)
    seed_everything(2026)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model, checkpoint = load_checkpoint(cp, device, backbone="resnet101")
    model.aux_classifier = None
    model.to(memory_format=torch.channels_last)
    model.eval()
    assert checkpoint["args"]["backbone"] == "resnet101"
    order = torch.randperm(len(all_names), generator=torch.Generator().manual_seed(checkpoint["args"]["seed"])).tolist()
    split_count = max(1, int(len(all_names) * checkpoint["args"]["val_ratio"]))
    assert names == [all_names[i] for i in order[:split_count]]
    assert train == [all_names[i] for i in order[split_count:]]
    config = {
        "checkpoint": str(cp), "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_recorded_miou": checkpoint["miou"], "source_sha256": hashes_before,
        "data_root": str(data), "split_path": str(sp),
        "train_images": len(train), "validation_images": len(names),
        "split_matches_checkpoint_seed": True, "names": names, "input_file_stats": input_stats,
        "input_size": [1024, 1024], "center_crop": False,
        "tile": 768, "overlap": 192, "patch_batch_size": 2,
        "amp": True, "autocast_dtype": "float16", "logit_accumulation_dtype": "float32",
        "A": {"scales": [1.0], "horizontal_flip": True},
        "B": {"scales": [0.75, 1.0, 1.25], "horizontal_flip": True},
        "ignore_ground_truth": 0, "evaluated_ids": list(range(1, 9)),
        "python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "shared_scale": "Reuse A's 1.0-scale logits in B, preserving original core.py's summation order"}
    del checkpoint
    count = len(names)
    cms_a = np.zeros((count, 9, 9), dtype=np.int64)
    cms_b = np.zeros_like(cms_a)
    times_a, times_b = np.zeros(count), np.zeros(count)
    start_index = 0
    config_path = out / "run_config.json"
    if args.resume:
        assert json.loads(config_path.read_text(encoding="utf-8")) == config, "Inputs/config changed"
        with np.load(out / "partial.npz", allow_pickle=False) as saved:
            assert saved["names"].tolist() == names
            start_index = int(saved["processed"])
            cms_a, cms_b = saved["cm_a"], saved["cm_b"]
            times_a, times_b = saved["seconds_a"], saved["seconds_b"]
    else:
        if config_path.exists() or (out / "partial.npz").exists():
            raise FileExistsError("Use --resume or a new output directory")
        dump(config_path, config)
    print(json.dumps({k: config[k] for k in ("checkpoint_epoch", "checkpoint_recorded_miou",
          "validation_images", "A", "B", "tile", "overlap", "amp", "gpu")}, ensure_ascii=False), flush=True)

    @torch.inference_mode()
    def infer(image, scales):
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            output = sliding_logits(model, image, tile=768, overlap=192,
                                    tta=True, batch_size=2, scales=scales)
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Nonfinite logits")
        return output

    @torch.inference_mode()
    def pair(image):
        torch.cuda.synchronize()
        started = time.perf_counter()
        a = infer(image, [1.0])
        torch.cuda.synchronize()
        elapsed_a = time.perf_counter() - started
        extra_started = time.perf_counter()
        small, large = infer(image, [0.75]), infer(image, [1.25])
        b = (small + a + large) / 3.0
        torch.cuda.synchronize()
        elapsed_b = elapsed_a + time.perf_counter() - extra_started
        return a, b, elapsed_a, elapsed_b

    print("Warm-up and parity check against the unchanged project function...", flush=True)
    parity = []
    for name in names[:2]:
        image, _ = image_tensor(data / "images" / name, device)
        a, b, _, _ = pair(image)
        direct_b = infer(image, [0.75, 1.0, 1.25])
        diff = float((b - direct_b).abs().max().item())
        equal_masks = bool(torch.equal(b.argmax(0), direct_b.argmax(0)))
        assert torch.allclose(b, direct_b, rtol=1e-5, atol=1e-5), diff
        assert equal_masks, "Shared-scale reuse changed predictions"
        parity.append({"name": name, "B_max_logit_difference": diff, "B_mask_exact_match": equal_masks})
        del image, a, b, direct_b
    dump(out / "shared_scale_equivalence.json", parity)
    print("Parity checks passed. Comparing all validation images.", flush=True)
    started = time.perf_counter()
    for index in range(start_index, count):
        name = names[index]
        image, _ = image_tensor(data / "images" / name, device)
        with Image.open(data / "masks" / name) as mask:
            target = np.asarray(mask, dtype=np.uint8).copy()
        a, b, seconds_a, seconds_b = pair(image)
        pa, pb = a.argmax(0), b.argmax(0)
        ca = independent_cm(pa.cpu().numpy(), target)
        cb = independent_cm(pb.cpu().numpy(), target)
        target_gpu = torch.from_numpy(target.astype(np.int64)).to(device)
        assert np.array_equal(ca, confusion_matrix(pa, target_gpu).cpu().numpy())
        assert np.array_equal(cb, confusion_matrix(pb, target_gpu).cpu().numpy())
        cms_a[index], cms_b[index] = ca, cb
        times_a[index], times_b[index] = seconds_a, seconds_b
        done = index + 1
        if done % 20 == 0 or done == count or done == start_index + 1:
            elapsed = time.perf_counter() - started
            eta = elapsed / (done - start_index) * (count - done)
            dump(out / "progress.json", {"processed": done, "total": count,
                 "elapsed_this_run_seconds": elapsed, "estimated_remaining_seconds": eta,
                 "complete": done == count})
            save_state(out, names, done, cms_a, cms_b, times_a, times_b)
            print(f"Compared {done}/{count} | elapsed={elapsed:.1f}s | ETA={eta:.1f}s", flush=True)
        del image, a, b, pa, pb, target_gpu

    ma, mb = metrics(cms_a.sum(0)), metrics(cms_b.sum(0))
    for m in (ma, mb):
        project_miou, _ = mean_iou(torch.tensor(m["confusion_matrix"], dtype=torch.int64))
        assert abs(project_miou - m["miou"]) < 1e-12
        assert m["valid_pixels"] == count * 1024 * 1024 - ignored
    ma.update({"scales": [1.0], "horizontal_flip": True, "inference_seconds": float(times_a.sum()),
               "mean_seconds_per_image": float(times_a.mean())})
    mb.update({"scales": [0.75, 1.0, 1.25], "horizontal_flip": True, "inference_seconds": float(times_b.sum()),
               "mean_seconds_per_image": float(times_b.mean())})
    hashes_after = {str(path): sha256(path) for path in protected}
    changed_inputs = []
    for path_string, previous in input_stats.items():
        stat = Path(path_string).stat()
        if previous != [stat.st_size, stat.st_mtime_ns]:
            changed_inputs.append(path_string)
    integrity = {"source_hashes_before": hashes_before, "source_hashes_after": hashes_after,
                 "source_hashes_match": hashes_before == hashes_after,
                 "validation_input_files_checked": len(input_stats), "changed_validation_inputs": changed_inputs}
    dump(out / "input_integrity.json", integrity)
    assert integrity["source_hashes_match"] and not changed_inputs
    delta = mb["miou"] - ma["miou"]
    result = {
        "complete": True, "checkpoint_epoch": config["checkpoint_epoch"],
        "validation_images": count, "evaluated_images": count, "A": ma, "B": mb,
        "delta_B_minus_A": delta, "delta_percentage_points": 100 * delta,
        "winner_on_this_validation_set": "B" if delta > 0 else ("A" if delta < 0 else "tie"),
        "B_to_A_inference_time_ratio": float(times_b.sum() / times_a.sum()),
        "per_class_delta_B_minus_A": [
            {"id": x["id"], "name": x["name"], "name_zh": x["name_zh"],
             "delta": y["iou"] - x["iou"], "delta_percentage_points": 100 * (y["iou"] - x["iou"])}
            for x, y in zip(ma["per_class"], mb["per_class"])],
        "independent_numpy_and_project_confusion_matrices_agree": True,
        "shared_scale_equivalence": parity, "inputs_and_sources_unchanged": True,
        "timing_note": "CUDA-synchronized warmed-up inference only; excludes I/O and metrics. B includes the reused 1.0-scale inference time. These are strategy inference-work timings, not independent end-to-end runs.",
        "limitations": [
            "Existing model-selection validation split, not an independent competition test set.",
            "Only scales differ; weights, full images, flip, tile, overlap, batch size and precision match.",
            "Validation results do not guarantee the same improvement on the test set.",
            "No training or changes to original training/production inference source files."]}
    dump(out / "results.json", result)
    (out / "partial.npz").replace(out / "per_image_confusion_matrices.npz")
    lines = [
        "# 固定权重的 A/B TTA 对比", "",
        f"同一份第 {config['checkpoint_epoch']} 轮权重，两种方案均完成全部 {count} 张原验证集完整图像评估。",
        "", "| 方案 | 尺度 | 水平翻转 | mIoU | 纯推理累计秒数 |", "|---|---|---|---:|---:|",
        f"| A | 1.0 | 开启 | {ma['miou']:.9f} | {times_a.sum():.2f} |",
        f"| B | 0.75、1.0、1.25 | 开启 | {mb['miou']:.9f} | {times_b.sum():.2f} |", "",
        f"B − A = {delta:+.9f}，即 **{100*delta:+.4f} 个百分点**。",
        f"本验证集总体 mIoU 更高的方案：**{result['winner_on_this_validation_set']}**。",
        f"B 累计纯推理时间约为 A 的 {result['B_to_A_inference_time_ratio']:.2f} 倍。", "",
        "| 类别 | A IoU | B IoU | B−A（百分点） |", "|---|---:|---:|---:|"]
    for x, y in zip(ma["per_class"], mb["per_class"]):
        lines.append(f"| {x['name_zh']} | {x['iou']:.6f} | {y['iou']:.6f} | {100*(y['iou']-x['iou']):+.4f} |")
    lines += [
        "", "## 对比条件",
        "- 两组权重、固定验证集名单、预处理和计算精度一致，仅尺度组合不同。",
        "- 完整 1024×1024 图像；滑窗 768，重叠 192，窗口 batch size 2。",
        "- 与当前 test 一致：AMP float16 前向、float32 累加、水平翻转、去除辅助输出头。",
        "- 全部验证图像累计混淆矩阵，忽略真实标签 0，对类别 1—8 的 IoU 等权平均。",
        "- 每张图都用独立 NumPy 与原项目 PyTorch 实现核对混淆矩阵，逐张一致。",
        "- 复用共享的 1.0 尺度；前两张图与原函数直接执行 B 的预测逐像素一致。",
        "- 经过预热并用 GPU 同步计时。时间不含读取和指标计算；B 包含共享尺度的时间，因此 A/B 时间相加不等于本次实际等待时间。",
        "- 原始训练/推理代码、权重和划分文件哈希未变，验证输入文件大小和修改时间未变。",
        "", "## 结果范围",
        "结论仅针对当前权重和固定验证集；不能保证测试集得到相同提升。没有重新训练，也没有更改每轮 val 的策略。",
        "", "## 证据文件",
        "- results.json：完整指标、各类差值和计时。",
        "- per_image_confusion_matrices.npz：逐图混淆矩阵与耗时。",
        "- run_config.json：完整参数、输入名单、源文件哈希和环境。",
        "- input_integrity.json：原文件检查。",
        "- shared_scale_equivalence.json：共享计算与原推理函数的数值对比。",
        "", "## 复现",
        "使用本项目的 PyTorch 环境运行 compare_tta.py，传入 --project 项目绝对路径以及 --output 新的结果目录。",
        "中断后可加 --resume，从原输出目录的 partial.npz 继续。已完成结果请使用新目录复现。"]
    (out / "对比报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "images": count,
          "A_miou": ma["miou"], "B_miou": mb["miou"], "B_minus_A_percentage_points": 100 * delta,
          "A_vehicle_iou": ma["per_class"][-1]["iou"], "B_vehicle_iou": mb["per_class"][-1]["iou"],
          "B_to_A_time_ratio": result["B_to_A_inference_time_ratio"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

