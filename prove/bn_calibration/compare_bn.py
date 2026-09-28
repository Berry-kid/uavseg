"""Training-only BN recalibration, followed by same-host paired validation.
Original project files are read-only. All outputs must be inside prove/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from uavseg.core import (load_checkpoint, read_checkpoint, seed_everything,
                         confusion_matrix, mean_iou, image_tensor)
from uavseg.data import SegmentationDataset
from uavseg.inference import resolve_config, prepare_model, predict_logits
from prove.training_monitor.monitoring import eval_losses, LossMeter

LABELS = ["背景", "建筑", "道路", "水体", "荒地", "森林", "农田", "车辆"]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def balanced_batches(count, batch_size, seed):
    """Cover every sample once; pooled BN must never receive a singleton."""
    if count < 2 or batch_size < 3:
        raise ValueError("Need at least two images and batch size >= 3")
    order = list(range(count))
    random.Random(seed).shuffle(order)
    batches = [order[i:i + batch_size] for i in range(0, count, batch_size)]
    if len(batches[-1]) == 1:
        batches[-1].insert(0, batches[-2].pop())
    assert sorted(i for batch in batches for i in batch) == list(range(count))
    assert all(2 <= len(batch) <= batch_size for batch in batches)
    return batches


class CalibrationImages(Dataset):
    """Only image paths are available here; masks/val/test are never loaded."""
    def __init__(self, image_dir, names, crop):
        self.image_dir, self.names, self.crop = Path(image_dir), list(names), crop

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with Image.open(self.image_dir / self.names[index]) as im:
            im = im.convert("RGB")
            if im.size != (1024, 1024):
                raise ValueError(f"Unexpected calibration image size: {im.size}")
            size = min(self.crop, im.width, im.height)
            left, top = (im.width - size) // 2, (im.height - size) // 2
            im = im.crop((left, top, left + size, top + size))
            x = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1).copy()) / 255
        x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        return x


def bn_buffer_keys(model):
    return {f"{name}.{key}" if name else key
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key in ("running_mean", "running_var", "num_batches_tracked")}


def verify_bn_only(before, after, allowed):
    if before.keys() != after.keys():
        raise AssertionError("State keys changed")
    changed = []
    for key, value in after.items():
        value = value.detach().cpu()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite state: {key}")
        if not torch.equal(before[key], value):
            if key not in allowed:
                raise AssertionError(f"Non-BN-statistics tensor changed: {key}")
            changed.append(key)
        if key.endswith("running_var") and (value < 0).any():
            raise AssertionError(f"Negative BN variance: {key}")
    if not any(k.endswith(("running_mean", "running_var")) for k in changed):
        raise AssertionError("No BN mean/variance changed")
    return {"all_learned_parameters_unchanged": True,
            "all_non_bn_statistics_buffers_unchanged": True,
            "changed_buffers": changed, "changed_buffer_count": len(changed)}


@torch.no_grad()
def recalibrate(model, loader, device, progress=None):
    """Sample-weighted average of within-batch BN estimates, FP32, dropout off."""
    layers = [m for m in model.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not layers or any(not m.track_running_stats for m in layers):
        raise ValueError("Expected BN layers with running statistics")
    old_modes = {m: m.training for m in model.modules()}
    old_momenta = {m: m.momentum for m in layers}
    seen = batches = 0
    started = time.perf_counter()
    try:
        model.eval()
        for m in layers:
            m.reset_running_stats()
            m.train()
        for images in loader:
            n = len(images)
            if n < 2:
                raise ValueError("Singleton batch is invalid for pooled BatchNorm")
            for m in layers:
                m.momentum = n / (seen + n)
            outputs = model(images.to(device))
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
            elif isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            if not all(torch.isfinite(t).all().item() for t in outputs):
                raise FloatingPointError("Non-finite calibration output")
            seen += n
            batches += 1
            if progress and (batches == 1 or batches % 25 == 0 or batches == len(loader)):
                progress(seen, batches, time.perf_counter() - started)
        for m in layers:
            if int(m.num_batches_tracked) != batches:
                raise AssertionError("Some BN layers did not execute on every batch")
    finally:
        for m, momentum in old_momenta.items():
            m.momentum = momentum
        for m, training in old_modes.items():
            m.training = training
    return {"images": seen, "batches": batches, "bn_layers": len(layers),
            "seconds": time.perf_counter() - started,
            "precision": "FP32", "augmentation": False, "dropout": False,
            "statistic_estimator": "sample-weighted average of batch means and unbiased within-batch variances",
            "optimizer_steps": 0}


def summarize(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    gt, pred = cm.sum(1), cm.sum(0)
    iou = tp[1:] / np.maximum(gt[1:] + pred[1:] - tp[1:], 1)
    return {"miou": float(iou.mean()), "per_class_iou": iou.tolist(),
            "precision": (tp[1:] / np.maximum(pred[1:], 1)).tolist(),
            "recall": (tp[1:] / np.maximum(gt[1:], 1)).tolist(),
            "valid_pixels": int(cm.sum()), "class_names": LABELS,
            "confusion_matrix": cm.tolist()}


@torch.inference_mode()
def center_evaluate(model, loader, weights, device, stage, status):
    model.eval()
    cm = torch.zeros(9, 9, dtype=torch.int64, device=device)
    loss, count = LossMeter(), 0
    started = time.perf_counter()
    for index, (images, masks, _) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        if not all(torch.isfinite(t).all().item() for t in outputs.values()):
            raise FloatingPointError("Non-finite center evaluation")
        cm += confusion_matrix(outputs["out"].argmax(1), masks)
        values = eval_losses(outputs, masks, weights)
        if values is not None:
            loss.add(values, len(images))
        count += len(images)
        if index == 1 or index % 25 == 0 or index == len(loader):
            status(stage, count, len(loader.dataset), started)
    result = summarize(cm.cpu().numpy())
    project_miou, _ = mean_iou(cm)
    assert abs(project_miou - result["miou"]) < 1e-12
    return dict(result, **loss.result(), images=count,
                seconds=time.perf_counter() - started)


def stat_manifest(paths):
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def save_tta_partial(output, names, rows_before, rows_after, done):
    tmp = output / "tta_partial.npz.tmp"
    with tmp.open("wb") as f:
        np.savez_compressed(f, names=np.array(names), processed=done,
                            before=np.asarray(rows_before), after=np.asarray(rows_after))
    tmp.replace(output / "tta_partial.npz")


@torch.inference_mode()
def paired_tta(before_model, after_model, data, names, config, device, output, status):
    before_model = prepare_model(before_model, device, config)
    after_model = prepare_model(after_model, device, config)
    before_rows, after_rows = [], []
    started = time.perf_counter()
    for index, name in enumerate(names, 1):
        image, shape = image_tensor(data / "images" / name, device)
        with Image.open(data / "masks" / name) as mask:
            if mask.mode != "L":
                raise ValueError(f"Invalid mask mode: {name}")
            target = np.asarray(mask).copy()
        if target.shape != shape or target.max() > 8:
            raise ValueError(f"Invalid mask: {name}")
        target_gpu = torch.from_numpy(target.astype(np.int64)).to(device)
        for model, rows in ((before_model, before_rows), (after_model, after_rows)):
            pred = predict_logits(model, image, config).argmax(0)
            cm = confusion_matrix(pred, target_gpu).cpu().numpy()
            valid = target != 0
            ids = target[valid].astype(np.int64) * 9 + pred.cpu().numpy()[valid]
            independent = np.bincount(ids, minlength=81).reshape(9, 9)
            if not np.array_equal(cm, independent):
                raise AssertionError("Independent confusion matrices disagree")
            rows.append(cm)
        if index == 1 or index % 10 == 0 or index == len(names):
            save_tta_partial(output, names, before_rows, after_rows, index)
            status("paired_full_val_tta", index, len(names), started)
    result = {"before": summarize(np.sum(before_rows, axis=0)),
              "after": summarize(np.sum(after_rows, axis=0)),
              "images": len(names), "seconds": time.perf_counter() - started}
    result["delta_miou"] = result["after"]["miou"] - result["before"]["miou"]
    result["delta_percentage_points"] = result["delta_miou"] * 100
    result["per_class_delta"] = (np.array(result["after"]["per_class_iou"]) -
                                  np.array(result["before"]["per_class_iou"])).tolist()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    cp, data, output = args.checkpoint.resolve(), args.data_root.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "prove") or output == ROOT / "prove":
        raise ValueError("Output must be a new subdirectory of prove/")
    if output.exists():
        raise ValueError("Refusing to overwrite existing experiment directory")
    output.mkdir(parents=True)
    os.environ["CUDA_CACHE_PATH"] = str(output / "cuda_cache")
    os.environ["TORCH_HOME"] = str(output / "torch_cache")
    started_all = time.perf_counter()

    def status(stage, done, total, started):
        seconds = time.perf_counter() - started
        value = {"stage": stage, "processed": done, "total": total,
                 "seconds": seconds, "eta_seconds": seconds / done * (total - done) if done else None}
        write_json(output / "progress.json", value)
        print(f"{stage}: {done}/{total}, elapsed={seconds:.1f}s, ETA={value['eta_seconds']:.1f}s", flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required to match saved inference profile")
        torch.set_num_threads(4)
        seed_everything(args.seed)
        device = torch.device("cuda")
        split_path = cp.parent / "split.json"
        info_path = cp.parent / "monitor/run_info.json"
        train_eval_path = cp.parent / "monitor/train_eval_names.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        train_names, val_names = split["train"], split["val"]
        train_eval_names = json.loads(train_eval_path.read_text(encoding="utf-8"))
        if not isinstance(train_eval_names, list):
            raise ValueError("Unexpected train-eval name format")
        all_names = sorted(p.name for p in (data / "images").glob("*.png"))
        if (not train_names or not val_names or len(set(train_names)) != len(train_names)
                or len(set(val_names)) != len(val_names) or set(train_names) & set(val_names)
                or set(train_names) | set(val_names) != set(all_names)
                or not set(train_eval_names).issubset(train_names)):
            raise ValueError("Invalid saved split")
        if any(Path(n).name != n or "/" in n or "\\" in n for n in all_names + train_names + val_names):
            raise ValueError("Names must be basenames")
        protected = [cp, split_path, info_path, train_eval_path, ROOT / "train.py", ROOT / "predict.py",
                     *sorted((ROOT / "uavseg").glob("*.py")),
                     ROOT / "prove/training_monitor/monitoring.py"]
        hashes = {str(p): file_hash(p) for p in protected}
        inputs = [data / "images" / n for n in all_names] + [data / "masks" / n for n in val_names + train_eval_names]
        before_manifest = stat_manifest(inputs)
        write_json(output / "input_manifest.json", {"protected_sha256": hashes, "input_stats": before_manifest})
        model, checkpoint = load_checkpoint(cp, device)
        expected_split = checkpoint.get("val_protocol", {}).get("split_sha256")
        if expected_split != file_hash(split_path):
            raise ValueError("Checkpoint and split hash disagree")
        config = resolve_config(checkpoint)
        crop = int(checkpoint["args"]["crop_size"])
        weights = torch.tensor(info["class_weights"], dtype=torch.float32, device=device)
        if int(checkpoint["epoch"]) != 62:
            raise ValueError("This experiment expects the agreed current best at epoch 62")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        recipe = {"checkpoint": str(cp), "checkpoint_sha256": hashes[str(cp)],
                  "epoch": checkpoint["epoch"], "data_root": str(data), "split_sha256": expected_split,
                  "calibration_images": len(train_names), "validation_images": len(val_names),
                  "train_eval_images": len(train_eval_names), "crop_size": crop,
                  "calibration_batch_size": args.batch_size, "seed": args.seed,
                  "calibration_source": "saved train split only; images only; center crop; no augmentation",
                  "inference_config": config.to_dict(), "torch": str(torch.__version__),
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                  "python": sys.version, "script_sha256": file_hash(__file__),
                  "historical_val_miou": checkpoint.get("val_miou"),
                  "historical_tta_miou": checkpoint.get("tta_miou")}
        write_json(output / "config.json", recipe)
        write_json(output / "calibration_names.json", train_names)
        print(json.dumps(recipe, ensure_ascii=False, default=str), flush=True)

        def eval_loader(names):
            ds = SegmentationDataset(data / "images", data / "masks", names, crop, False)
            return DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

        val_loader, train_loader = eval_loader(val_names), eval_loader(train_eval_names)
        before_center = {
            "val": center_evaluate(model, val_loader, weights, device, "before_center_val", status),
            "train_subset": center_evaluate(model, train_loader, weights, device, "before_center_train", status)}
        write_json(output / "before_center.json", before_center)
        print("Before center val:", before_center["val"]["miou"], flush=True)
        bn_keys = bn_buffer_keys(model)
        batches = balanced_batches(len(train_names), args.batch_size, args.seed)
        loader = DataLoader(CalibrationImages(data / "images", train_names, crop),
                            batch_sampler=batches, num_workers=args.workers, pin_memory=True)
        cal_started = time.perf_counter()
        calibration = recalibrate(model, loader, device,
            lambda n, b, s: status("calibrate_train_bn", n, len(train_names), cal_started))
        if calibration["images"] != len(train_names):
            raise AssertionError("Calibration did not cover the complete train split")
        calibrated_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        invariants = verify_bn_only(checkpoint["model"], calibrated_state, bn_keys)
        write_json(output / "calibration.json", dict(calibration, **invariants))
        after_center = {
            "val": center_evaluate(model, val_loader, weights, device, "after_center_val", status),
            "train_subset": center_evaluate(model, train_loader, weights, device, "after_center_train", status)}
        write_json(output / "after_center.json", after_center)
        candidate = {"model": calibrated_state, "epoch": checkpoint["epoch"],
                     "args": dict(checkpoint["args"]), "val_protocol": checkpoint["val_protocol"],
                     "checkpoint_kind": "bn_calibrated_diagnostic",
                     "selected_by": "training_only_bn_calibration",
                     "miou": after_center["val"]["miou"], "val_miou": after_center["val"]["miou"],
                     "val_per_class_iou": after_center["val"]["per_class_iou"],
                     "inference_config": config.to_dict(),
                     "bn_calibration": dict(calibration, source_checkpoint_sha256=hashes[str(cp)],
                                            split_sha256=expected_split, seed=args.seed,
                                            crop_size=crop, batch_size=args.batch_size),
                     "source_scores": {"val_miou": checkpoint.get("val_miou"),
                                       "tta_miou": checkpoint.get("tta_miou")}}
        candidate_path = output / "bn_calibrated.pt"
        torch.save(candidate, candidate_path)
        baseline, baseline_checkpoint = load_checkpoint(cp, device)
        del baseline_checkpoint
        tta = paired_tta(baseline, model, data, val_names, config, device, output, status)
        write_json(output / "tta_results.json", tta)
        candidate.update(tta_miou=tta["after"]["miou"], tta_per_class_iou=tta["after"]["per_class_iou"])
        torch.save(candidate, candidate_path)
        saved = read_checkpoint(candidate_path)
        verify_bn_only(checkpoint["model"], saved["model"], bn_keys)
        after_hashes = {str(p): file_hash(p) for p in protected}
        if hashes != after_hashes or before_manifest != stat_manifest(inputs):
            raise AssertionError("Protected source/checkpoint or dataset input changed")
        result = {"complete": True, "calibration": calibration, "invariants": invariants,
                  "before_center": before_center, "after_center": after_center, "tta": tta,
                  "original_checkpoint_and_sources_unchanged": True, "input_stats_unchanged": True,
                  "calibrated_checkpoint": str(candidate_path),
                  "calibrated_checkpoint_sha256": file_hash(candidate_path),
                  "seconds": time.perf_counter() - started_all,
                  "limitation": "One fixed training-only BN calibration recipe; validation gains do not establish test gains."}
        write_json(output / "results.json", result)
        write_json(output / "progress.json", {"stage": "complete", "seconds": result["seconds"]})
        print("COMPLETE:", json.dumps({"before": tta["before"]["miou"], "after": tta["after"]["miou"],
                                     "delta_pp": tta["delta_percentage_points"]}), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"error": repr(exc), "seconds": time.perf_counter() - started_all})
        raise


if __name__ == "__main__":
    main()

