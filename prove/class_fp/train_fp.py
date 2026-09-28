from __future__ import annotations

import os
import sys
import time
from pathlib import Path as _Path
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
PROJECT_ROOT = _Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prove.training_monitor.monitoring import LossMeter, numeric_losses, eval_losses, evaluate_train_subset
from prove.class_fp.diagnostics import FalsePositiveMonitor as TrainingMonitor
from prove.class_fp.weights import count_train_pixels, weights_from_counts, weight_report
ACTIVE_MONITOR = None


import numpy as np
import PIL.Image as Image

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from uavseg.core import (
    build_model, 
    confusion_matrix, 
    dice_loss, 
    mean_iou, 
    seed_everything,
    read_checkpoint,
)
from uavseg.data import SegmentationDataset
from uavseg.inference import InferenceConfig
from uavseg.selection import CandidatePool, save_checkpoint, select_candidates, val_protocol, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Controlled class false-positive ablation; only CE class weighting changes")
    p.add_argument("--data-root", type=Path, required=True, help="Directory containing images/ and masks/")
    p.add_argument("--output", type=Path, default=Path("prove/class_fp/runs/inverse-sqrt-v1"))
    p.add_argument("--epochs", type=int, default=70, help="Total training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    p.add_argument("--crop-size", type=int, default=768, help="Crop size for training (512/768/1024)")
    p.add_argument("--lr", type=float, default=3e-5, help="Initial learning rate")
    p.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=2026, help="Random seed")
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights")
    p.add_argument("--resume", type=Path, help="Checkpoint to resume from")
    p.add_argument("--restart-optimizer", action="store_true",
                   help="With --resume and a fresh output: keep model/split/epoch, restart AdamW, AMP and LR schedule over remaining epochs")
    p.add_argument("--backbone", type=str, default="resnet101", choices=["resnet50", "resnet101"],
                   help="Model backbone")
    p.add_argument("--class-weighting", choices=["inverse", "inverse_sqrt", "none"],
                   default="inverse_sqrt", help="Change only the main/auxiliary CE class weights")
    p.add_argument("--split-file", type=Path,
                   default=_Path(__file__).resolve().parent / "reference/split.json",
                   help="Original fixed split; copied byte-for-byte for a new run")
    p.add_argument("--patience", type=int, default=10, 
                   help="Early stopping patience (set 0 to disable)")
    p.add_argument("--keep-best", type=int, default=3,
                   help="Keep the top K ordinary-val checkpoints for final B-TTA selection")
    p.add_argument("--tta-batch-size", type=int, default=2,
                   help="Final validation/test sliding-window batch size")
    p.add_argument("--skip-final-tta", action="store_true",
                   help="Defer final selection; run select_tta.py later")
    p.add_argument("--log-every", type=int, default=50, help="Record a batch sample every N training batches")
    p.add_argument("--train-eval-images", type=int, default=256,
                   help="Fixed no-augmentation training subset: 0 disables; -1 evaluates all training images")
    p.add_argument("--train-eval-every", type=int, default=1, help="Evaluate the fixed training subset every N epochs")
    p.add_argument("--no-tensorboard", action="store_true", help="Keep JSON/CSV/HTML without TensorBoard events")
    args = p.parse_args(argv)
    args.use_class_weights = args.class_weighting != "none"
    return args


def create_stratified_split(names: list[str], image_dir: Path, mask_dir: Path, 
                            val_ratio: float, seed: int):
    """Create stratified train/val split based on image content."""
    # Analyze each image's class distribution
    image_classes = {}
    for name in names:
        mask_path = mask_dir / name
        if mask_path.exists():
            mask = torch.from_numpy(
                np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)
            )
            # Get unique classes present (excluding ignore)
            classes = torch.unique(mask[mask != 0]).tolist()
            image_classes[name] = set(classes)
    
    # Simple stratification: ensure each class appears in val set
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(names), generator=generator).tolist()
    
    split_idx = max(1, int(len(names) * val_ratio))
    val_names = [names[i] for i in order[:split_idx]]
    train_names = [names[i] for i in order[split_idx:]]
    
    print(f"Stratified split: {len(train_names)} train, {len(val_names)} val")
    
    return train_names, val_names


def run_training(args):
    global ACTIVE_MONITOR
    if args.log_every < 1 or args.train_eval_every < 1 or args.train_eval_images < -1:
        raise ValueError("Invalid monitoring interval or train-eval sample count")
    if min(args.epochs, args.batch_size, args.crop_size, args.keep_best, args.tta_batch_size) < 1:
        raise ValueError("Epochs, batch sizes, crop size and keep-best must be positive")
    if not 0 < args.val_ratio < 1 or args.workers < 0:
        raise ValueError("Invalid validation ratio or worker count")
    if args.restart_optimizer and not args.resume:
        raise ValueError("--restart-optimizer requires --resume")
    seed_everything(args.seed)
    
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; inference may use --device cpu.")
    
    image_dir, mask_dir = args.data_root / "images", args.data_root / "masks"
    names = sorted(p.name for p in image_dir.glob("*.png"))
    missing = [n for n in names if not (mask_dir / n).is_file()]
    
    if not names or missing:
        raise FileNotFoundError(f"Invalid dataset: {len(names)} images, {len(missing)} masks missing")
    
    # A new ablation never inherits rankings, scheduler state or BN calibration.
    if args.restart_optimizer:
        raise ValueError("This entry uses fresh ablations; --resume only recovers an interrupted identical experiment")
    if args.resume and args.output.resolve() != args.resume.resolve().parent:
        raise ValueError("Resume this experiment in its original output directory")
    if args.resume and not (args.output / "class_weights.json").is_file():
        raise ValueError("Resume requires this experiment's saved class_weights.json")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Fresh ablations require an empty or new output directory")
    if not args.resume and not args.split_file.is_file():
        raise FileNotFoundError("The fixed --split-file is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output / name).exists() for name in
                               ("best.pt", "last.pt", "candidates/index.json", "best_tta.pt")):
        raise ValueError("Run output already contains weights. Use a new --output or --resume.")
    resume_checkpoint = read_checkpoint(args.resume) if args.resume else None
    schedule_start_epoch = 1
    if resume_checkpoint is not None:
        saved_args = resume_checkpoint["args"]
        keys = ("class_weighting", "seed", "lr", "batch_size", "crop_size", "backbone",
                "epochs", "workers", "train_eval_images", "train_eval_every", "keep_best",
                "tta_batch_size", "patience", "no_pretrained")
        if resume_checkpoint.get("checkpoint_kind") != "training":
            raise ValueError("Only this experiment's training checkpoint can resume")
        for key in keys:
            if saved_args.get(key) != getattr(args, key):
                raise ValueError(f"Resume would change {key}; use a fresh experiment")
        if args.split_file.is_file():
            if args.split_file.read_bytes() != (args.output / "split.json").read_bytes():
                raise ValueError("Explicit split file differs from the saved experiment split")
    if resume_checkpoint is not None:
        completed_epoch = int(resume_checkpoint["epoch"])
        planned_epochs = int(resume_checkpoint["args"]["epochs"])
        if args.restart_optimizer:
            if args.output.resolve() == args.resume.resolve().parent:
                raise ValueError("--restart-optimizer requires a new output directory")
            if any((args.output / name).exists() for name in
                   ("best.pt", "last.pt", "candidates/index.json", "best_tta.pt", "monitor/epochs.jsonl")):
                raise ValueError("Restart output already contains an experiment; use a new directory")
            if args.epochs <= completed_epoch:
                raise ValueError("--epochs must exceed the checkpoint epoch when restarting")
            schedule_start_epoch = completed_epoch + 1
        else:
            if args.epochs != planned_epochs:
                raise ValueError("Changing the total epoch budget requires --restart-optimizer and a new output directory")
            schedule_start_epoch = int(resume_checkpoint.get("lr_schedule_start_epoch", 1))
            if not 1 <= schedule_start_epoch <= completed_epoch:
                raise ValueError("Invalid saved LR schedule origin")
            groups = resume_checkpoint.get("optimizer", {}).get("param_groups", [])
            if completed_epoch < args.epochs and groups and all(float(g["lr"]) <= 0 for g in groups):
                raise ValueError("Checkpoint LR is zero; use --restart-optimizer in a new output directory")
    if resume_checkpoint is not None:
        if resume_checkpoint.get("checkpoint_kind") in ("candidate", "final_tta"):
            raise ValueError("Resume training from last.pt/best.pt; candidate and best_tta files are inference-only")
        for key in ("backbone", "crop_size"):
            if resume_checkpoint["args"].get(key) != getattr(args, key):
                raise ValueError(f"Resume would change {key}; ordinary-val rankings would not be comparable")
    split_path = args.output / "split.json"
    if args.resume and not split_path.is_file():
        source_split = args.resume.parent / "split.json"
        if not source_split.is_file():
            raise FileNotFoundError("Resume requires the original split.json")
        split_path.write_bytes(source_split.read_bytes())

    # Load or create the fixed train/val split before entering the epoch loop.
    if args.resume:
        saved_split = json.loads(split_path.read_text(encoding="utf-8"))
        train_names, val_names = saved_split["train"], saved_split["val"]
        unknown = (set(train_names) | set(val_names)) - set(names)
        if unknown:
            raise ValueError(f"Saved split contains {len(unknown)} files absent from the dataset")
    else:
        split_path.write_bytes(args.split_file.read_bytes())
        saved_split = json.loads(split_path.read_text(encoding="utf-8"))
        train_names, val_names = saved_split["train"], saved_split["val"]
    if any(Path(n).name != n or "/" in n or "\\" in n for n in train_names + val_names):
        raise ValueError("Split names must be basenames")
    print(f"Fixed split: {len(train_names)} train, {len(val_names)} val")
    
    if (not train_names or not val_names or len(set(train_names)) != len(train_names)
            or len(set(val_names)) != len(val_names) or set(train_names) & set(val_names)
            or set(train_names) | set(val_names) != set(names)):
        raise ValueError("Saved split must be disjoint and exactly cover the dataset")

    # Create datasets; ordinary val remains a fixed center crop with no TTA.
    train_ds = SegmentationDataset(image_dir, mask_dir, train_names, args.crop_size, True)
    val_ds = SegmentationDataset(image_dir, mask_dir, val_names, args.crop_size, False)
    
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=args.workers, 
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, num_workers=args.workers, pin_memory=True
    )
    
    if len(train_loader) == 0:
        raise ValueError("Training batch size exceeds the available training samples")
    device = torch.device("cuda")
    
    # Build model
    model = build_model(
        pretrained=not args.no_pretrained and not args.resume, 
        backbone=args.backbone
    ).to(device)
    
    # Count only the fixed training masks. No val/test pixels enter the weights.
    counts = count_train_pixels(args.data_root, train_names, workers=max(1, args.workers),
        progress=lambda done,total: print(f"Train pixel counts: {done}/{total}", flush=True))
    class_weights = weights_from_counts(counts, args.class_weighting)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    weighting = weight_report(counts, train_names, args.class_weighting)
    weighting_path = args.output / "class_weights.json"
    if args.resume:
        saved_weighting = json.loads(weighting_path.read_text(encoding="utf-8"))
        if saved_weighting != weighting:
            raise ValueError("Training counts or class weighting changed since the saved run")
        if resume_checkpoint.get("class_weighting") != weighting:
            raise ValueError("Checkpoint class weights do not match this experiment")
    else:
        write_json(weighting_path, weighting)
    print(f"Class weighting: {args.class_weighting}; weights: {weighting['class_weights_ids_0_to_8']}")
    
    # Optimizer and scheduler with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Warmup + polynomial decay
    schedule_epochs = args.epochs - schedule_start_epoch + 1
    warmup_epochs = min(5, schedule_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=max(1, schedule_epochs - warmup_epochs), power=0.9
    )
    
    scaler = GradScaler("cuda")
    
    # Training state
    best_miou = -1.0
    start_epoch = 1
    patience_counter = 0
    
    # Resume from checkpoint
    if args.resume:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"])
        if not args.restart_optimizer:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("warmup_scheduler") is not None:
                warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
            if "scheduler" in checkpoint:
                poly_scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        else:
            print(f"Restarting optimizer/LR schedule at epoch {schedule_start_epoch}; "
                  f"base LR={args.lr:g}, remaining epochs={schedule_epochs}, warmup={warmup_epochs}")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_val_miou", checkpoint.get("best_miou", checkpoint.get("miou", -1.0))))
        patience_counter = 0 if args.restart_optimizer else int(checkpoint.get("patience_counter", 0))
        print(f"Resuming from epoch {start_epoch} (best ordinary val mIoU={best_miou:.5f})")

    protocol = val_protocol(split_path, args.crop_size)
    if resume_checkpoint is not None:
        original_split = args.resume.parent / "split.json"
        if not original_split.is_file() or val_protocol(original_split, args.crop_size) != protocol:
            raise ValueError("Resume requires exactly the original validation split")
        if resume_checkpoint.get("val_protocol", protocol) != protocol:
            raise ValueError("Checkpoint ordinary-val protocol differs from this run")
    pool = CandidatePool(args.output, args.keep_best, protocol,
                         resume_epoch=start_epoch - 1 if args.resume else None)
    if resume_checkpoint is not None:
        pool.consider(resume_checkpoint)
        previous_best_path = args.resume.parent / "best.pt"
        if previous_best_path.is_file() and previous_best_path.resolve() != args.resume.resolve():
            previous_best = read_checkpoint(previous_best_path)
            if (int(previous_best["epoch"]) <= start_epoch - 1
                    and previous_best["args"].get("crop_size") == args.crop_size
                    and previous_best["args"].get("backbone") == args.backbone):
                pool.consider(previous_best)
            del previous_best
        del checkpoint, resume_checkpoint
    history_path = args.output / "val_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if args.resume and history_path.exists() else []
    history = [row for row in history if row["epoch"] < start_epoch]

    # A separate generator prevents diagnostic evaluation from consuming the
    # training DataLoader's global random sequence.
    count = len(train_names) if args.train_eval_images == -1 else min(args.train_eval_images, len(train_names))
    subset_order = torch.randperm(len(train_names), generator=torch.Generator().manual_seed(args.seed + 913)).tolist()
    train_eval_names = sorted(train_names[i] for i in subset_order[:count])
    train_eval_loader = None
    if train_eval_names:
        train_eval_loader = DataLoader(
            SegmentationDataset(image_dir, mask_dir, train_eval_names, args.crop_size, False),
            batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed + 1913))
    monitor = TrainingMonitor(args, protocol, start_epoch, class_weights, train_eval_names, PROJECT_ROOT)
    ACTIVE_MONITOR = monitor
    print(f"Monitoring enabled; fixed train-eval subset: {len(train_eval_names)} images")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats(device)
        monitor.set_state("training", epoch=epoch)
        train_meter = LossMeter()
        skipped_steps = 0
        train_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        
        for batch_index, (images, masks, _) in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"), 1):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast("cuda"):
                outputs = model(images)
                
                # Main loss: CrossEntropy + Dice
                ce_weight = class_weights if class_weights is not None else None
                main_ce = F.cross_entropy(outputs["out"], masks, ignore_index=0, weight=ce_weight)
                main_dice = dice_loss(outputs["out"], masks)
                main_loss = main_ce + 0.5 * main_dice
                
                # Auxiliary loss
                aux_loss = F.cross_entropy(outputs["aux"], masks, ignore_index=0, weight=ce_weight)
                
                loss = main_loss + 0.4 * aux_loss
            
            loss_values = numeric_losses(main_ce, main_dice, aux_loss, loss)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            skipped = scaler.get_scale() < scale_before
            skipped_steps += int(skipped)
            train_meter.add(loss_values, images.shape[0])
            monitor.batch(epoch, batch_index, len(train_loader), loss_values, optimizer, scaler, skipped)
        
        train_seconds = time.perf_counter() - train_started

        # Update scheduler
        if epoch - schedule_start_epoch + 1 <= warmup_epochs:
            warmup_scheduler.step()
        else:
            poly_scheduler.step()
        
        # Validation
        val_started = time.perf_counter()
        val_meter = LossMeter()
        monitor.set_state("validation", epoch=epoch)
        model.eval()
        cm = torch.zeros(9, 9, dtype=torch.int64, device=device)
        
        with torch.inference_mode():
            for images, masks, _ in tqdm(val_loader, desc="Validate"):
                images, masks = images.to(device), masks.to(device)
                val_outputs = model(images)
                pred = val_outputs["out"].argmax(1)
                cm += confusion_matrix(pred, masks)
                values = eval_losses(val_outputs, masks, class_weights)
                if values is not None:
                    val_meter.add(values, len(images))
        
        miou, per_class = mean_iou(cm)
        avg_loss = running_loss / len(train_loader)
        val_seconds = time.perf_counter() - val_started
        train_eval = None
        if train_eval_loader is not None and (epoch % args.train_eval_every == 0 or epoch == args.epochs):
            monitor.set_state("train_subset_evaluation", epoch=epoch)
            train_eval = evaluate_train_subset(model, train_eval_loader, device, class_weights)
            print(f"  Fixed train-eval mIoU(no augmentation): {train_eval['miou']:.5f}")
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_mIoU(no TTA)={miou:.5f}")
        print(f"  Per-class IoU: {[f'{x:.3f}' for x in per_class]}")
        
        # Early stopping check
        improved = miou > best_miou
        if improved:
            best_miou = miou
            patience_counter = 0
            print(f"  ✓ New best ordinary val mIoU: {best_miou:.5f}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Save checkpoints
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": poly_scheduler.state_dict(),
            "warmup_scheduler": warmup_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "lr_schedule_start_epoch": schedule_start_epoch,
            "class_weighting": weighting,
            "val_miou": miou,
            "best_val_miou": best_miou,
            "val_per_class_iou": per_class,
            "val_protocol": protocol,
            "miou": miou,  # Compatibility alias: always ordinary val, never TTA.
            "best_miou": best_miou,
            "patience_counter": patience_counter,
            "checkpoint_kind": "training",
            "args": vars(args),
        }
        save_checkpoint(args.output / "last.pt", state)
        
        if improved:
            save_checkpoint(args.output / "best.pt", state)
            print(f"  Saved ordinary-val best checkpoint to {args.output / 'best.pt'}")
        
        pool.consider(state)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_miou": miou,
                        "val_per_class_iou": per_class, "tta": False})
        write_json(history_path, history)
        print(f"  Retained ordinary-val candidate epochs: {[e['epoch'] for e in pool.entries]}")

        monitor.epoch(
            {"epoch": epoch, "lr": lr_used, "lr_next_epoch": optimizer.param_groups[0]["lr"],
             "val_miou": miou, "best_val_miou": best_miou, "val_per_class_iou": per_class,
             "train_seconds": train_seconds, "val_seconds": val_seconds,
             "train_eval_seconds": train_eval["seconds"] if train_eval else 0.0,
             "epoch_seconds": time.perf_counter() - epoch_started,
             "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
             "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
             "amp_scale": float(scaler.get_scale()), "optimizer_steps_skipped": skipped_steps,
             "improved": improved, "patience_counter": patience_counter,
             "train_images_seen": train_meter.count, "val_images": len(val_ds)},
            train_meter.result(), val_meter.result(), train_eval, cm)

        # Early stopping only monitors ordinary val. Final TTA selection runs after this loop.
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best ordinary val mIoU: {best_miou:.5f}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best ordinary val mIoU: {best_miou:.5f}")
    print(f"Model saved to: {args.output}")


def main(argv=None):
    global ACTIVE_MONITOR
    ACTIVE_MONITOR = None
    args = parse_args(argv)
    try:
        run_training(args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.skip_final_tta:
            print("Final B-TTA selection deferred. Run select_tta.py when ready.")
            ACTIVE_MONITOR.set_state("completed_tta_deferred")
        else:
            ACTIVE_MONITOR.set_state("final_tta_selection")
            result = select_candidates(args.output, args.data_root,
                                       InferenceConfig(batch_size=args.tta_batch_size), device="cuda")
            ACTIVE_MONITOR.final_selection(result)
            ACTIVE_MONITOR.set_state("completed")
    except BaseException as exc:
        if ACTIVE_MONITOR is not None:
            ACTIVE_MONITOR.fail(exc)
        raise
    finally:
        if ACTIVE_MONITOR is not None:
            ACTIVE_MONITOR.close()


if __name__ == "__main__":
    main()
