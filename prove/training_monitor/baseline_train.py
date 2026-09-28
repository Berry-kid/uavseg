from __future__ import annotations

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
    compute_class_weights, 
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
    p = argparse.ArgumentParser(description="Train the optimized UAV segmentation baseline")
    p.add_argument("--data-root", type=Path, required=True, help="Directory containing images/ and masks/")
    p.add_argument("--output", type=Path, default=Path("runs/deeplabv3-opt"))
    p.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    p.add_argument("--crop-size", type=int, default=768, help="Crop size for training (512/768/1024)")
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    p.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=2026, help="Random seed")
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights")
    p.add_argument("--resume", type=Path, help="Checkpoint to resume from")
    p.add_argument("--backbone", type=str, default="resnet101", choices=["resnet50", "resnet101"],
                   help="Model backbone")
    p.add_argument("--use-class-weights", action="store_true", default=True,
                   help="Use inverse frequency class weights")
    p.add_argument("--patience", type=int, default=10, 
                   help="Early stopping patience (set 0 to disable)")
    p.add_argument("--keep-best", type=int, default=3,
                   help="Keep the top K ordinary-val checkpoints for final B-TTA selection")
    p.add_argument("--tta-batch-size", type=int, default=2,
                   help="Final validation/test sliding-window batch size")
    p.add_argument("--skip-final-tta", action="store_true",
                   help="Defer final selection; run select_tta.py later")
    return p.parse_args(argv)


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
    if min(args.epochs, args.batch_size, args.crop_size, args.keep_best, args.tta_batch_size) < 1:
        raise ValueError("Epochs, batch sizes, crop size and keep-best must be positive")
    if not 0 < args.val_ratio < 1 or args.workers < 0:
        raise ValueError("Invalid validation ratio or worker count")
    seed_everything(args.seed)
    
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; inference may use --device cpu.")
    
    image_dir, mask_dir = args.data_root / "images", args.data_root / "masks"
    names = sorted(p.name for p in image_dir.glob("*.png"))
    missing = [n for n in names if not (mask_dir / n).is_file()]
    
    if not names or missing:
        raise FileNotFoundError(f"Invalid dataset: {len(names)} images, {len(missing)} masks missing")
    
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output / name).exists() for name in
                               ("best.pt", "last.pt", "candidates/index.json", "best_tta.pt")):
        raise ValueError("Run output already contains weights. Use a new --output or --resume.")
    resume_checkpoint = read_checkpoint(args.resume) if args.resume else None
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
        train_names, val_names = create_stratified_split(
            names, image_dir, mask_dir, args.val_ratio, args.seed
        )
        split_path.write_text(
            json.dumps({"train": train_names, "val": val_names}), 
            encoding="utf-8"
        )
    
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
    
    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        print("Computing class weights...")
        class_weights = compute_class_weights(args.data_root, train_names).to(device).float()
    
    # Optimizer and scheduler with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Warmup + polynomial decay
    warmup_epochs = min(5, args.epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=max(1, args.epochs - warmup_epochs), power=0.9
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
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "warmup_scheduler" in checkpoint:
            warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
        if "scheduler" in checkpoint:
            poly_scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_val_miou", checkpoint.get("best_miou", checkpoint.get("miou", -1.0))))
        patience_counter = int(checkpoint.get("patience_counter", 0))
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

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for images, masks, _ in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
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
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        # Update scheduler
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            poly_scheduler.step()
        
        # Validation
        model.eval()
        cm = torch.zeros(9, 9, dtype=torch.int64, device=device)
        
        with torch.inference_mode():
            for images, masks, _ in tqdm(val_loader, desc="Validate"):
                images, masks = images.to(device), masks.to(device)
                pred = model(images)["out"].argmax(1)
                cm += confusion_matrix(pred, masks)
        
        miou, per_class = mean_iou(cm)
        avg_loss = running_loss / len(train_loader)
        
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

        # Early stopping only monitors ordinary val. Final TTA selection runs after this loop.
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best ordinary val mIoU: {best_miou:.5f}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best ordinary val mIoU: {best_miou:.5f}")
    print(f"Model saved to: {args.output}")


def main(argv=None):
    args = parse_args(argv)
    run_training(args)
    # Let all optimizer/model/last-batch references go before loading candidates.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.skip_final_tta:
        print("Final B-TTA selection deferred. Run select_tta.py when ready.")
    else:
        select_candidates(args.output, args.data_root,
                          InferenceConfig(batch_size=args.tta_batch_size), device="cuda")


if __name__ == "__main__":
    main()
