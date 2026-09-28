from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uavseg.core import load_checkpoint
from uavseg.inference import prepare_model, predict_mask, resolve_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict masks and create competition-ready zip")
    p.add_argument("--images", type=Path, required=True, help="Test images directory")
    p.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    p.add_argument("--output", type=Path, default=Path("submission/masks"), 
                   help="Output directory for masks")
    p.add_argument("--tile", type=int, default=None, help="Default: selected checkpoint profile, otherwise 768")
    p.add_argument("--overlap", type=int, default=None, help="Default: selected checkpoint profile, otherwise 192")
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None, help="Horizontal flip; defaults to B/profile")
    p.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=None, 
                   help="Enable multi-scale inference [0.75, 1.0, 1.25]")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Device to use")
    p.add_argument("--backbone", type=str, default=None, 
                   choices=["resnet50", "resnet101"], help="Model backbone")
    p.add_argument("--batch-size", type=int, default=None, 
                   help="Batch size for sliding window inference")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Precision defaults to selected profile/B")
    p.add_argument("--resume", action="store_true", help="Resume a matching inference run")
    p.add_argument("--zip", type=Path, help="Explicit output ZIP path")
    p.add_argument("--threads", type=int, default=4, help="CPU preprocessing threads")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = load_checkpoint(args.checkpoint, device, backbone=args.backbone)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}, ordinary val mIoU: "
          f"{checkpoint.get('val_miou', checkpoint.get('miou', '?'))}, "
          f"TTA val mIoU: {checkpoint.get('tta_miou', 'not evaluated')}")
    overrides = {"tile": args.tile, "overlap": args.overlap,
                 "batch_size": args.batch_size, "tta": args.tta, "amp": args.amp}
    if args.multi_scale is not None:
        overrides["scales"] = [0.75, 1.0, 1.25] if args.multi_scale else [1.0]
    if device.type == "cpu" and args.amp is None and not checkpoint.get("inference_config"):
        overrides["amp"] = False
    inference = resolve_config(checkpoint, overrides)
    model = prepare_model(model, device, inference)
    print(f"Final inference settings: {inference.to_dict()}")

    
    # Get test images
    names = sorted(p.name for p in args.images.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No PNG files found in {args.images}")
    
    print(f"Found {len(names)} test images")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    scales = list(inference.scales)

    with args.checkpoint.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
    config = {
        "images": str(args.images.resolve()), "names": names,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha256,
        "epoch": checkpoint.get("epoch"), "validation_miou": checkpoint.get("val_miou", checkpoint.get("miou")),
        "val_miou": checkpoint.get("val_miou", checkpoint.get("miou")),
        "tta_val_miou": checkpoint.get("tta_miou"),
        "backbone": checkpoint.get("args", {}).get("backbone", "resnet101"),
        "tile": inference.tile, "overlap": inference.overlap,
        "tta": inference.tta, "scales": scales, "amp": inference.amp,
        "device": args.device, "batch_size": inference.batch_size,
        "inference_config": inference.to_dict(),
        "torch": str(torch.__version__),
    }
    manifest_path = args.output.parent / "inference.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume requires exactly matching inference.json")
    else:
        if any(args.output.glob("*.png")):
            raise ValueError("Output masks already exist; use a new output folder or --resume")
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    del checkpoint
    started = time.perf_counter()
    predicted = 0
    
    # Predict
    for index, name in enumerate(names, 1):
        if args.resume and (args.output / name).exists():
            with Image.open(args.output / name) as existing, Image.open(args.images / name) as source:
                a = np.asarray(existing)
                if existing.mode == "L" and existing.size == source.size and a.max() <= 8:
                    continue
        prediction = predict_mask(model, args.images / name, device, inference)
        with Image.open(args.images / name) as source:
            shape = (source.height, source.width)
        mask = prediction.byte().cpu().numpy()
        
        # Validate prediction
        if mask.shape != shape or mask.min() < 0 or mask.max() > 8:
            raise ValueError(f"Invalid prediction for {name}: shape={mask.shape}, range=[{mask.min()}, {mask.max()}]")
        
        temporary = args.output / (name + ".tmp")
        Image.fromarray(mask).save(temporary, format="PNG")
        temporary.replace(args.output / name)
        predicted += 1
        if index == 1 or index % 25 == 0 or index == len(names):
            elapsed = time.perf_counter() - started
            eta = elapsed / predicted * (len(names) - index)
            print(f"Progress {index}/{len(names)} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)
    
    # Create submission zip
    zip_path = args.zip or args.output.parent / "submission.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating submission zip: {zip_path}")
    
    temporary_zip = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(args.output / name, arcname=name)
    temporary_zip.replace(zip_path)
    
    print(f"✓ Created {zip_path} with {len(names)} masks")
    print(f"✓ Submission ready!")


if __name__ == "__main__":
    main()
