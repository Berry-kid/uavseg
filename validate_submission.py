from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description="Validate competition submission contents")
    p.add_argument("--images", type=Path, required=True, help="Test images directory")
    p.add_argument("--masks", type=Path, required=True, help="Predicted masks directory")
    p.add_argument("--zip", type=Path, help="Submission zip file to validate")
    args = p.parse_args()
    
    # Get expected filenames
    expected = sorted(p.name for p in args.images.glob("*.png"))
    actual = sorted(p.name for p in args.masks.glob("*.png"))
    
    if expected != actual:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        msg = f"Filename mismatch: expected={len(expected)}, actual={len(actual)}"
        if missing:
            msg += f"\nMissing: {sorted(missing)[:10]}"
        if extra:
            msg += f"\nExtra: {sorted(extra)[:10]}"
        raise ValueError(msg)
    
    # Validate each mask
    print(f"Validating {len(expected)} masks...")
    for name in expected:
        with Image.open(args.images / name) as src, Image.open(args.masks / name) as mask:
            # Check mode and size
            if mask.mode != "L":
                raise ValueError(f"{name}: invalid mode {mask.mode}, expected L")
            if mask.size != src.size:
                raise ValueError(f"{name}: size mismatch mask={mask.size}, image={src.size}")
            
            # Check pixel values
            values = np.unique(np.asarray(mask))
            if values.size and (values[0] < 0 or values[-1] > 8):
                raise ValueError(f"{name}: invalid IDs {values.tolist()}, expected [0, 8]")
    
    # Validate zip if provided
    if args.zip:
        print(f"Validating zip: {args.zip}")
        with zipfile.ZipFile(args.zip) as zf:
            members = sorted(n for n in zf.namelist() if not n.endswith("/"))
        
        if members != expected:
            missing = set(expected) - set(members)
            extra = set(members) - set(expected)
            msg = "ZIP structure error:"
            if missing:
                msg += f"\nMissing: {sorted(missing)[:10]}"
            if extra:
                msg += f"\nExtra: {sorted(extra)[:10]}"
            msg += "\nZIP must contain PNG files directly at its root (no subdirectories)"
            raise ValueError(msg)
    
    print(f"✓ OK: {len(expected)} single-channel masks")
    print(f"✓ All filenames match")
    print(f"✓ All sizes match")
    print(f"✓ All pixel values in [0, 8]")
    if args.zip:
        print(f"✓ ZIP structure valid")


if __name__ == "__main__":
    main()
