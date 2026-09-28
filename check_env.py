#!/usr/bin/env python
"""Environment check script for UAV segmentation project."""

import sys
from pathlib import Path

def check_python():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_dependencies():
    """Check required packages."""
    required = {
        'torch': '2.3',
        'torchvision': '0.18',
        'numpy': '1.26',
        'Pillow': '10.0',
        'tqdm': '4.66',
    }
    
    all_ok = True
    for package, min_version in required.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {package:12s} {version}")
        except ImportError:
            print(f"❌ {package:12s} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA available: {gpu_name} ({gpu_count} GPU(s), {gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU (slow)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_data(data_root: Path):
    """Check dataset structure."""
    if not data_root.exists():
        print(f"❌ Data directory not found: {data_root}")
        return False
    
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    images = list(images_dir.glob("*.png"))
    masks = list(masks_dir.glob("*.png"))
    
    print(f"✅ Found {len(images)} images, {len(masks)} masks")
    
    if len(images) == 0:
        print("❌ No images found")
        return False
    
    if len(masks) == 0:
        print("❌ No masks found")
        return False
    
    # Check for missing masks
    image_names = set(p.name for p in images)
    mask_names = set(p.name for p in masks)
    missing = image_names - mask_names
    
    if missing:
        print(f"⚠️  {len(missing)} images without masks")
    
    return True

def main():
    print("=" * 60)
    print("  UAV Segmentation - Environment Check")
    print("=" * 60)
    print()
    
    # Check Python
    print("1. Checking Python...")
    python_ok = check_python()
    print()
    
    # Check dependencies
    print("2. Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # Check CUDA
    print("3. Checking CUDA...")
    cuda_ok = check_cuda()
    print()
    
    # Check data
    print("4. Checking dataset...")
    data_root = Path("2026-低空图像语义分割赛道-训练集/train/train")
    data_ok = check_data(data_root)
    print()
    
    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    
    if python_ok and deps_ok:
        print("✅ Environment is ready for training")
        if cuda_ok:
            print("✅ GPU acceleration available")
        else:
            print("⚠️  Running on CPU (will be slow)")
        
        if data_ok:
            print("✅ Dataset is valid")
            print()
            print("You can start training with:")
            print(f'  python train.py --data-root "{data_root}" --output runs/opt-v1')
        else:
            print("⚠️  Please check your dataset path")
    else:
        print("❌ Environment is NOT ready")
        print()
        print("Please install dependencies:")
        print("  python -m pip install -r requirements.txt")
    
    print()

if __name__ == "__main__":
    main()
