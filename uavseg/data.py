from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Paired PNG dataset with synchronized, dependency-free augmentation."""

    def __init__(self, image_dir: Path, mask_dir: Path, names: list[str], 
                 crop_size: int, augment: bool):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.names = names
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = Image.open(self.image_dir / name).convert("RGB")
        mask = Image.open(self.mask_dir / name).convert("L")
        
        if self.augment:
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._center_crop(image, mask)
        
        # Normalize to [0, 1] and apply ImageNet stats
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        
        y = torch.from_numpy(np.asarray(mask, dtype=np.uint8).astype(np.int64, copy=True))
        return x, y, name

    def _center_crop(self, image: Image.Image, mask: Image.Image):
        """Center crop for validation."""
        size = min(self.crop_size, image.width, image.height)
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        box = (left, top, left + size, top + size)
        return image.crop(box), mask.crop(box)

    def _augment(self, image: Image.Image, mask: Image.Image):
        """Apply data augmentation."""
        # Random horizontal flip
        if random.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Random vertical flip
        if random.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randrange(4)
        if k:
            image, mask = image.rotate(90 * k, expand=False), mask.rotate(90 * k, expand=False)
        
        # Random scaling
        scale = random.uniform(0.75, 1.35)
        side = max(self.crop_size, int(round(image.width * scale)))
        image = image.resize((side, side), Image.Resampling.BILINEAR)
        mask = mask.resize((side, side), Image.Resampling.NEAREST)
        
        # Random crop
        left = random.randint(0, side - self.crop_size)
        top = random.randint(0, side - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        image, mask = image.crop(box), mask.crop(box)
        
        # Color jittering
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        
        return image, mask
