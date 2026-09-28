from __future__ import annotations

import random
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from tqdm import tqdm

NUM_CLASSES = 9
IGNORE_INDEX = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool = True, backbone: str = "resnet101"):
    """Build DeepLabV3 model with configurable backbone.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        backbone: Model backbone, either 'resnet50' or 'resnet101'
    """
    if backbone == "resnet101":
        weights_backbone = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        return deeplabv3_resnet101(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=NUM_CLASSES,
            aux_loss=True,
        )
    else:  # resnet50
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=NUM_CLASSES,
            aux_loss=True,
        )


def compute_class_weights(data_root: Path, names: list[str], num_classes: int = NUM_CLASSES):
    """Compute inverse frequency class weights for balanced training.
    
    Args:
        data_root: Root directory containing images/ and masks/
        names: List of image filenames
        num_classes: Number of classes (including ignore)
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    mask_dir = data_root / "masks"
    
    for name in tqdm(names, desc="Computing class weights"):
        mask = np.asarray(Image.open(mask_dir / name).convert("L"))
        for c in range(num_classes):
            pixel_counts[c] += (mask == c).sum()
    
    # Ignore class (0) gets weight 0
    pixel_counts[0] = 0
    
    # Inverse frequency weighting with smoothing
    weights = 1.0 / (pixel_counts.clamp_min(1))
    # Normalize so mean weight = 1 for non-ignore classes
    weights = weights / weights[1:].mean()
    
    print(f"Class weights computed: {weights.tolist()}")
    print(f"Pixel distribution: {pixel_counts.tolist()}")
    
    return weights


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice loss that ignores specified index."""
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    probs = logits.softmax(1)
    one_hot = F.one_hot(safe_target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    valid4 = valid[:, None]
    probs, one_hot = probs * valid4, one_hot * valid4
    intersection = (probs * one_hot).sum((0, 2, 3))
    denominator = (probs + one_hot).sum((0, 2, 3))
    # Skip ignore class (index 0)
    dice = (2 * intersection[1:] + 1.0) / (denominator[1:] + 1.0)
    return 1.0 - dice.mean()


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute confusion matrix excluding ignore index."""
    keep = target != IGNORE_INDEX
    ids = target[keep] * NUM_CLASSES + pred[keep]
    return torch.bincount(ids, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def mean_iou(cm: torch.Tensor) -> tuple[float, list[float]]:
    """Calculate mean IoU from confusion matrix (excluding ignore class)."""
    cm = cm.double()
    intersection = cm.diag()
    union = cm.sum(0) + cm.sum(1) - intersection
    ious = intersection[1:] / union[1:].clamp_min(1)
    return ious.mean().item(), ious.tolist()


def read_checkpoint(path: Path):
    """Read CPU tensors and cross-platform path metadata without model construction."""
    with torch.serialization.safe_globals([
        PurePosixPath, PureWindowsPath,
        (PurePosixPath, "pathlib.PosixPath"),
        (PureWindowsPath, "pathlib.WindowsPath"),
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_checkpoint(path: Path, device: torch.device, backbone: str | None = None):
    """Load the backbone recorded by training; reject an explicit mismatch."""
    checkpoint = read_checkpoint(path)
    saved_backbone = checkpoint.get("args", {}).get("backbone", "resnet101")
    if backbone is not None and backbone != saved_backbone:
        raise ValueError(f"Backbone mismatch: requested {backbone}, checkpoint uses {saved_backbone}")
    model = build_model(pretrained=False, backbone=saved_backbone)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def image_tensor(path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and normalize image to tensor."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
    x = torch.from_numpy(array)
    # ImageNet normalization
    x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225]
    )[:, None, None]
    return x.to(device), (image.height, image.width)


@torch.inference_mode()
def sliding_logits(model, image: torch.Tensor, tile: int, overlap: int, 
                   tta: bool = False, batch_size: int = 8, scales: list[float] | None = None) -> torch.Tensor:
    """Sliding window inference with optional TTA and multi-scale.
    
    Args:
        model: Segmentation model
        image: Input tensor (C, H, W)
        tile: Window size
        overlap: Overlap between windows
        tta: Enable test-time augmentation (horizontal flip)
        batch_size: Batch size for processing patches
        scales: List of scales for multi-scale inference (e.g., [0.75, 1.0, 1.25])
    
    Returns:
        Averaged logits tensor (NUM_CLASSES, H, W)
    """
    if image.ndim != 3 or min(image.shape) < 1 or not image.is_floating_point():
        raise ValueError("image must be a nonempty floating-point tensor (C, H, W)")
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise ValueError("tile must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < tile:
        raise ValueError("overlap must be an integer in [0, tile)")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    scales = [1.0] if scales is None else list(scales)
    if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("scales must contain finite positive numbers")

    _, height, width = image.shape
    stride = tile - overlap
    scores = torch.zeros(NUM_CLASSES, height, width, device=image.device, dtype=torch.float32)

    def predict(batch):
        logits = model(batch)["out"].float()
        if logits.ndim != 4 or logits.shape[:2] != (batch.shape[0], NUM_CLASSES):
            raise ValueError("model output must have shape (batch, NUM_CLASSES, H, W)")
        if logits.shape[-2:] != (tile, tile):
            logits = F.interpolate(logits, size=(tile, tile), mode="bilinear", align_corners=False)
        return logits

    for scale in scales:
        new_h, new_w = max(1, int(height * scale)), max(1, int(width * scale))
        if (new_h, new_w) != (height, width):
            scaled_image = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        else:
            scaled_image = image

        # Stitch in this scale's coordinates before resizing the complete prediction.
        last_y, last_x = max(new_h - tile, 0), max(new_w - tile, 0)
        ys = list(range(0, last_y + 1, stride))
        xs = list(range(0, last_x + 1, stride))
        if ys[-1] != last_y:
            ys.append(last_y)
        if xs[-1] != last_x:
            xs.append(last_x)
        coords = [(y, x) for y in ys for x in xs]
        scale_scores = torch.zeros(NUM_CLASSES, new_h, new_w, device=image.device, dtype=torch.float32)
        counts = torch.zeros(1, new_h, new_w, device=image.device, dtype=torch.float32)

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for y, x in batch_coords:
                patch = scaled_image[:, y:y + tile, x:x + tile]
                # F.pad preserves the input channel count, dtype, and device.
                patches.append(F.pad(patch, (0, tile - patch.shape[2], 0, tile - patch.shape[1])))
            batch = torch.stack(patches)
            logits_batch = predict(batch)
            if tta:
                flipped_logits = predict(torch.flip(batch, (-1,)))
                logits_batch = (logits_batch + torch.flip(flipped_logits, (-1,))) * 0.5

            for j, (y, x) in enumerate(batch_coords):
                end_y, end_x = min(y + tile, new_h), min(x + tile, new_w)
                scale_scores[:, y:end_y, x:end_x] += logits_batch[j, :, :end_y - y, :end_x - x]
                counts[:, y:end_y, x:end_x] += 1

        scale_scores /= counts
        if (new_h, new_w) != (height, width):
            scale_scores = F.interpolate(
                scale_scores.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(0)
        scores += scale_scores

    return scores / len(scales)
