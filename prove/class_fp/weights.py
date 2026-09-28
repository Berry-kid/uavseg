"""Train-split-only pixel counts and controlled class-weighting alternatives."""
from __future__ import annotations
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import torch

MODES = ("inverse", "inverse_sqrt", "none")


def names_digest(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def weights_from_counts(counts, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown class-weight mode: {mode}")
    array = np.asarray(counts)
    if (array.shape != (9,) or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all() or (array < 0).any()
            or not np.equal(array, np.floor(array)).all()):
        raise ValueError("Expected nine nonnegative integer pixel counts")
    if array[1:].sum() <= 0:
        raise ValueError("No supervised pixels in training split")
    if mode == "none":
        return None
    # Match the original inverse-frequency arithmetic for IDs 1..8 exactly.
    values = torch.tensor(array, dtype=torch.float64)
    values[0] = 0
    inverse = 1.0 / values.clamp_min(1)
    weights = inverse if mode == "inverse" else torch.sqrt(inverse)
    weights = weights / weights[1:].mean()
    weights[0] = 0  # Targets with ID 0 are ignored by CE/Dice.
    return weights.float()


def _count_mask(path):
    with Image.open(path) as mask:
        if mask.mode != "L":
            raise ValueError(f"Expected single-channel class-ID mask: {path}")
        array = np.asarray(mask)
        if array.dtype != np.uint8 or array.ndim != 2 or array.size == 0 or array.max() > 8:
            raise ValueError(f"Mask IDs must be integers 0..8: {path}")
        return np.bincount(array.reshape(-1), minlength=9).astype(np.int64)


def count_train_pixels(data_root, train_names, workers=4, progress=None):
    names = list(train_names)
    if (not names or len(set(names)) != len(names)
            or any(Path(n).name != n or "/" in n or "\\" in n for n in names)):
        raise ValueError("Training filenames must be unique basenames")
    mask_dir = Path(data_root) / "masks"
    paths = [mask_dir / n for n in names]
    counts = np.zeros(9, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
        for index, row in enumerate(pool.map(_count_mask, paths), 1):
            counts += row
            if progress and (index == 1 or index % 500 == 0 or index == len(paths)):
                progress(index, len(paths))
    if counts[1:].sum() <= 0:
        raise ValueError("No supervised pixels in training split")
    return counts.tolist()


def weight_report(counts, names, mode):
    weights = weights_from_counts(counts, mode)
    return {
        "schema_version": 1, "class_weighting": mode,
        "formula": {"inverse": "1/frequency", "inverse_sqrt": "1/sqrt(frequency)",
                    "none": "unweighted CE"}[mode],
        "normalization": "mean weight over IDs 1..8 equals 1; ID0 target ignored",
        "training_images": len(names), "training_names_sha256": names_digest(names),
        "pixel_counts_ids_0_to_8": list(counts),
        "class_weights_ids_0_to_8": None if weights is None else weights.tolist(),
        "missing_training_classes": [i for i in range(1, 9) if counts[i] == 0],
        "count_source": "only masks named in the saved training split",
        "loss": "weighted main CE + 0.5 Dice + 0.4 weighted auxiliary CE",
        "cross_experiment_loss_warning": "Weighted CE losses from different weighting schemes are not directly comparable.",
    }

