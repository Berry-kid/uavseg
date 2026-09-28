"""One inference contract shared by final validation and test prediction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch

from uavseg.core import image_tensor, sliding_logits


@dataclass(frozen=True)
class InferenceConfig:
    scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    tta: bool = True
    tile: int = 768
    overlap: int = 192
    batch_size: int = 2
    amp: bool = True

    def __post_init__(self):
        if not self.scales or any(not math.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("scales must be finite and positive")
        if self.tile <= 0 or not 0 <= self.overlap < self.tile or self.batch_size <= 0:
            raise ValueError("Invalid tile, overlap or inference batch size")
        if not isinstance(self.amp, bool) or not isinstance(self.tta, bool):
            raise ValueError("amp and tta must be booleans")

    def to_dict(self):
        values = asdict(self)
        values["scales"] = list(self.scales)
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["scales"] = tuple(values["scales"])
        return cls(**values)


def resolve_config(checkpoint, overrides=None):
    """A selected checkpoint must retain the exact profile used for selection."""
    saved = checkpoint.get("inference_config")
    config = InferenceConfig.from_dict(saved) if saved else InferenceConfig()
    changes = {k: v for k, v in (overrides or {}).items() if v is not None}
    requested = config.to_dict()
    requested.update(changes)
    requested = InferenceConfig.from_dict(requested)
    if saved and requested != config:
        raise ValueError(
            "Inference overrides differ from this checkpoint's final validation settings. "
            "Remove the overrides, or reselect candidates with the desired configuration."
        )
    return requested


def prepare_model(model, device, config):
    device = torch.device(device)
    if config.amp and device.type != "cuda":
        raise ValueError("This inference profile requires CUDA float16 AMP; CPU would change precision")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model.to(device).eval()
    # The auxiliary head is trained, but is not used for final predictions.
    if hasattr(model, "aux_classifier"):
        model.aux_classifier = None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        model.to(memory_format=torch.channels_last)
    return model


@torch.inference_mode()
def predict_logits(model, image, config):
    with torch.autocast(image.device.type, dtype=torch.float16,
                        enabled=config.amp and image.device.type == "cuda"):
        logits = sliding_logits(
            model, image, tile=config.tile, overlap=config.overlap,
            tta=config.tta, batch_size=config.batch_size, scales=list(config.scales)
        )
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite prediction")
    return logits


@torch.inference_mode()
def predict_mask(model, path: Path, device, config):
    image, shape = image_tensor(path, torch.device(device))
    prediction = predict_logits(model, image, config).argmax(0)
    if tuple(prediction.shape) != shape:
        raise ValueError(f"Prediction size changed: {path}")
    return prediction

