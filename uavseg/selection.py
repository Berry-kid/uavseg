"""Retain ordinary-val candidates and select one using complete B-TTA validation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from uavseg.core import confusion_matrix, load_checkpoint, mean_iou, read_checkpoint
from uavseg.inference import InferenceConfig, predict_mask, prepare_model

LABELS = ["Background", "Building", "Road", "Water", "Barren",
          "Vegetation", "Agricultural", "Vehicle"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def val_protocol(split_path, crop_size):
    return {"split_sha256": file_hash(split_path), "crop_size": int(crop_size),
            "tta": False, "precision": "float32", "metric": "dataset_miou_ids_1_to_8_ignore_gt_0"}


class CandidatePool:
    """Owns only candidates/epoch_NNNN.pt files listed in its manifest."""

    def __init__(self, run_dir, keep, protocol, resume_epoch=None):
        if keep < 1:
            raise ValueError("keep must be positive")
        self.directory = Path(run_dir).resolve() / "candidates"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = self.directory / "index.json"
        self.keep, self.protocol = keep, protocol
        entries = []
        if self.manifest.exists():
            saved = json.loads(self.manifest.read_text(encoding="utf-8"))
            if saved["val_protocol"] != protocol:
                raise ValueError("Candidate validation split/crop/protocol changed; use a new run directory")
            entries = saved["candidates"]
            for entry in entries:
                self.path(entry)
                if not math.isfinite(entry["val_miou"]):
                    raise ValueError("Invalid candidate score")
        retained = [e for e in entries if resume_epoch is None or e["epoch"] <= resume_epoch]
        retained = sorted(retained, key=self.rank)[:keep]
        self.entries = retained
        self._commit(entries)

    @staticmethod
    def rank(entry):
        return (-entry["val_miou"], entry["epoch"])

    def path(self, entry):
        expected = f"epoch_{int(entry['epoch']):04d}.pt"
        if entry["file"] != expected:
            raise ValueError("Invalid managed candidate filename")
        path = (self.directory / expected).resolve()
        if path.parent != self.directory or not path.is_file():
            raise ValueError(f"Missing or unsafe candidate: {path}")
        return path

    def _commit(self, previous):
        write_json(self.manifest, {"version": 1, "keep": self.keep,
                   "val_protocol": self.protocol, "candidates": self.entries})
        retained = {e["file"] for e in self.entries}
        for entry in previous:
            if entry["file"] not in retained:
                # Exact managed filename, checked against this absolute directory.
                self.path(entry).unlink()

    def consider(self, state):
        epoch = int(state["epoch"])
        score = float(state.get("val_miou", state.get("miou")))
        if not math.isfinite(score):
            raise ValueError("Cannot rank a nonfinite validation score")
        for entry in self.entries:
            if entry["epoch"] == epoch:
                if entry["val_miou"] != score:
                    raise ValueError("Conflicting ordinary-val scores for the same epoch")
                return False
        entry = {"epoch": epoch, "val_miou": score, "file": f"epoch_{epoch:04d}.pt"}
        ranked = sorted(self.entries + [entry], key=self.rank)[:self.keep]
        if entry not in ranked:
            return False
        compact = {
            "model": {k: v.detach().cpu() for k, v in state["model"].items()},
            "epoch": epoch, "val_miou": score, "miou": score,
            "val_per_class_iou": state.get("val_per_class_iou"),
            "args": dict(state["args"]), "val_protocol": self.protocol,
            "checkpoint_kind": "candidate",
        }
        save_checkpoint(self.directory / entry["file"], compact)
        previous, self.entries = self.entries, ranked
        self._commit(previous)
        return True


def _validation_inputs(data_root, split_path):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train, names = split["train"], split["val"]
    if not names or len(names) != len(set(names)) or len(train) != len(set(train)) or set(train) & set(names):
        raise ValueError("Invalid or overlapping saved train/val split")
    digest = hashlib.sha256()
    for name in names:
        if Path(name).name != name:
            raise ValueError("Validation names must be basenames")
        for directory in ("images", "masks"):
            path = data_root / directory / name
            stat = path.stat()
            digest.update(f"{directory}/{name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return names, digest.hexdigest()


def _evaluate(path, data_root, names, config, device):
    model, checkpoint = load_checkpoint(path, device)
    model = prepare_model(model, device, config)
    del checkpoint
    cm = torch.zeros(9, 9, dtype=torch.int64, device=device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, name in enumerate(names, 1):
                pred = predict_mask(model, data_root / "images" / name, device, config)
                with Image.open(data_root / "masks" / name) as image:
                    if image.mode != "L":
                        raise ValueError(f"Expected class-ID grayscale mask: {name}")
                    target = np.asarray(image, dtype=np.uint8).copy()
                if tuple(target.shape) != tuple(pred.shape) or target.max() > 8:
                    raise ValueError(f"Invalid label size or IDs: {name}")
                cm += confusion_matrix(pred, torch.from_numpy(target.astype(np.int64)).to(device))
                if index == 1 or index % 50 == 0 or index == len(names):
                    print(f"TTA {path.name}: {index}/{len(names)}", flush=True)
        score, per_class = mean_iou(cm)
        return {"complete": True, "evaluated_images": len(names), "tta_miou": score,
                "tta_per_class_iou": per_class, "class_names": LABELS,
                "confusion_matrix": cm.cpu().tolist(),
                "elapsed_seconds": time.perf_counter() - started}
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def select_candidates(run_dir, data_root, config=None, device="cuda", threads=4):
    """Evaluate full validation images, then export best_tta.pt with its exact profile."""
    run_dir, data_root = Path(run_dir).resolve(), Path(data_root).resolve()
    config = config or InferenceConfig()
    device = torch.device(device)
    if config.amp and device.type != "cuda":
        raise ValueError("Final B profile uses CUDA AMP; choose a matching device")
    torch.set_num_threads(threads)
    index_path = run_dir / "candidates/index.json"
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    split_path = run_dir / "split.json"
    if manifest["val_protocol"]["split_sha256"] != file_hash(split_path):
        raise ValueError("Saved validation split changed")
    entries = manifest["candidates"]
    if not entries:
        raise ValueError("No candidates are available")
    names, data_fingerprint = _validation_inputs(data_root, split_path)
    folder = run_dir / "tta_eval"
    folder.mkdir(exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name("inference.py"),
               Path(__file__).with_name("core.py")]
    recipe = {"inference_config": config.to_dict(), "split_sha256": file_hash(split_path),
              "data_fingerprint": data_fingerprint, "device": device.type,
              "torch": str(torch.__version__),
              "source_hashes": {p.name: file_hash(p) for p in sources}}
    write_json(folder / "evaluation_config.json", recipe)
    rows = []
    for entry in entries:
        expected = f"epoch_{int(entry['epoch']):04d}.pt"
        if entry["file"] != expected:
            raise ValueError("Invalid candidate filename")
        candidate_dir = (run_dir / "candidates").resolve()
        path = (candidate_dir / expected).resolve()
        if path.parent != candidate_dir:
            raise ValueError("Candidate path escaped its directory")
        digest = file_hash(path)
        cache_key = dict(recipe, checkpoint_sha256=digest)
        report_path = folder / f"{path.stem}.json"
        result = None
        if report_path.exists():
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if (cached.get("cache_key") == cache_key and cached.get("complete") is True
                    and cached.get("evaluated_images") == len(names)):
                result = cached
                print(f"Reuse complete TTA result: {path.name}", flush=True)
        if result is None:
            result = _evaluate(path, data_root, names, config, device)
            if file_hash(path) != digest:
                raise ValueError("Candidate changed during evaluation")
            result.update({"epoch": entry["epoch"], "val_miou": entry["val_miou"],
                           "checkpoint": expected, "cache_key": cache_key})
            write_json(report_path, result)
        rows.append(result)
    # A moved or modified data source must not silently change the selected model.
    if _validation_inputs(data_root, split_path)[1] != data_fingerprint:
        raise ValueError("Validation inputs changed during selection")
    winner = max(rows, key=lambda row: (row["tta_miou"], row["val_miou"], -row["epoch"]))
    winner_path = run_dir / "candidates" / winner["checkpoint"]
    if file_hash(winner_path) != winner["cache_key"]["checkpoint_sha256"]:
        raise ValueError("Winning checkpoint changed")
    selected = read_checkpoint(winner_path)
    selected.update({
        "checkpoint_kind": "final_tta", "selected_by": "tta_miou",
        "tta_miou": winner["tta_miou"], "tta_per_class_iou": winner["tta_per_class_iou"],
        "inference_config": config.to_dict(),
        "tta_selection": {"split_sha256": recipe["split_sha256"], "validation_images": len(names),
                          "source_checkpoint_sha256": winner["cache_key"]["checkpoint_sha256"],
                          "candidate_epochs": [row["epoch"] for row in rows]},
    })
    # miou/val_miou keep their ordinary validation meaning; TTA score has its own key.
    save_checkpoint(run_dir / "best_tta.pt", selected)
    report = {"complete": True, "selected_epoch": winner["epoch"],
              "selected_val_miou": winner["val_miou"], "selected_tta_miou": winner["tta_miou"],
              "best_tta_checkpoint": str(run_dir / "best_tta.pt"),
              "inference_config": config.to_dict(), "validation_images": len(names),
              "candidate_results": rows}
    write_json(run_dir / "tta_selection.json", report)
    write_json(run_dir / "final_inference.json", config.to_dict())
    print(f"Selected epoch {winner['epoch']}: ordinary val={winner['val_miou']:.6f}, "
          f"TTA val={winner['tta_miou']:.6f} -> {run_dir / 'best_tta.pt'}", flush=True)
    return report

