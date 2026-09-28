"""Strict competition submission validator; never computes metrics without labels."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zipfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_png(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a real PNG: invalid signature")
    offset = 8
    chunks = []
    header = None
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("PNG chunk exceeds file length")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        chunks.append(kind)
        if kind == b"IHDR":
            if len(chunks) != 1 or length != 13:
                raise ValueError("IHDR must occur once as the first chunk")
            header = struct.unpack(">IIBBBBB", payload)
        if kind == b"PLTE":
            raise ValueError("PLTE palette chunk is forbidden")
        offset = end
        if kind == b"IEND":
            if length != 0 or offset != len(data):
                raise ValueError("invalid IEND or trailing bytes")
            break
    if header is None or not chunks or chunks[-1] != b"IEND" or b"IDAT" not in chunks:
        raise ValueError("missing required PNG chunks")
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    if bit_depth != 8 or color_type != 0:
        raise ValueError(f"expected 8-bit grayscale color type 0; got depth={bit_depth}, type={color_type}")
    if compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ValueError("invalid PNG header methods")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError(f"expected PNG mode L; got {image.format}/{image.mode}")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("animated/multiframe PNG is forbidden")
        image.load()
        if image.size != (width, height):
            raise ValueError("PIL dimensions differ from IHDR")
        pixels = np.asarray(image)
        if pixels.dtype != np.uint8 or pixels.ndim != 2:
            raise ValueError(f"expected 2D uint8; got {pixels.dtype}, shape={pixels.shape}")
        if (width, height) != (1024, 1024):
            raise ValueError(f"expected 1024x1024; got {width}x{height}")
        if int(pixels.max()) > 8:
            raise ValueError(f"invalid label IDs: {np.unique(pixels[pixels > 8]).tolist()}")
        return pixels.copy(), (width, height)


def validate(images_dir: Path, masks_dir: Path, zip_path: Path) -> dict:
    images_dir, masks_dir, zip_path = map(lambda p: Path(p).resolve(), (images_dir, masks_dir, zip_path))
    errors: list[str] = []
    report = {
        "schema_version": 1,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "images": str(images_dir), "masks": str(masks_dir), "zip": str(zip_path),
        "passed": False, "expected_images": 0, "mask_files": 0, "zip_entries": 0,
        "valid_masks": 0, "byte_identical_zip_masks": 0,
        "required_size": [1024, 1024], "mask_dimensions": [],
        "pixel_id_counts": {str(i): 0 for i in range(9)},
        "total_pixels": 0, "zip_sha256": None,
        "zip_crc_verified": False,
        "pixel_statistics_scope": "successfully validated masks; complete only when passed=true",
        "metric_note": "No ground-truth masks supplied; mIoU is not computed.",
    }
    sizes = set()
    counts = np.zeros(9, dtype=np.int64)
    try:
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise ValueError("images and masks must both be existing directories")
        images = {p.name: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
        if not images:
            raise ValueError("no source images found")
        report["expected_images"] = len(images)
        expected = set(images)
        if any(not n.endswith(".png") for n in expected):
            errors.append("source filenames must end with .png to preserve exact names and PNG output")
        mask_entries = list(masks_dir.iterdir())
        mask_files = {p.name: p for p in mask_entries if p.is_file()}
        report["mask_files"] = len(mask_files)
        if any(not p.is_file() for p in mask_entries):
            errors.append("masks directory contains nested directories or non-files")
        if set(mask_files) != expected:
            errors.append(f"mask names mismatch; missing={sorted(expected-set(mask_files))[:10]}, extra={sorted(set(mask_files)-expected)[:10]}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            report["zip_entries"] = len(infos)
            duplicates = sorted(n for n, count in Counter(names).items() if count > 1)
            if duplicates:
                errors.append(f"duplicate ZIP entry names: {duplicates[:10]}")
            if any(info.is_dir() or "/" in info.filename or "\\" in info.filename or ":" in info.filename or info.filename in (".", "..") for info in infos):
                errors.append("ZIP must contain only mask files at its root, without paths or directory entries")
            if set(names) != expected:
                errors.append(f"ZIP names mismatch; missing={sorted(expected-set(names))[:10]}, extra={sorted(set(names)-expected)[:10]}")
            info_by_name = {info.filename: info for info in infos}
            crc_passed = True
            for info in infos:
                try:
                    archive.read(info)
                except Exception as exc:
                    crc_passed = False
                    errors.append(f"ZIP read/CRC failure for {info.filename}: {type(exc).__name__}: {exc}")
            report["zip_crc_verified"] = crc_passed
            for name in sorted(expected):
                try:
                    if name not in mask_files or name not in info_by_name:
                        continue
                    with Image.open(images[name]) as original:
                        original_size = original.size
                        original.verify()
                    data = mask_files[name].read_bytes()
                    pixels, size = strict_png(data)
                    sizes.add(size)
                    if size != original_size:
                        raise ValueError(f"mask size {size} differs from source size {original_size}")
                    counts += np.bincount(pixels.reshape(-1), minlength=9)
                    report["valid_masks"] += 1
                    zipped = archive.read(info_by_name[name])
                    if zipped != data:
                        raise ValueError("ZIP contents differ from mask file bytes")
                    report["byte_identical_zip_masks"] += 1
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
        report["zip_sha256"] = sha256(zip_path)
    except Exception as exc:
        errors.append(f"validation failure: {type(exc).__name__}: {exc}")
    report["mask_dimensions"] = [list(size) for size in sorted(sizes)]
    report["pixel_id_counts"] = {str(i): int(counts[i]) for i in range(9)}
    report["total_pixels"] = int(counts.sum())
    report["error_count"] = len(errors)
    report["errors"] = errors[:50]
    report["passed"] = not errors and report["valid_masks"] == report["expected_images"] > 0 and report["byte_identical_zip_masks"] == report["expected_images"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.images, args.masks, args.zip)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())