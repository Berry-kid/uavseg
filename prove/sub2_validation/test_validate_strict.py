"""Synthetic checks only; fixtures do not contain competition data."""
import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from validate_strict import validate

HERE = Path(__file__).resolve().parent
FIXTURES = Path(tempfile.mkdtemp(prefix="synthetic_", dir=HERE))


class StrictValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = FIXTURES / self._testMethodName
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.images.mkdir(parents=True)
        self.masks.mkdir()
        self.mask = self.masks / "sample.png"
        self.archive = self.root / "submission.zip"
        Image.new("RGB", (1024, 1024), (20, 40, 60)).save(self.images / "sample.png")
        self.pixels = np.tile(np.arange(1024, dtype=np.uint16) % 9, (1024, 1)).astype(np.uint8)
        Image.fromarray(self.pixels).save(self.mask)
        self.pack()

    def pack(self, name="sample.png", data=None, duplicate=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as archive:
                content = self.mask.read_bytes() if data is None else data
                archive.writestr(name, content)
                if duplicate:
                    archive.writestr(name, content)

    def check_invalid(self, text):
        result = validate(self.images, self.masks, self.archive)
        self.assertFalse(result["passed"], result)
        self.assertIn(text, "\n".join(result["errors"]))

    def test_valid_all_ids_and_cli(self):
        result = validate(self.images, self.masks, self.archive)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["zip_crc_verified"])
        self.assertEqual(result["total_pixels"], 1024 * 1024)
        self.assertEqual(result["pixel_id_counts"], {str(i): int((self.pixels == i).sum()) for i in range(9)})
        self.assertEqual(result["byte_identical_zip_masks"], 1)
        self.assertEqual(len(result["zip_sha256"]), 64)
        report_path = self.root / "cli_report.json"
        proc = subprocess.run([sys.executable, "-B", str(HERE / "validate_strict.py"), "--images", str(self.images), "--masks", str(self.masks), "--zip", str(self.archive), "--report", str(report_path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(report_path.read_text())["passed"])

    def test_rgb(self):
        Image.new("RGB", (1024, 1024)).save(self.mask)
        self.pack()
        self.check_invalid("expected 8-bit grayscale")

    def test_palette(self):
        Image.new("P", (1024, 1024)).save(self.mask)
        self.pack()
        self.check_invalid("palette chunk is forbidden")

    def test_grayscale_16bit(self):
        Image.fromarray(np.zeros((1024, 1024), dtype=np.uint16)).save(self.mask)
        self.pack()
        self.check_invalid("depth=16")

    def test_plte_in_grayscale(self):
        data = self.mask.read_bytes()
        payload = b"\x00\x00\x00"
        chunk = struct.pack(">I", len(payload)) + b"PLTE" + payload + struct.pack(">I", zlib.crc32(b"PLTE" + payload) & 0xFFFFFFFF)
        self.mask.write_bytes(data[:33] + chunk + data[33:])
        self.pack()
        self.check_invalid("palette chunk is forbidden")

    def test_fake_png_extension(self):
        Image.new("L", (1024, 1024)).save(self.mask, format="JPEG")
        self.pack()
        self.check_invalid("invalid signature")

    def test_wrong_mask_size(self):
        Image.new("L", (512, 512)).save(self.mask)
        self.pack()
        self.check_invalid("expected 1024x1024")

    def test_source_size_mismatch(self):
        Image.new("RGB", (512, 512)).save(self.images / "sample.png")
        self.check_invalid("differs from source size")

    def test_invalid_id(self):
        Image.new("L", (1024, 1024), 9).save(self.mask)
        self.pack()
        self.check_invalid("invalid label IDs: [9]")

    def test_mask_name_mismatch(self):
        self.mask.rename(self.masks / "different.png")
        self.check_invalid("mask names mismatch")

    def test_zip_duplicate(self):
        self.pack(duplicate=True)
        self.check_invalid("duplicate ZIP entry names")

    def test_zip_nested(self):
        self.pack(name="mask/sample.png")
        self.check_invalid("only mask files at its root")

    def test_zip_byte_mismatch(self):
        other = io.BytesIO()
        Image.new("L", (1024, 1024), 8).save(other, format="PNG")
        self.pack(data=other.getvalue())
        self.check_invalid("differ from mask file bytes")

    def test_zip_crc_failure(self):
        data = bytearray(self.archive.read_bytes())
        name_len, extra_len = struct.unpack_from("<HH", data, 26)
        data[30 + name_len + extra_len + 40] ^= 1
        self.archive.write_bytes(data)
        self.check_invalid("ZIP read/CRC failure")

    def test_png_crc_failure(self):
        data = bytearray(self.mask.read_bytes())
        data[-1] ^= 1
        self.mask.write_bytes(data)
        self.pack()
        self.check_invalid("PNG chunk CRC mismatch")


if __name__ == "__main__":
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StrictValidationTests))
    (FIXTURES / "SYNTHETIC_TEST_DATA.txt").write_text("Synthetic format-validation tests only. No competition data or real prediction results.\n", encoding="utf-8")
    report = {"synthetic": True, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(), "fixtures": str(FIXTURES)}
    (HERE / "test_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (HERE / "tests.log").write_text(output.getvalue(), encoding="utf-8")
    print(output.getvalue())
    print(json.dumps(report, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)