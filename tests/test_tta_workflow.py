from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

import predict
import train
from uavseg.core import read_checkpoint
from uavseg.inference import InferenceConfig, resolve_config
from uavseg.selection import CandidatePool, file_hash, select_candidates, val_protocol


class ConstantModel(torch.nn.Module):
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id

    def forward(self, x):
        logits = torch.zeros((x.shape[0], 9, *x.shape[-2:]), device=x.device)
        logits[:, self.class_id] = 10
        return {"out": logits}


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 9, 1)

    def forward(self, x):
        output = self.head(x)
        return {"out": output, "aux": output}


def constant_loader(path, device, backbone=None):
    checkpoint = read_checkpoint(path)
    return ConstantModel(int(checkpoint["model"]["class_id"])).to(device), checkpoint


def tiny_loader(path, device, backbone=None):
    checkpoint = read_checkpoint(path)
    model = TinyTrainModel()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture_parent = Path(__file__).resolve().parent
        self.root = (self.fixture_parent / ("workflow_" + uuid.uuid4().hex)).resolve()
        self.assertEqual(self.root.parent, self.fixture_parent)
        self.root.mkdir()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.data = self.root / "data"
        for directory in ("images", "masks"):
            (self.data / directory).mkdir(parents=True)
        for name in ("train.png", "val.png"):
            Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(self.data / "images" / name)
            Image.fromarray(np.ones((8, 8), np.uint8)).save(self.data / "masks" / name)
        self.split = self.run_dir / "split.json"
        self.split.write_text(json.dumps({"train": ["train.png"], "val": ["val.png"]}), encoding="utf-8")
        self.protocol = val_protocol(self.split, 8)
        self.config = InferenceConfig(tile=4, overlap=1, batch_size=2, amp=False)

    def tearDown(self):
        # Resolve and verify this exact test workspace before recursive cleanup.
        self.assertEqual(self.root.resolve().parent, self.fixture_parent)
        self.assertFalse(self.root.is_symlink())
        shutil.rmtree(self.root)

    def state(self, epoch, score, class_id=1):
        return {"model": {"class_id": torch.tensor(class_id)}, "epoch": epoch,
                "val_miou": score, "miou": score, "optimizer": {"must_not_be_exported": True},
                "args": {"backbone": "resnet101", "crop_size": 8, "data_root": self.data}}

    def test_topk_retains_non_record_epochs_and_prunes_only_owned_files(self):
        pool = CandidatePool(self.run_dir, 3, self.protocol)
        unrelated = pool.directory / "user_note.txt"
        unrelated.write_text("keep", encoding="utf-8")
        for epoch, score in enumerate([0.7, 0.9, 0.85, 0.8, 0.6], 1):
            pool.consider(self.state(epoch, score))
        self.assertEqual([x["epoch"] for x in pool.entries], [2, 3, 4])
        self.assertEqual(len(list(pool.directory.glob("epoch_*.pt"))), 3)
        self.assertTrue(unrelated.exists())
        saved = read_checkpoint(pool.path(pool.entries[0]))
        self.assertNotIn("optimizer", saved)
        self.assertEqual(saved["val_miou"], 0.9)

    def test_resume_rewinds_future_candidates_and_checks_validation_protocol(self):
        pool = CandidatePool(self.run_dir, 2, self.protocol)
        for epoch, score in enumerate([0.5, 0.8, 0.7, 0.9], 1):
            pool.consider(self.state(epoch, score))
        resumed = CandidatePool(self.run_dir, 2, self.protocol, resume_epoch=3)
        self.assertEqual([x["epoch"] for x in resumed.entries], [2])
        self.assertFalse((pool.directory / "epoch_0004.pt").exists())
        resumed.consider(self.state(3, 0.7))
        self.assertEqual([x["epoch"] for x in resumed.entries], [2, 3])
        with self.assertRaises(ValueError):
            CandidatePool(self.run_dir, 2, dict(self.protocol, crop_size=16))
        with self.assertRaises(ValueError):
            resumed.consider(self.state(5, float("nan")))

    def test_tta_can_select_a_different_epoch_without_overwriting_regular_best(self):
        pool = CandidatePool(self.run_dir, 3, self.protocol)
        pool.consider(self.state(1, 0.9, class_id=2))
        pool.consider(self.state(2, 0.8, class_id=1))
        ordinary_best = self.run_dir / "best.pt"
        torch.save({"marker": "ordinary-best"}, ordinary_best)
        previous_hash = file_hash(ordinary_best)
        with patch("uavseg.selection.load_checkpoint", side_effect=constant_loader):
            result = select_candidates(self.run_dir, self.data, self.config, "cpu")
        self.assertEqual(result["selected_epoch"], 2)
        self.assertAlmostEqual(result["selected_tta_miou"], 1 / 8)
        self.assertEqual(file_hash(ordinary_best), previous_hash)
        selected = read_checkpoint(self.run_dir / "best_tta.pt")
        self.assertEqual(selected["val_miou"], 0.8)
        self.assertEqual(selected["miou"], 0.8)
        self.assertAlmostEqual(selected["tta_miou"], 1 / 8)
        self.assertEqual(resolve_config(selected), self.config)
        with patch("uavseg.selection._evaluate", side_effect=AssertionError("Must reuse complete cache")):
            cached = select_candidates(self.run_dir, self.data, self.config, "cpu")
        self.assertEqual(cached["selected_epoch"], 2)

    def test_cache_is_invalidated_when_inference_settings_change(self):
        pool = CandidatePool(self.run_dir, 1, self.protocol)
        pool.consider(self.state(1, 0.8))
        with patch("uavseg.selection.load_checkpoint", side_effect=constant_loader):
            select_candidates(self.run_dir, self.data, self.config, "cpu")
        changed = InferenceConfig(scales=(1.0,), tile=4, overlap=1, amp=False)
        with patch("uavseg.selection._evaluate", side_effect=RuntimeError("reevaluate")):
            with self.assertRaisesRegex(RuntimeError, "reevaluate"):
                select_candidates(self.run_dir, self.data, changed, "cpu")

    def test_selected_profile_is_used_by_prediction_and_cannot_silently_change(self):
        pool = CandidatePool(self.run_dir, 1, self.protocol)
        pool.consider(self.state(1, 0.8))
        with patch("uavseg.selection.load_checkpoint", side_effect=constant_loader):
            select_candidates(self.run_dir, self.data, self.config, "cpu")
        checkpoint = read_checkpoint(self.run_dir / "best_tta.pt")
        with self.assertRaises(ValueError):
            resolve_config(checkpoint, {"tta": False})
        with self.assertRaises(ValueError):
            resolve_config(checkpoint, {"amp": True})
        with self.assertRaises(ValueError):
            resolve_config(checkpoint, {"scales": [1.0]})
        output = self.root / "prediction/masks"
        with patch("predict.load_checkpoint", side_effect=constant_loader):
            predict.main(["--images", str(self.data / "images"),
                          "--checkpoint", str(self.run_dir / "best_tta.pt"),
                          "--output", str(output), "--device", "cpu"])
        metadata = json.loads((output.parent / "inference.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["inference_config"], self.config.to_dict())
        self.assertEqual(metadata["val_miou"], 0.8)
        self.assertAlmostEqual(metadata["tta_val_miou"], 1 / 8)
        with zipfile.ZipFile(output.parent / "submission.zip") as archive:
            self.assertEqual(sorted(archive.namelist()), ["train.png", "val.png"])
        with Image.open(output / "val.png") as mask:
            self.assertEqual(mask.mode, "L")
            self.assertEqual(mask.size, (8, 8))
            self.assertTrue(np.all(np.asarray(mask) == 1))

    def test_legacy_checkpoint_defaults_to_B(self):
        self.assertEqual(resolve_config({}).to_dict(),
                         {"scales": [0.75, 1.0, 1.25], "tta": True, "tile": 768,
                          "overlap": 192, "batch_size": 2, "amp": True})
        self.assertFalse(resolve_config({}, {"tta": False}).tta)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA training smoke test")
    def test_early_stopping_runs_final_selection_and_resume_retains_candidates(self):
        # lr=0 makes ordinary val constant: epoch 2 triggers patience=1.
        args = ["--data-root", str(self.data), "--output", str(self.run_dir),
                "--epochs", "3", "--batch-size", "1", "--crop-size", "8",
                "--workers", "0", "--no-pretrained", "--backbone", "resnet50",
                "--lr", "0", "--patience", "1", "--keep-best", "3"]
        tiny_config = InferenceConfig(tile=4, overlap=1, batch_size=2, amp=True)
        with patch("train.build_model", side_effect=lambda **kw: TinyTrainModel()), \
             patch("train.InferenceConfig", return_value=tiny_config), \
             patch("train.GradScaler", side_effect=lambda *a: torch.amp.GradScaler("cuda", init_scale=1.0)), \
             patch("uavseg.selection.load_checkpoint", side_effect=tiny_loader):
            train.main(args)
        last = read_checkpoint(self.run_dir / "last.pt")
        self.assertEqual(last["epoch"], 2)
        self.assertEqual(last["patience_counter"], 1)
        self.assertTrue((self.run_dir / "best_tta.pt").exists())
        selected = read_checkpoint(self.run_dir / "best_tta.pt")
        self.assertEqual(selected["inference_config"], tiny_config.to_dict())
        selection = json.loads((self.run_dir / "tta_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(len(selection["candidate_results"]), 2)
        history = json.loads((self.run_dir / "val_history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(history), 2)
        self.assertTrue(all(row["tta"] is False for row in history))
        with patch("train.build_model", side_effect=lambda **kw: TinyTrainModel()):
            train.main(args + ["--resume", str(self.run_dir / "last.pt"), "--skip-final-tta"])
        resumed = read_checkpoint(self.run_dir / "last.pt")
        self.assertEqual(resumed["epoch"], 3)
        index = json.loads((self.run_dir / "candidates/index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index["candidates"]), 3)
        self.assertEqual(index["val_protocol"]["split_sha256"], file_hash(self.split))


if __name__ == "__main__":
    unittest.main(verbosity=2)

