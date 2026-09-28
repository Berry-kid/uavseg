"""Synthetic CPU/GPU integration tests. Scores are NOT competition results."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
import uuid
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prove/training_monitor"))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from prove.class_fp.weights import count_train_pixels, weights_from_counts, weight_report
from prove.class_fp.diagnostics import FalsePositiveMonitor, summarize_confusion
from prove.class_fp import train_fp
import train_monitored as baseline
from uavseg import core
from uavseg.core import read_checkpoint
from uavseg.inference import InferenceConfig

OUTPUT = HERE / ("smoke_" + uuid.uuid4().hex[:10])
OUTPUT.mkdir()
(OUTPUT / "SYNTHETIC_NOT_COMPETITION_SCORES.txt").write_text(
    "Synthetic 10-image, 32x32 test fixtures and a tiny network only.\n"
    "No score here measures the competition data or predicts an online gain.\n", encoding="utf-8")
ORIGINAL_FILES = [ROOT/"train.py", ROOT/"prove/training_monitor/train_monitored.py",
                  ROOT/"prove/training_monitor/monitoring.py"] + sorted((ROOT/"uavseg").glob("*.py"))
ORIGINAL_HASHES = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ORIGINAL_FILES}
EVIDENCE = {}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1),
            torch.nn.BatchNorm2d(8), torch.nn.ReLU(), torch.nn.Dropout2d(.2))
        self.head = torch.nn.Conv2d(8, 9, 1)
        self.aux = torch.nn.Conv2d(8, 9, 1)

    def forward(self, x):
        z = self.body(x)
        return {"out": self.head(z), "aux": self.aux(z)}


def fixture(path, count=10):
    for kind in ("images", "masks"):
        (path / kind).mkdir(parents=True)
    rng = np.random.default_rng(713)
    yy, xx = np.indices((32, 32))
    table = np.array([0] + [1]*14 + [2]*6 + [3]*4 + [4]*2 + [5,6,7,8,1], dtype=np.uint8)
    for i in range(count):
        mask = table[(xx + yy*3 + i) % 32]
        pixels = rng.integers(0, 256, size=(32,32,3), dtype=np.uint8)
        Image.fromarray(pixels).save(path/"images"/f"{i:02d}.png")
        Image.fromarray(mask).save(path/"masks"/f"{i:02d}.png")


def rows(path):
    return [json.loads(line) for line in (path/"monitor/epochs.jsonl").read_text(encoding="utf-8").splitlines()]


class WeightTests(unittest.TestCase):
    def test_01_inverse_matches_old_arithmetic_loss_and_gradient(self):
        counts = [37,100003,23001,8103,1104,787,167003,4321,201]
        old_counts = torch.tensor(counts, dtype=torch.float64)
        old_counts[0] = 0
        old = 1.0 / old_counts.clamp_min(1)
        old = (old / old[1:].mean()).float()
        current = weights_from_counts(counts, "inverse")
        self.assertTrue(torch.equal(old[1:], current[1:]))
        self.assertEqual(float(current[0]), 0)
        generator = torch.Generator().manual_seed(21)
        values = torch.randn(2,9,5,5,generator=generator)
        target = (torch.arange(50).reshape(2,5,5) % 9).long()
        a, b = values.clone().requires_grad_(), values.clone().requires_grad_()
        loss_a = F.cross_entropy(a, target, weight=old, ignore_index=0)
        loss_b = F.cross_entropy(b, target, weight=current, ignore_index=0)
        loss_a.backward(); loss_b.backward()
        self.assertTrue(torch.equal(loss_a,loss_b))
        self.assertTrue(torch.equal(a.grad,b.grad))
        sqrt = weights_from_counts(counts,"inverse_sqrt")
        c = values.clone().requires_grad_()
        F.cross_entropy(c,target,weight=sqrt,ignore_index=0).backward()
        self.assertFalse(torch.equal(a.grad,c.grad))
        EVIDENCE["inverse_ignore_zero"] = {"loss_equal":True,"gradient_equal":True,
                                           "sqrt_gradient_changed":True}

    def test_02_sqrt_none_and_invalid_count_contract(self):
        counts = [123,256,64,16,4,1,100,25,9]
        weights = weights_from_counts(counts,"inverse_sqrt")
        expected = torch.tensor([1/np.sqrt(n) for n in counts[1:]],dtype=torch.float64)
        expected = (expected/expected.mean()).float()
        self.assertTrue(torch.equal(weights[1:],expected))
        self.assertAlmostEqual(float(weights[5]/weights[1]),16)
        self.assertIsNone(weights_from_counts(counts,"none"))
        for bad in ([0]*9, [1]*8, [-1]+[1]*8, [1.5]+[1]*8, [float("nan")]+[1]*8):
            with self.subTest(counts=bad), self.assertRaises(ValueError):
                weights_from_counts(bad,"inverse_sqrt")
        with self.assertRaises(ValueError): weights_from_counts(counts,"unknown")

    def test_03_only_named_training_masks_are_counted(self):
        path = OUTPUT/"counts_only"
        (path/"masks").mkdir(parents=True)
        train = np.arange(9,dtype=np.uint8).reshape(3,3)
        Image.fromarray(train).save(path/"masks/train.png")
        Image.fromarray(np.full((3,3),255,dtype=np.uint8)).save(path/"masks/val.png")
        self.assertEqual(count_train_pixels(path,["train.png"],workers=2),[1]*9)
        self.assertEqual(weight_report([1]*9,["train.png"],"inverse")["training_images"],1)
        for names in (["train.png","train.png"], ["../train.png"], ["a\\train.png"], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                count_train_pixels(path,names)
        with self.assertRaises(ValueError): count_train_pixels(path,["val.png"])

    def test_04_nonstandard_and_invalid_mask_rejected(self):
        path = OUTPUT/"invalid_masks"
        (path/"masks").mkdir(parents=True)
        for mode, array in [("RGB",np.ones((2,2,3),dtype=np.uint8)),
                            ("P",np.ones((2,2),dtype=np.uint8)),
                            ("L",np.full((2,2),9,dtype=np.uint8))]:
            image = Image.fromarray(array)
            if mode=="P": image=image.convert("P")
            name = mode+".png"; image.save(path/"masks"/name)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                count_train_pixels(path,[name])
        Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(path/"masks/zero.png")
        with self.assertRaises(ValueError): count_train_pixels(path,["zero.png"])


@unittest.skipUnless(torch.cuda.is_available(),"CUDA integration tests require CUDA")
class GPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.data = OUTPUT/"synthetic_data"
        fixture(cls.data)
        names = sorted(p.name for p in (cls.data/"images").glob("*.png"))
        train, val = baseline.create_stratified_split(names,cls.data/"images",cls.data/"masks",.2,2026)
        cls.split = OUTPUT/"fixed_split.json"
        cls.split.write_text(json.dumps({"train":train,"val":val}),encoding="utf-8")
        cls.train_names = train
        EVIDENCE["fixture"] = {"train_images":len(train),"val_images":len(val),"size":[32,32],
                              "cuda":torch.cuda.get_device_name(0),"torch":str(torch.__version__)}

    def args(self, output, epochs=3, final=False):
        values = ["--data-root",str(self.data),"--output",str(output),"--epochs",str(epochs),
                  "--batch-size","2","--crop-size","16","--lr","3e-5","--workers","0",
                  "--val-ratio",".2","--seed","2026","--no-pretrained","--backbone","resnet50",
                  "--patience","0","--keep-best","2","--tta-batch-size","2",
                  "--train-eval-images","4","--train-eval-every","1",
                  "--log-every","2","--no-tensorboard"]
        if not final: values.append("--skip-final-tta")
        return values

    def fp_args(self, output, mode="inverse", epochs=3, final=False):
        return self.args(output,epochs,final)+["--split-file",str(self.split),"--class-weighting",mode]

    def run_fp(self, args):
        with patch.object(train_fp,"build_model",side_effect=lambda **kw:TinyModel()):
            train_fp.main(args)

    def test_01_inverse_control_is_bitwise_identical_to_original(self):
        old, new = OUTPUT/"original_control", OUTPUT/"inverse_control"
        with patch.object(baseline,"build_model",side_effect=lambda **kw:TinyModel()):
            baseline.main(self.args(old))
        self.run_fp(self.fp_args(new))
        expected, actual = read_checkpoint(old/"last.pt"), read_checkpoint(new/"last.pt")
        self.assertEqual((old/"split.json").read_bytes(),(new/"split.json").read_bytes())
        for key,value in expected["model"].items():
            self.assertTrue(torch.equal(value,actual["model"][key]),key)
        self.assertEqual(expected["val_miou"],actual["val_miou"])
        self.assertEqual([x["train_loss"] for x in rows(old)],[x["train_loss"] for x in rows(new)])
        self.assertEqual([x["val_miou"] for x in rows(old)],[x["val_miou"] for x in rows(new)])
        self.assertEqual(actual["args"]["class_weighting"],"inverse")
        EVIDENCE["inverse_control"] = {"epochs":3,"all_model_tensors_bitwise_equal":True,
                                     "losses_and_scores_equal":True}

    def test_02_sqrt_updates_differ_and_logs_precision_recall(self):
        path = OUTPUT/"inverse_sqrt"
        self.run_fp(self.fp_args(path,mode="inverse_sqrt"))
        control = read_checkpoint(OUTPUT/"inverse_control/last.pt")
        changed = read_checkpoint(path/"last.pt")
        different=[key for key in control["model"] if not torch.equal(control["model"][key],changed["model"][key])]
        self.assertTrue(different)
        records=rows(path)
        self.assertEqual([r["epoch"] for r in records],[1,2,3])
        report=json.loads((path/"monitor/false_positives/epoch_0003.json").read_text(encoding="utf-8"))
        for item in report["val"]["per_class"]:
            name=item["class_name"]
            self.assertEqual(records[-1]["val_precision_"+name],item["precision"])
            self.assertEqual(records[-1]["val_recall_"+name],item["recall"])
            self.assertEqual(records[-1]["val_fp_pixels_"+name],item["fp"])
        info=json.loads((path/"monitor/run_info.json").read_text(encoding="utf-8"))
        for name in ("train_fp.py","weights.py","diagnostics.py"):
            self.assertTrue(any(Path(p).name==name for p in info["source_sha256"]))
        self.assertEqual(info["class_weighting"],"inverse_sqrt")
        EVIDENCE["sqrt_treatment"]={"updated_tensors_differ":different,"precision_recall_logged":True}

    def test_03_unweighted_mode_executes(self):
        path=OUTPUT/"none"
        self.run_fp(self.fp_args(path,mode="none",epochs=1))
        info=json.loads((path/"class_weights.json").read_text(encoding="utf-8"))
        self.assertIsNone(info["class_weights_ids_0_to_8"])
        self.assertEqual(read_checkpoint(path/"last.pt")["args"]["class_weighting"],"none")

    def test_04_interruption_resume_and_reject_changed_mode_split_counts(self):
        path=OUTPUT/"interrupted"
        original= FalsePositiveMonitor.epoch
        def stop_after_first(instance,*args,**kwargs):
            original(instance,*args,**kwargs)
            if instance.rows[-1]["epoch"]==1:
                raise KeyboardInterrupt("Synthetic interruption after complete epoch")
        with patch.object(FalsePositiveMonitor,"epoch",stop_after_first):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fp(self.fp_args(path,mode="inverse_sqrt"))
        first=read_checkpoint(path/"last.pt")
        self.assertEqual(first["epoch"],1)
        resume=self.fp_args(path,mode="inverse_sqrt")+["--resume",str(path/"last.pt")]
        forged=read_checkpoint(path/"last.pt")
        forged["class_weighting"]["pixel_counts_ids_0_to_8"][1] += 1
        torch.save(forged,path/"changed_weight_metadata.pt")
        changed_checkpoint=resume.copy()
        changed_checkpoint[-1]=str(path/"changed_weight_metadata.pt")
        with self.assertRaisesRegex(ValueError,"Checkpoint class weights"):
            self.run_fp(changed_checkpoint)
        wrong_mode=resume.copy()
        wrong_mode[wrong_mode.index("--class-weighting")+1]="inverse"
        with self.assertRaisesRegex(ValueError,"class_weighting"):
            self.run_fp(wrong_mode)
        split_backup=self.split.read_bytes()
        try:
            value=json.loads(split_backup)
            value["train"][0],value["val"][0]=value["val"][0],value["train"][0]
            self.split.write_text(json.dumps(value),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"split"):
                self.run_fp(resume)
        finally: self.split.write_bytes(split_backup)
        mask=self.data/"masks"/self.train_names[0]
        backup=mask.read_bytes()
        try:
            pixels=np.array(Image.open(mask))
            pixels[0,0]=8 if pixels[0,0]!=8 else 1
            Image.fromarray(pixels).save(mask)
            with self.assertRaisesRegex(ValueError,"counts|weighting"):
                self.run_fp(resume)
        finally: mask.write_bytes(backup)
        self.run_fp(resume)
        last=read_checkpoint(path/"last.pt")
        self.assertEqual(last["epoch"],3)
        self.assertEqual([r["epoch"] for r in rows(path)],[1,2,3])
        self.assertTrue(any(not torch.equal(first["model"][k],last["model"][k]) for k in last["model"]))
        EVIDENCE["resume"]={"interrupted_epoch":1,"completed_epoch":3,"canonical_epochs":[1,2,3],
                           "changed_mode_split_counts_rejected":True,
                           "checkpoint_weight_metadata_mismatch_rejected":True,
                           "bitwise_uninterrupted_equivalence_claimed":False}

    def test_05_real_tta_pipeline_writes_separate_fp_report(self):
        path=OUTPUT/"real_tta"
        small_config=InferenceConfig(tile=16,overlap=4,batch_size=2,amp=True)
        with patch.object(train_fp,"build_model",side_effect=lambda **kw:TinyModel()), \
             patch.object(core,"build_model",side_effect=lambda **kw:TinyModel()), \
             patch.object(train_fp,"InferenceConfig",return_value=small_config):
            train_fp.main(self.fp_args(path,mode="inverse_sqrt",epochs=2,final=True))
        selected=read_checkpoint(path/"best_tta.pt")
        report=json.loads((path/"monitor/final_tta_false_positives.json").read_text(encoding="utf-8"))
        self.assertEqual(selected["checkpoint_kind"],"final_tta")
        self.assertEqual(selected["args"]["class_weighting"],"inverse_sqrt")
        self.assertTrue(report["available"])
        self.assertEqual(report["validation_images"],2)
        self.assertEqual(len(report["diagnostics"]["per_class"]),8)
        result=json.loads((path/"tta_selection.json").read_text(encoding="utf-8"))
        winning=[r for r in result["candidate_results"] if r["epoch"]==result["selected_epoch"]][0]
        self.assertEqual(report["diagnostics"],summarize_confusion(winning["confusion_matrix"]))
        self.assertEqual(selected["inference_config"],small_config.to_dict())
        EVIDENCE["final_tta"]={"actual_inference":True,"validation_images":2,
                              "scales":[.75,1,1.25],"flip":True,"tile":16,
                              "separate_fp_report":True}


def main():
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ORIGINAL_FILES}
    unchanged=ORIGINAL_HASHES==hashes
    summary={"synthetic_only":True,"not_competition_scores":True,"passed":result.wasSuccessful() and unchanged,
             "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "skipped":len(result.skipped),"original_sources_unchanged":unchanged,
             "evidence":EVIDENCE,"output":str(OUTPUT),
             "failure_details":[{"test":str(t),"traceback":detail} for t,detail in result.failures+result.errors]}
    (OUTPUT/"test_results.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"test_output":str(OUTPUT),"passed":summary["passed"],"tests_run":result.testsRun}))
    return 0 if summary["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())

