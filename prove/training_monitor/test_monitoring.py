"""Synthetic GPU smoke/regression tests; these scores are NOT competition results."""
from __future__ import annotations
import csv
import hashlib
import uuid
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
os.environ['CUDA_CACHE_PATH'] = str(HERE / 'test_cuda_cache')
os.environ['TORCH_HOME'] = str(HERE / 'test_torch_cache')

import numpy as np
from PIL import Image
import torch
import monitoring
import train_monitored as monitored
from uavseg.core import read_checkpoint

spec = importlib.util.spec_from_file_location('monitor_baseline_snapshot', HERE / 'baseline_train.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class TinyModel(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1), torch.nn.BatchNorm2d(8),
                                          torch.nn.ReLU(), torch.nn.Dropout2d(.2))
        self.head = torch.nn.Conv2d(8, 9, 1)
        self.aux = torch.nn.Conv2d(8, 9, 1)
        self.nonfinite = nonfinite

    def forward(self, x):
        z = self.layers(x)
        out = self.head(z)
        if self.nonfinite:
            out = out * float('nan')
        return {'out': out, 'aux': self.aux(z)}


class MonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('GPU integration tests require CUDA')
        torch.set_num_threads(4)
        cls.directory = HERE / ('smoke_output_' + uuid.uuid4().hex[:8])
        if cls.directory.exists():
            raise FileExistsError('Use a fresh smoke_output directory for this test run')
        cls.directory.mkdir()
        (cls.directory / 'SYNTHETIC_TEST_DATA.txt').write_text(
            'Synthetic ten-image / tiny-network integration test. Scores are not real training results.\n', encoding='utf-8')
        cls.data = cls.directory / 'data'
        for name in ('images','masks'):
            (cls.data / name).mkdir(parents=True)
        rng = np.random.default_rng(47)
        y,x = np.indices((32,32))
        mask = ((x // 4) % 8 + 1).astype(np.uint8)
        mask[:2,:2] = 0
        for i in range(10):
            pixels = rng.integers(0,256,size=(32,32,3),dtype=np.uint8)
            Image.fromarray(pixels).save(cls.data / 'images' / f'{i:02d}.png')
            Image.fromarray(np.roll(mask,i,axis=0)).save(cls.data / 'masks' / f'{i:02d}.png')

    def arguments(self, directory, epochs=3, final=False):
        args = ['--data-root',str(self.data),'--output',str(directory),'--epochs',str(epochs),
                '--batch-size','2','--crop-size','16','--workers','0','--val-ratio','.2',
                '--no-pretrained','--backbone','resnet50','--patience','0','--keep-best','2']
        if not final:
            args.append('--skip-final-tta')
        return args

    def monitored_arguments(self, directory, epochs=3, final=False):
        return self.arguments(directory,epochs,final) + ['--train-eval-images','4','--log-every','2']

    @staticmethod
    def rows(directory):
        return [json.loads(line) for line in (directory/'monitor/epochs.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_01_passive_monitor_does_not_change_training_and_logs_real_values(self):
        base_dir, mon_dir = self.directory/'baseline', self.directory/'monitored'
        with patch.object(baseline,'build_model',side_effect=lambda **kw:TinyModel()):
            baseline.main(self.arguments(base_dir))
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()):
            monitored.main(self.monitored_arguments(mon_dir))
        expected, actual = read_checkpoint(base_dir/'last.pt'), read_checkpoint(mon_dir/'last.pt')
        for key, value in expected['model'].items():
            self.assertTrue(torch.equal(value,actual['model'][key]),key)
        self.assertEqual(expected['miou'],actual['miou'])
        rows = self.rows(mon_dir)
        self.assertEqual([r['epoch'] for r in rows],[1,2,3])
        self.assertTrue(all(r['train_eval_images']==4 for r in rows))
        self.assertTrue(all(r['train_images_seen']==8 and r['val_images']==2 for r in rows))
        for row in rows:
            self.assertAlmostEqual(row['train_loss'],row['train_ce']+.5*row['train_dice']+.4*row['train_aux_ce'],places=5)
            self.assertAlmostEqual(row['val_loss'],row['val_ce']+.5*row['val_dice']+.4*row['val_aux_ce'],places=5)
            self.assertTrue(0 <= row['train_eval_miou'] <= 1)
            self.assertEqual(sum(k.startswith('val_iou_') for k in row),8)
            self.assertEqual(sum(k.startswith('train_eval_iou_') for k in row),8)
            self.assertGreater(row['gpu_peak_allocated_mb'],0)
        with (mon_dir/'monitor/epochs.csv').open(encoding='utf-8-sig',newline='') as f:
            self.assertEqual(len(list(csv.DictReader(f))),3)
        steps = (mon_dir/'monitor/steps.jsonl').read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(steps),9)
        dashboard=(mon_dir/'monitor/dashboard.html').read_text(encoding='utf-8')
        self.assertIn('<svg',dashboard)
        self.assertIn('train_eval_miou',dashboard)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(mon_dir/'monitor/tensorboard')).Reload()
        values=events.Scalars('val_miou')
        self.assertEqual([r.step for r in values],[1,2,3])
        for saved,row in zip(values,rows):self.assertAlmostEqual(saved.value,row['val_miou'],places=6)

    def test_02_interruption_resume_and_rollback_keep_canonical_history(self):
        directory=self.directory/'resume'
        original_epoch=monitoring.TrainingMonitor.epoch
        def stop_after_first(instance,*args,**kwargs):
            original_epoch(instance,*args,**kwargs)
            if instance.rows[-1]['epoch']==1:
                raise KeyboardInterrupt('Synthetic interruption after a committed epoch')
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()), patch.object(monitoring.TrainingMonitor,'epoch',stop_after_first):
            with self.assertRaises(KeyboardInterrupt):monitored.main(self.monitored_arguments(directory))
        self.assertEqual(json.loads((directory/'monitor/status.json').read_text())['state'],'interrupted')
        shutil.copyfile(directory/'last.pt',directory/'epoch1.pt')
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()):
            monitored.main(self.monitored_arguments(directory)+['--resume',str(directory/'last.pt')])
        self.assertEqual([r['epoch'] for r in self.rows(directory)],[1,2,3])
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()):
            monitored.main(self.monitored_arguments(directory)+['--resume',str(directory/'epoch1.pt')])
        self.assertEqual([r['epoch'] for r in self.rows(directory)],[1,2,3])
        self.assertTrue(list((directory/'monitor/archives').glob('previous_*.jsonl')))
        self.assertEqual(json.loads((directory/'monitor/status.json').read_text())['state'],'completed_tta_deferred')

    def test_03_final_tta_score_remains_separate(self):
        directory=self.directory/'final_tta'
        fake={'selected_epoch':1,'selected_val_miou':.42,'selected_tta_miou':.91,'validation_images':2}
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()), patch.object(monitored,'select_candidates',return_value=fake) as choose:
            monitored.main(self.monitored_arguments(directory,epochs=1,final=True)+['--no-tensorboard'])
        choose.assert_called_once()
        rows=self.rows(directory)
        self.assertEqual(rows[0]['val_miou'],read_checkpoint(directory/'last.pt')['val_miou'])
        self.assertNotEqual(rows[0]['val_miou'],.91)
        self.assertEqual(json.loads((directory/'monitor/final_tta.json').read_text())['selected_tta_miou'],.91)
        self.assertEqual(json.loads((directory/'monitor/status.json').read_text())['state'],'completed')

    def test_04_nonfinite_loss_records_failure_before_checkpointing(self):
        directory=self.directory/'failure'
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel(nonfinite=True)):
            with self.assertRaises(FloatingPointError):
                monitored.main(self.monitored_arguments(directory,epochs=1)+['--no-tensorboard'])
        self.assertEqual(json.loads((directory/'monitor/status.json').read_text())['state'],'failed')
        self.assertTrue(list((directory/'monitor').glob('failure_*.json')))
        self.assertFalse((directory/'last.pt').exists())


    def test_05_zero_lr_checkpoint_can_restart_and_resume_its_new_schedule(self):
        original_dir = self.directory / 'monitored'
        source_dir = self.directory / 'completed_source'
        source_dir.mkdir()
        shutil.copyfile(original_dir / 'split.json', source_dir / 'split.json')
        old = read_checkpoint(original_dir / 'last.pt')
        # Model a completed old run whose polynomial schedule reached zero.
        for group in old['optimizer']['param_groups']:
            group['lr'] = 0.0
        old['scheduler'].update(total_iters=45, last_epoch=45, _last_lr=[0.0])
        old.pop('warmup_scheduler', None)
        old.pop('lr_schedule_start_epoch', None)
        old['val_miou'] = old['miou'] = old['best_val_miou'] = .99
        old_path = source_dir / 'completed_zero_lr.pt'
        torch.save(old, old_path)
        old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
        directory = self.directory / 'continued'
        arguments = self.monitored_arguments(directory, epochs=10) + ['--lr','0.0001','--no-tensorboard']
        original_epoch = monitoring.TrainingMonitor.epoch
        def stop_after_fifth(instance, *args, **kwargs):
            original_epoch(instance, *args, **kwargs)
            if instance.rows[-1]['epoch'] == 5:
                raise KeyboardInterrupt('Interrupt the restarted schedule')
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()), patch.object(monitoring.TrainingMonitor,'epoch',stop_after_fifth):
            with self.assertRaises(KeyboardInterrupt):
                monitored.main(arguments + ['--resume',str(old_path),'--restart-optimizer'])
        interrupted = read_checkpoint(directory / 'last.pt')
        self.assertEqual(interrupted['lr_schedule_start_epoch'],4)
        self.assertEqual(interrupted['epoch'],5)
        self.assertFalse(torch.equal(interrupted['model']['head.weight'], old['model']['head.weight']))
        with patch.object(monitored,'build_model',side_effect=lambda **kw:TinyModel()):
            monitored.main(arguments + ['--resume',str(directory / 'last.pt')])
        rows = [row for row in self.rows(directory) if row['epoch'] > 3]
        self.assertEqual([row['epoch'] for row in rows],list(range(4,11)))
        self.assertTrue(all(row['lr'] > 0 for row in rows))
        self.assertAlmostEqual(rows[0]['lr'],1e-5)
        self.assertAlmostEqual(rows[1]['lr'],2.8e-5)
        self.assertAlmostEqual(rows[2]['lr'],4.6e-5)
        final = read_checkpoint(directory / 'last.pt')
        self.assertEqual(final['lr_schedule_start_epoch'],4)
        self.assertEqual(final['scheduler']['total_iters'],2)
        self.assertEqual(final['scheduler']['last_epoch'],2)
        self.assertAlmostEqual(final['optimizer']['param_groups'][0]['lr'],0.0)
        self.assertEqual((directory/'split.json').read_bytes(),(source_dir/'split.json').read_bytes())
        self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(),old_hash)
        candidates = __import__('json').loads((directory/'candidates/index.json').read_text())['candidates']
        self.assertIn(3,[entry['epoch'] for entry in candidates])

    def test_06_schedule_changes_require_an_explicit_fresh_restart(self):
        source_dir = self.directory / 'completed_source'
        old_path = source_dir / 'completed_zero_lr.pt'
        with self.assertRaisesRegex(ValueError,'requires --resume'):
            monitored.main(self.monitored_arguments(self.directory/'missing_resume') + ['--restart-optimizer'])
        with self.assertRaisesRegex(ValueError,'Changing the total epoch budget'):
            monitored.main(self.monitored_arguments(self.directory/'silent_extend',epochs=10) + ['--resume',str(old_path)])
        with self.assertRaisesRegex(ValueError,'new output directory'):
            monitored.main(self.monitored_arguments(source_dir,epochs=10) + ['--resume',str(old_path),'--restart-optimizer'])
        with self.assertRaisesRegex(ValueError,'must exceed'):
            monitored.main(self.monitored_arguments(self.directory/'empty_restart',epochs=3) + ['--resume',str(old_path),'--restart-optimizer'])



if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MonitoringTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'synthetic_smoke_only':True,'tests_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),
            'checks':['unchanged parameter updates','epoch loss components and class IoU','CSV and TensorBoard',
                      'interruption/resume/rollback','separate final TTA','nonfinite-loss diagnostics',
                      'zero-LR restart and schedule continuity','explicit restart safeguards'],
            'synthetic_output':str(MonitoringTests.directory)}
    (HERE/'test_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)