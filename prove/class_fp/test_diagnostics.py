"""CPU-only tests for matrix semantics and original-monitor integration."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from prove.class_fp.diagnostics import FalsePositiveMonitor, summarize_confusion


def example_cm():
    cm = torch.zeros((9, 9), dtype=torch.int64)
    cm[8, 8] = 40
    cm[1, 8] = 10
    cm[3, 8] = 5
    cm[8, 1] = 4
    cm[8, 0] = 1
    cm[0, 8] = 100
    return cm


def class_metric(report, class_id):
    return next(item for item in report['per_class'] if item['class_id'] == class_id)


class ConfusionTests(unittest.TestCase):
    def test_direction_and_source_ranking(self):
        report = summarize_confusion(example_cm())
        vehicle = class_metric(report, 8)
        self.assertEqual((vehicle['tp'], vehicle['fp'], vehicle['fn']), (40, 15, 5))
        self.assertEqual((vehicle['support'], vehicle['predicted_pixels']), (45, 55))
        self.assertAlmostEqual(vehicle['precision'], 40 / 55)
        self.assertAlmostEqual(vehicle['recall'], 40 / 45)
        self.assertAlmostEqual(vehicle['iou'], 40 / 60)
        self.assertEqual([item['class_id'] for item in vehicle['top_fp_sources']], [1, 3])
        self.assertAlmostEqual(vehicle['top_fp_sources'][0]['share_of_fp'], 2 / 3)
        self.assertEqual(report['valid_pixels'], 60)

    def test_ignore_gt_zero_without_mutation_and_keep_prediction_zero(self):
        cm = example_cm()
        before = cm.clone()
        report = summarize_confusion(cm)
        self.assertTrue(torch.equal(cm, before))
        self.assertEqual(report['ignored_ground_truth_pixels'], 100)
        self.assertEqual(class_metric(report, 8)['fp'], 15)
        cm[8, 0] = 0
        self.assertEqual(class_metric(summarize_confusion(cm), 8)['fn'], 4)

    def test_undefined_metrics_are_none(self):
        empty = summarize_confusion(torch.zeros((9, 9), dtype=torch.int64))
        for item in empty['per_class']:
            for field in ('precision', 'recall', 'iou'):
                self.assertIsNone(item[field])
            self.assertEqual(item['top_fp_sources'], [])
        cm = torch.zeros((9, 9), dtype=torch.int64)
        cm[2, 0] = 5
        cm[1, 3] = 2
        report = summarize_confusion(cm)
        self.assertIsNone(class_metric(report, 2)['precision'])
        self.assertEqual(class_metric(report, 2)['recall'], 0)
        self.assertEqual(class_metric(report, 2)['iou'], 0)
        self.assertEqual(class_metric(report, 3)['precision'], 0)
        self.assertIsNone(class_metric(report, 3)['recall'])
        self.assertEqual(class_metric(report, 3)['iou'], 0)
        json.dumps(report, allow_nan=False)

    def test_list_and_tensor_give_same_values(self):
        self.assertEqual(summarize_confusion(example_cm().tolist()), summarize_confusion(example_cm()))

    def test_invalid_counts_and_shapes_rejected(self):
        invalid = [torch.zeros(8, 8), [[1, 2], [3]], torch.zeros(9, 9, dtype=torch.complex64)]
        for value in (-1, .25, float('nan'), float('inf')):
            cm = torch.zeros(9, 9, dtype=torch.float64)
            cm[1, 1] = value
            invalid.append(cm)
        for cm in invalid:
            with self.subTest(matrix=str(cm)[:70]):
                with self.assertRaises(ValueError):
                    summarize_confusion(cm)


class RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.closed = False

    def add_scalar(self, name, value, epoch):
        self.scalars.append((name, value, epoch))

    def flush(self):
        pass

    def close(self):
        self.closed = True


class MonitorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='class_fp_diagnostics_')
        self.output = Path(self.temporary.name)
        self.args = SimpleNamespace(output=self.output, data_root=self.output, resume=None,
                                    no_tensorboard=True, class_weighting='inverse_sqrt')
        self.cuda_patch = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patch.start()
        self.monitor = FalsePositiveMonitor(self.args, 'synthetic center-crop', 1,
                                            torch.ones(9), ['train_example.png'], ROOT)

    def tearDown(self):
        self.monitor.close()
        self.cuda_patch.stop()
        self.temporary.cleanup()

    def write_epoch(self, number=1, include_train=True):
        cm = example_cm()
        details = summarize_confusion(cm)
        per_class = [item['iou'] if item['iou'] is not None else 0.0 for item in details['per_class']]
        train_eval = dict(miou=sum(per_class) / 8, loss=.8, images=2,
                          per_class_iou=per_class, confusion_matrix=cm.tolist()) if include_train else None
        row = dict(epoch=number, val_miou=sum(per_class) / 8,
                   val_per_class_iou=per_class, lr=3e-5)
        losses = dict(loss=.8, ce=.4, dice=.4, aux_ce=.5)
        self.monitor.epoch(row, losses, losses, train_eval, cm)
        self.assertNotIn('val_precision_vehicle', row)
        return cm

    def test_logs_dashboard_tensorboard_and_source_metadata(self):
        recorder = RecordingWriter()
        self.monitor.writer = recorder
        self.write_epoch()
        directory = self.output / 'monitor'
        row = json.loads((directory / 'epochs.jsonl').read_text(encoding='utf-8').strip())
        self.assertAlmostEqual(row['val_precision_vehicle'], 40 / 55)
        self.assertAlmostEqual(row['train_eval_recall_vehicle'], 40 / 45)
        self.assertEqual(row['val_fp_pixels_vehicle'], 15)
        self.assertIsNone(row['val_precision_building'])
        with (directory / 'epochs.csv').open(encoding='utf-8-sig', newline='') as stream:
            saved = next(csv.DictReader(stream))
        self.assertAlmostEqual(float(saved['val_precision_vehicle']), 40 / 55)
        epoch = json.loads((directory / 'epochs' / 'epoch_0001.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(epoch['metrics']['val_recall_vehicle'], 40 / 45)
        report = json.loads((directory / 'false_positives' / 'epoch_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(class_metric(report['val'], 8)['fp'], 15)
        dashboard = (directory / 'dashboard.html').read_text(encoding='utf-8')
        self.assertIn('Validation precision by class', dashboard)
        self.assertIn('Vehicle precision and recall', dashboard)
        self.monitor.set_state('training')
        self.assertEqual((directory / 'dashboard.html').read_text(encoding='utf-8').count('Validation precision by class'), 1)
        self.assertIn(('val_precision_vehicle', 40 / 55, 1), recorder.scalars)
        self.assertFalse(any(name == 'val_precision_building' for name, _, _ in recorder.scalars))
        info = json.loads((directory / 'run_info.json').read_text(encoding='utf-8'))
        self.assertEqual(info['class_weighting'], 'inverse_sqrt')
        self.assertEqual(info['args']['class_weighting'], 'inverse_sqrt')
        self.assertIn(str((ROOT / 'prove/class_fp/diagnostics.py').resolve()), info['source_sha256'])
        for name in ('train_fp.py', 'weights.py'):
            path = ROOT / 'prove/class_fp' / name
            if path.is_file():
                self.assertIn(str(path.resolve()), info['source_sha256'])

    def test_missing_train_eval_stays_missing(self):
        self.write_epoch(include_train=False)
        self.assertIsNone(self.monitor.rows[0]['train_eval_precision_vehicle'])
        report = json.loads((self.output / 'monitor/false_positives/epoch_0001.json').read_text(encoding='utf-8'))
        self.assertIsNone(report['train_eval'])

    def test_final_tta_selects_correct_epoch_and_keeps_ordinary_metrics(self):
        self.write_epoch()
        original = self.monitor.rows[0].copy()
        cm = torch.eye(9, dtype=torch.int64) * 10
        result = dict(selected_epoch=2, selected_val_miou=.7, selected_tta_miou=1.0,
                      validation_images=3,
                      candidate_results=[dict(epoch=1, confusion_matrix=example_cm().tolist()),
                                         dict(epoch=2, confusion_matrix=cm.tolist())])
        self.monitor.final_selection(result)
        report = json.loads((self.output / 'monitor/final_tta_false_positives.json').read_text(encoding='utf-8'))
        self.assertTrue(report['available'])
        self.assertEqual(class_metric(report['diagnostics'], 8)['precision'], 1)
        self.assertEqual(self.monitor.rows[0], original)
        self.assertEqual(self.monitor.final_tta['selected_tta_miou'], 1)

    def test_metadata_only_final_selection_is_explicitly_unavailable(self):
        self.monitor.final_selection(dict(selected_epoch=1, selected_val_miou=.7,
                                          selected_tta_miou=.71, validation_images=3))
        report = json.loads((self.output / 'monitor/final_tta_false_positives.json').read_text(encoding='utf-8'))
        self.assertFalse(report['available'])
        self.assertNotIn('diagnostics', report)

    def test_resume_preserves_original_run_and_tracks_new_session(self):
        self.write_epoch()
        info_path = self.output / 'monitor/run_info.json'
        original_info = info_path.read_bytes()
        self.monitor.close()
        self.args.resume = self.output / 'best.pt'
        self.monitor = FalsePositiveMonitor(self.args, 'synthetic center-crop', 2,
                                            torch.ones(9), ['train_example.png'], ROOT)
        self.assertEqual(info_path.read_bytes(), original_info)
        self.assertEqual(len(self.monitor.rows), 1)
        session = json.loads((self.output / 'monitor/sessions' / f'{self.monitor.session}.json').read_text(encoding='utf-8'))
        self.assertEqual(session['class_weighting'], 'inverse_sqrt')
        self.write_epoch(number=2)
        self.assertEqual(len(self.monitor.rows), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)