"""Small CPU checks for the BN-only experiment; no project inputs are modified."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from compare_bn import balanced_batches, bn_buffer_keys, recalibrate, verify_bn_only


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(0.9)
        self.bn = torch.nn.BatchNorm2d(2, momentum=0.2)
        self.conv = torch.nn.Conv2d(2, 2, 1, bias=True)
        self.register_buffer('non_bn_counter', torch.tensor(7))
        self.observed_modes = []

    def forward(self, x):
        self.observed_modes.append((self.training, self.dropout.training, self.bn.training))
        return self.conv(self.bn(self.dropout(x)))


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class BnCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def calibrated_toy(self):
        torch.manual_seed(2026)
        model = ToyModel()
        model.train()
        model.conv.eval()  # Exercise restoration of mixed per-module modes.
        modes_before = {name: child.training for name, child in model.named_modules()}
        before = snapshot(model)
        images = torch.arange(9 * 2 * 2 * 2, dtype=torch.float32).reshape(9, 2, 2, 2) / 7
        batches = balanced_batches(len(images), 4, 2026)
        loader = [images[indices] for indices in batches]
        report = recalibrate(model, loader, torch.device('cpu'))
        return model, before, images, batches, report, modes_before

    def test_batches_cover_every_training_sample_once_without_singletons(self):
        train_names = [f'train_{i:04d}.png' for i in range(6297)]
        val_names = {'val_0000.png', 'val_0001.png'}
        batches = balanced_batches(len(train_names), 4, 2026)
        observed = [train_names[i] for batch in batches for i in batch]
        self.assertEqual(len(observed), 6297)
        self.assertEqual(set(observed), set(train_names))
        self.assertFalse(set(observed) & val_names)
        self.assertEqual(len(set(observed)), len(observed))
        self.assertEqual([len(b) for b in batches[-2:]], [3, 2])
        self.assertTrue(all(2 <= len(b) <= 4 for b in batches))
        self.assertEqual(batches, balanced_batches(6297, 4, 2026))

    def test_batches_handle_small_and_uneven_counts(self):
        for batch_size in (3, 4, 5, 8):
            for count in range(2, 36):
                with self.subTest(count=count, batch_size=batch_size):
                    batches = balanced_batches(count, batch_size, 17)
                    self.assertEqual(sorted(i for b in batches for i in b), list(range(count)))
                    self.assertTrue(all(2 <= len(b) <= batch_size for b in batches))
        for count, batch_size in ((0, 4), (1, 4), (9, 1), (9, 2)):
            with self.assertRaises(ValueError):
                balanced_batches(count, batch_size, 2026)

    def test_known_statistics_dropout_off_and_modes_restored(self):
        model, before, images, batches, report, modes_before = self.calibrated_toy()
        expected_mean = images.mean(dim=(0, 2, 3))
        expected_var = sum(images[ids].var(dim=(0, 2, 3), unbiased=True) * len(ids)
                           for ids in batches) / len(images)
        torch.testing.assert_close(model.bn.running_mean, expected_mean, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(model.bn.running_var, expected_var, rtol=1e-6, atol=1e-6)
        self.assertTrue(all(modes == (False, False, True) for modes in model.observed_modes))
        self.assertEqual({name: child.training for name, child in model.named_modules()}, modes_before)
        self.assertEqual(model.bn.momentum, 0.2)
        self.assertEqual(model.bn.num_batches_tracked.item(), len(batches))
        self.assertEqual(report['images'], len(images))
        self.assertEqual(report['batches'], len(batches))
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertFalse(report['dropout'])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_only_bn_statistics_change(self):
        model, before, _, _, _, _ = self.calibrated_toy()
        after = snapshot(model)
        allowed = bn_buffer_keys(model)
        report = verify_bn_only(before, after, allowed)
        self.assertTrue(report['all_learned_parameters_unchanged'])
        self.assertTrue(report['all_non_bn_statistics_buffers_unchanged'])
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter.detach()), name)
        self.assertTrue(torch.equal(before['non_bn_counter'], after['non_bn_counter']))
        self.assertEqual(set(report['changed_buffers']),
                         {'bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked'})

    def test_rejects_tampered_learned_parameter_and_non_bn_buffer(self):
        model, before, _, _, _, _ = self.calibrated_toy()
        for key in ('conv.weight', 'bn.weight', 'non_bn_counter'):
            with self.subTest(key=key):
                after = snapshot(model)
                after[key] += 1
                with self.assertRaisesRegex(AssertionError, 'Non-BN-statistics tensor changed'):
                    verify_bn_only(before, after, bn_buffer_keys(model))

    def test_rejects_nonfinite_or_negative_variance_and_missing_keys(self):
        model, before, _, _, _, _ = self.calibrated_toy()
        after = snapshot(model)
        after['bn.running_mean'][0] = float('nan')
        with self.assertRaises(FloatingPointError):
            verify_bn_only(before, after, bn_buffer_keys(model))
        after = snapshot(model)
        after['bn.running_var'][0] = -1
        with self.assertRaisesRegex(AssertionError, 'Negative BN variance'):
            verify_bn_only(before, after, bn_buffer_keys(model))
        after = snapshot(model)
        del after['conv.bias']
        with self.assertRaisesRegex(AssertionError, 'State keys changed'):
            verify_bn_only(before, after, bn_buffer_keys(model))

    def test_rejects_singleton_and_restores_modes_after_failure(self):
        model = ToyModel().eval()
        with self.assertRaisesRegex(ValueError, 'Singleton'):
            recalibrate(model, [torch.ones(1, 2, 2, 2)], torch.device('cpu'))
        self.assertFalse(model.training)
        self.assertFalse(model.bn.training)
        self.assertFalse(model.dropout.training)
        self.assertEqual(model.bn.momentum, 0.2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
