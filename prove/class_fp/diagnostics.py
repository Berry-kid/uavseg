"""False-positive diagnostics built from saved confusion matrices, without inference."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from prove.training_monitor.monitoring import (
    CLASSES, TrainingMonitor, atomic_text, chart, file_hash, write_json,
)


def summarize_confusion(cm):
    """Summarize a 9x9 matrix whose rows are ground truth and columns predictions.

    All ground-truth ID 0 pixels are excluded. Predictions of ID 0 on a valid
    target are false negatives. Undefined ratios are None, including IoU for a
    class absent from both targets and predictions. The input is never mutated.
    """
    try:
        raw = torch.as_tensor(cm).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Expected a numeric 9x9 confusion matrix') from error
    if raw.shape != (9, 9):
        raise ValueError(f'Expected a 9x9 confusion matrix, got {tuple(raw.shape)}')
    if raw.is_complex() or not bool(torch.isfinite(raw).all()):
        raise ValueError('Confusion counts must be finite nonnegative integers')
    if bool((raw < 0).any()) or (raw.is_floating_point() and not bool((raw == raw.round()).all())):
        raise ValueError('Confusion counts must be finite nonnegative integers')
    counts = raw.to(dtype=torch.int64).clone()
    if bool((counts < 0).any()):
        raise ValueError('Confusion counts exceed supported integer range')
    ignored = int(counts[0].sum().item())
    counts[0] = 0
    classes = []
    for class_id, name in enumerate(CLASSES, start=1):
        tp = int(counts[class_id, class_id].item())
        support = int(counts[class_id].sum().item())
        predicted = int(counts[:, class_id].sum().item())
        fp, fn = predicted - tp, support - tp
        sources = [
            {'class_id': source_id, 'class_name': CLASSES[source_id - 1],
             'pixels': int(counts[source_id, class_id].item()),
             'share_of_fp': int(counts[source_id, class_id].item()) / fp}
            for source_id in range(1, 9)
            if source_id != class_id and int(counts[source_id, class_id].item()) > 0
        ]
        sources.sort(key=lambda item: (-item['pixels'], item['class_id']))
        classes.append({
            'class_id': class_id, 'class_name': name,
            'tp': tp, 'fp': fp, 'fn': fn, 'support': support,
            'predicted_pixels': predicted,
            'precision': tp / predicted if predicted else None,
            'recall': tp / support if support else None,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
            'top_fp_sources': sources,
        })
    return {
        'schema_version': 1,
        'confusion_matrix_axes': {'rows': 'ground_truth', 'columns': 'prediction'},
        'ignored_ground_truth_id': 0,
        'ignored_ground_truth_pixels': ignored,
        'valid_pixels': int(counts.sum().item()),
        'per_class': classes,
    }


def _scalar_metrics(prefix, report):
    values = {}
    fields = {
        'precision': 'precision', 'recall': 'recall', 'tp_pixels': 'tp',
        'fp_pixels': 'fp', 'fn_pixels': 'fn', 'support_pixels': 'support',
        'predicted_pixels': 'predicted_pixels',
    }
    classes = report['per_class'] if report else [dict(class_name=name) for name in CLASSES]
    for item in classes:
        for metric, field in fields.items():
            values[f'{prefix}_{metric}_{item["class_name"]}'] = item.get(field)
    return values


class FalsePositiveMonitor(TrainingMonitor):
    """Retain the parent monitor contract and add precision/recall diagnostics."""

    def __init__(self, args, protocol, start_epoch, class_weights, train_eval_names, project_root):
        super().__init__(args, protocol, start_epoch, class_weights, train_eval_names, project_root)
        extra_sources = [Path(__file__).with_name(name)
                         for name in ('train_fp.py', 'weights.py', 'diagnostics.py')]
        extra_hashes = {str(path.resolve()): file_hash(path) for path in extra_sources if path.is_file()}
        session_file = self.directory / 'sessions' / f'{self.session}.json'
        for path in (session_file, self.directory / 'run_info.json'):
            info = json.loads(path.read_text(encoding='utf-8'))
            # An existing run_info describes its original session. Preserve it on
            # resume; the new session always records current code and weighting.
            if info.get('session') != self.session:
                continue
            info['class_weighting'] = getattr(args, 'class_weighting', None)
            info['false_positive_protocol'] = {
                'axes': {'rows': 'ground_truth', 'columns': 'prediction'},
                'ignore_ground_truth_id': 0,
                'prediction_zero_is_false_negative': True,
                'undefined_ratio': None,
                'ordinary_eval': 'same saved confusion matrices as ordinary val and fixed train subset',
                'final_tta': 'separate selected full-validation TTA confusion matrix',
            }
            info['source_sha256'].update(extra_hashes)
            write_json(path, info)

    def epoch(self, row, train_losses, val_losses, train_eval, cm):
        val_report = summarize_confusion(cm)
        train_report = (summarize_confusion(train_eval['confusion_matrix'])
                        if train_eval and train_eval.get('confusion_matrix') is not None else None)
        enriched = dict(row)
        enriched.update(_scalar_metrics('val', val_report))
        enriched.update(_scalar_metrics('train_eval', train_report))
        super().epoch(enriched, train_losses, val_losses, train_eval, cm)
        target = self.directory / 'false_positives'
        target.mkdir(exist_ok=True)
        write_json(target / f'epoch_{row["epoch"]:04d}.json', {
            'epoch': row['epoch'],
            'protocol': 'ordinary evaluation; no TTA',
            'val': val_report, 'train_eval': train_report,
        })

    def final_selection(self, result):
        # Some synthetic/legacy callers provide metadata only. Keep the parent
        # contract usable while explicitly marking missing diagnostic evidence.
        selected = [item for item in result.get('candidate_results', [])
                    if item.get('epoch') == result['selected_epoch']]
        if len(selected) > 1:
            raise ValueError('Multiple TTA candidates match the selected epoch')
        candidate = selected[0] if selected else None
        cm = candidate.get('confusion_matrix') if candidate else None
        report = {
            'selected_epoch': result['selected_epoch'],
            'selected_tta_miou': result['selected_tta_miou'],
            'validation_images': result['validation_images'],
            'protocol': 'selected full-validation TTA; separate from ordinary epoch metrics',
            'available': cm is not None,
        }
        if cm is not None:
            report['diagnostics'] = summarize_confusion(cm)
        else:
            report['reason'] = 'Selected candidate confusion matrix was not supplied'
        super().final_selection(result)
        write_json(self.directory / 'final_tta_false_positives.json', report)
        if self.writer and cm is not None:
            for key, value in _scalar_metrics('final_tta', report['diagnostics']).items():
                if value is not None:
                    self.writer.add_scalar(key, value, result['selected_epoch'])
            self.writer.flush()

    def _save(self):
        super()._save()
        path = self.directory / 'dashboard.html'
        document = path.read_text(encoding='utf-8')
        extra = '<p>False-positive diagnostics: rows are ground truth, columns predictions; GT 0 ignored. Undefined ratios are omitted from charts. See false_positives/ and final_tta_false_positives.json for source classes.</p>'
        extra += chart('Validation precision by class', self.rows,
                       ['val_precision_' + name for name in CLASSES], (0, 1))
        extra += chart('Validation recall by class', self.rows,
                       ['val_recall_' + name for name in CLASSES], (0, 1))
        for name in ('barren', 'vehicle'):
            extra += chart(name.title() + ' precision and recall: train subset / val', self.rows,
                           [prefix + '_' + metric + '_' + name
                            for prefix in ('train_eval', 'val') for metric in ('precision', 'recall')], (0, 1))
        atomic_text(path, document.replace('</html>', extra + '</html>'))