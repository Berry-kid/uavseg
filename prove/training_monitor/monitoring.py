"""Persistent epoch/step telemetry, optional TensorBoard, and an offline dashboard."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import torch
import torch.nn.functional as F
from uavseg.core import confusion_matrix, mean_iou, dice_loss

CLASSES = ['background', 'building', 'road', 'water', 'barren', 'vegetation', 'agricultural', 'vehicle']


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_losses(main_ce, main_dice, aux_ce, total):
    values = {key: float(value.detach().item()) for key, value in
              [('ce', main_ce), ('dice', main_dice), ('aux_ce', aux_ce), ('loss', total)]}
    if not all(math.isfinite(x) for x in values.values()):
        raise FloatingPointError(f'Nonfinite loss: {values}')
    return values


class LossMeter:
    def __init__(self):
        self.count = 0
        self.sums = {k: 0.0 for k in ('ce', 'dice', 'aux_ce', 'loss')}

    def add(self, values, batch_size):
        self.count += batch_size
        for key in self.sums:
            self.sums[key] += values[key] * batch_size

    def result(self):
        return {k: v / self.count if self.count else None for k, v in self.sums.items()}


def eval_losses(outputs, masks, weights):
    if not bool((masks != 0).any().item()):
        return None
    ce = F.cross_entropy(outputs['out'], masks, ignore_index=0, weight=weights)
    dice = dice_loss(outputs['out'], masks)
    aux = F.cross_entropy(outputs['aux'], masks, ignore_index=0, weight=weights)
    return numeric_losses(ce, dice, aux, ce + 0.5 * dice + 0.4 * aux)


@torch.inference_mode()
def evaluate_train_subset(model, loader, device, weights):
    was_training = model.training
    model.eval()
    cm = torch.zeros(9, 9, dtype=torch.int64, device=device)
    losses = LossMeter()
    count = 0
    started = time.perf_counter()
    try:
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            cm += confusion_matrix(outputs['out'].argmax(1), masks)
            values = eval_losses(outputs, masks, weights)
            if values is not None:
                losses.add(values, len(images))
            count += len(images)
        miou, per_class = mean_iou(cm)
        return dict(losses.result(), miou=miou, per_class_iou=per_class,
                    confusion_matrix=cm.cpu().tolist(), images=count,
                    seconds=time.perf_counter() - started)
    finally:
        model.train(was_training)


def chart(title, rows, columns, fixed_range=None):
    colors = ['#2563eb', '#ea580c', '#059669', '#7c3aed', '#db2777', '#0891b2', '#a16207', '#475569']
    series = [[(r['epoch'], r.get(key)) for r in rows
               if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])] for key in columns]
    points = [p for group in series for p in group]
    if not points:
        return '<section><h2>' + html.escape(title) + '</h2><p>Waiting for complete epochs.</p></section>'
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = fixed_range or (min(p[1] for p in points), max(p[1] for p in points))
    if ymax <= ymin:
        ymax = ymin + max(abs(ymin) * .1, 1e-6)
    if xmax == xmin:
        xmax += 1
    x = lambda value: 62 + (value-xmin) / (xmax-xmin) * 760
    y = lambda value: 230 - (value-ymin) / (ymax-ymin) * 195
    parts = ['<section><h2>' + html.escape(title) + '</h2><svg viewBox="0 0 860 275" role="img">']
    for i in range(5):
        value = ymin + (ymax-ymin) * i/4
        yy = y(value)
        parts.append(f'<line x1="62" y1="{yy}" x2="822" y2="{yy}" stroke="#e2e8f0"/><text x="55" y="{yy+4}" text-anchor="end">{value:.3g}</text>')
    parts += [f'<text x="62" y="254">Epoch {xmin}</text>', f'<text x="822" y="254" text-anchor="end">Epoch {max(r["epoch"] for r in rows)}</text>']
    for index, group in enumerate(series):
        coords = ' '.join(f'{x(a):.2f},{y(b):.2f}' for a,b in group)
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for a,b in group:
            parts.append(f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="2" fill="{color}"><title>Epoch {a}: {b:.7g}</title></circle>')
    parts.append('</svg><div class="legend">')
    parts += [f'<span style="color:{colors[i%len(colors)]}">{html.escape(k)}</span>' for i,k in enumerate(columns)]
    parts.append('</div></section>')
    return ''.join(parts)


class TrainingMonitor:
    def __init__(self, args, protocol, start_epoch, class_weights, train_eval_names, project_root):
        self.args = args
        self.directory = Path(args.output).resolve() / 'monitor'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        self.path = self.directory / 'epochs.jsonl'
        self.rows = []
        self.writer = None
        self.state = 'training'
        self.error = None
        self.final_tta = None
        self.closed = False
        source = self.path
        if not source.exists() and args.resume:
            source = Path(args.resume).resolve().parent / 'monitor/epochs.jsonl'
        if source.exists():
            if not args.resume:
                raise ValueError('Existing monitor history requires --resume or a new output directory')
            raw = source.read_text(encoding='utf-8')
            previous = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len({r['epoch'] for r in previous}) != len(previous):
                raise ValueError('Duplicate epochs in saved monitor history')
            self.rows = [r for r in previous if r['epoch'] < start_epoch]
            if source != self.path or len(self.rows) != len(previous):
                archive = self.directory / 'archives'
                archive.mkdir(exist_ok=True)
                atomic_text(archive / f'previous_{self.session}.jsonl', raw)
        eval_file = self.directory / 'train_eval_names.json'
        if args.resume and not eval_file.exists():
            old = Path(args.resume).resolve().parent / 'monitor/train_eval_names.json'
            if old.exists() and json.loads(old.read_text(encoding='utf-8')) != train_eval_names:
                raise ValueError('Resume changed the fixed training-evaluation subset')
        if eval_file.exists() and json.loads(eval_file.read_text(encoding='utf-8')) != train_eval_names:
            raise ValueError('Fixed train-eval subset changed; use a new experiment')
        write_json(eval_file, train_eval_names)
        sources = [Path(project_root) / 'train.py', Path(__file__), Path(__file__).with_name('train_monitored.py')]
        sources += sorted((Path(project_root) / 'uavseg').glob('*.py'))
        info = {'schema_version': 1, 'session': self.session, 'started_utc': now(),
                'argv': sys.argv, 'args': vars(args), 'resolved_data_root': str(args.data_root.resolve()),
                'resolved_output': str(args.output.resolve()), 'start_epoch': start_epoch,
                'val_protocol': protocol, 'class_names_ids_1_to_8': CLASSES,
                'class_weights': class_weights.detach().cpu().tolist() if class_weights is not None else None,
                'train_eval_images': len(train_eval_names), 'train_eval_protocol': 'fixed subset, center crop, no augmentation, FP32, no TTA',
                'loss_formula': 'CE + 0.5 * Dice + 0.4 * auxiliary CE; epoch losses are sample-weighted batch means',
                'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'source_sha256': {str(p.resolve()): file_hash(p) for p in sources}}
        sessions = self.directory / 'sessions'
        sessions.mkdir(exist_ok=True)
        write_json(sessions / f'{self.session}.json', info)
        if not (self.directory / 'run_info.json').exists():
            write_json(self.directory / 'run_info.json', info)
        if not args.no_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.directory / 'tensorboard'), purge_step=0, flush_secs=10)
                for row in self.rows:
                    self._tensorboard(row)
            except ImportError:
                print('TensorBoard unavailable; JSON, CSV and HTML monitoring remain enabled.')
        self.event('session_started', start_epoch=start_epoch, resumed=bool(args.resume))
        self._save()

    def event(self, kind, **fields):
        row = dict(time_utc=now(), session=self.session, event=kind, **fields)
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str, allow_nan=False) + '\n')
            stream.flush()

    def batch(self, epoch, step, steps, loss_values, optimizer, scaler, skipped):
        if step != 1 and step % self.args.log_every != 0 and step != steps:
            return
        row = dict(time_utc=now(), session=self.session, epoch=epoch, batch=step,
                   batches_in_epoch=steps, global_batch=(epoch-1)*steps+step,
                   lr=[g['lr'] for g in optimizer.param_groups], amp_scale=float(scaler.get_scale()),
                   optimizer_step_skipped=skipped, **loss_values)
        with (self.directory / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        write_json(self.directory / 'progress.json', row)

    def epoch(self, row, train_losses, val_losses, train_eval, cm):
        record = dict(time_utc=now(), session=self.session, **row)
        record.update({f'train_{k}': v for k,v in train_losses.items()})
        record.update({f'val_{k}': v for k,v in val_losses.items()})
        for name, score in zip(CLASSES, record.pop('val_per_class_iou')):
            record[f'val_iou_{name}'] = score
        record['train_eval_miou'] = train_eval['miou'] if train_eval else None
        record['train_eval_loss'] = train_eval['loss'] if train_eval else None
        record['train_eval_images'] = train_eval['images'] if train_eval else 0
        if train_eval:
            for name, score in zip(CLASSES, train_eval['per_class_iou']):
                record[f'train_eval_iou_{name}'] = score
        if any(r['epoch'] >= record['epoch'] for r in self.rows):
            raise ValueError('Refusing duplicate or non-increasing epoch records')
        details = self.directory / 'epochs'
        details.mkdir(exist_ok=True)
        write_json(details / f"epoch_{record['epoch']:04d}.json",
                   {'metrics': record, 'val_confusion_matrix': cm.detach().cpu().tolist(), 'train_eval': train_eval})
        self.rows.append(record)
        self._tensorboard(record)
        self._save()
        self.event('epoch_completed', epoch=record['epoch'], val_miou=record['val_miou'])
        print(f"Monitor: {self.directory / 'dashboard.html'}")

    def _tensorboard(self, row):
        if self.writer is None:
            return
        for key, value in row.items():
            if key != 'epoch' and isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(key, value, row['epoch'])
        self.writer.flush()

    def set_state(self, state, **fields):
        self.state = state
        self.event(state, **fields)
        self._save()

    def final_selection(self, result):
        self.final_tta = {k: result[k] for k in ('selected_epoch', 'selected_val_miou', 'selected_tta_miou', 'validation_images')}
        write_json(self.directory / 'final_tta.json', self.final_tta)
        if self.writer:
            self.writer.add_scalar('final_tta/miou', result['selected_tta_miou'], result['selected_epoch'])
            self.writer.flush()
        self.event('final_tta_completed', **self.final_tta)
        self._save()

    def fail(self, exception):
        self.error = {'type': type(exception).__name__, 'message': str(exception), 'traceback': traceback.format_exc()}
        write_json(self.directory / f'failure_{self.session}.json', self.error)
        self.set_state('interrupted' if isinstance(exception, KeyboardInterrupt) else 'failed', error=self.error)

    def _save(self):
        atomic_text(self.path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in self.rows))
        columns = ['epoch'] + sorted({k for r in self.rows for k in r} - {'epoch'})
        temporary = self.directory / 'epochs.csv.tmp'
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.directory / 'epochs.csv')
        status = {'state': self.state, 'updated_utc': now(), 'session': self.session,
                  'completed_epochs_recorded': len(self.rows),
                  'latest_epoch': self.rows[-1]['epoch'] if self.rows else None,
                  'best_val_miou': max((r['val_miou'] for r in self.rows), default=None),
                  'final_tta': self.final_tta, 'error': self.error}
        write_json(self.directory / 'status.json', status)
        title = html.escape(str(self.args.output))
        doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="15"><title>Training monitor</title>'
        doc += '<style>body{font:15px system-ui;margin:32px auto;padding:0 24px;max-width:1000px;background:#f8fafc;color:#0f172a}section{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:18px;margin:20px 0}h2{font-size:18px}svg{width:100%;font:12px system-ui;fill:#64748b}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        doc += f'<h1>Training monitor</h1><p>{title}</p><pre>{html.escape(json.dumps(status, ensure_ascii=False, indent=2))}</pre>'
        doc += '<p>train_eval: fixed training subset, no augmentation; ordinary val: center crop, no TTA. Final TTA is recorded separately. Augmented training loss is not directly comparable to evaluation loss.</p>'
        doc += chart('mIoU', self.rows, ['val_miou','train_eval_miou'], (0,1))
        doc += chart('Loss', self.rows, ['train_loss','val_loss','train_eval_loss'])
        doc += chart('Learning rate used during epoch', self.rows, ['lr'])
        doc += chart('Validation IoU by class', self.rows, ['val_iou_'+c for c in CLASSES], (0,1))
        doc += '<p>Auto-refresh every 15 seconds. Canonical data: epochs.jsonl / epochs.csv.</p></html>'
        atomic_text(self.directory / 'dashboard.html', doc)

    def close(self):
        if not self.closed and self.writer:
            self.writer.close()
        self.closed = True