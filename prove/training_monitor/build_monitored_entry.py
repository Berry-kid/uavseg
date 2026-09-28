from pathlib import Path
import hashlib,json
root=Path(r'C:\Users\47348\Desktop\AIC2\qw-advice')
out=root/'prove/training_monitor'
if (out/'train_monitored.py').exists():
    raise FileExistsError('One-time bootstrap only: refusing to overwrite the maintained monitoring entry')
source=(root/'train.py').read_text(encoding='utf-8')
original=source
(out/'baseline_train.py').write_text(original,encoding='utf-8')

def replace(old,new):
    global source
    if source.count(old)!=1:
        raise RuntimeError(f'Expected one source anchor, found {source.count(old)}: {old[:100]}')
    source=source.replace(old,new,1)

replace('from __future__ import annotations', '''from __future__ import annotations

import os
import sys
import time
from pathlib import Path as _Path
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
PROJECT_ROOT = _Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from monitoring import TrainingMonitor, LossMeter, numeric_losses, eval_losses, evaluate_train_subset
ACTIVE_MONITOR = None
''')
replace('default=Path("runs/deeplabv3-opt")', 'default=Path("runs/monitored-v1")')
replace('    return p.parse_args(argv)', '''    p.add_argument("--log-every", type=int, default=50, help="Record a batch sample every N training batches")
    p.add_argument("--train-eval-images", type=int, default=256,
                   help="Fixed no-augmentation training subset: 0 disables; -1 evaluates all training images")
    p.add_argument("--train-eval-every", type=int, default=1, help="Evaluate the fixed training subset every N epochs")
    p.add_argument("--no-tensorboard", action="store_true", help="Keep JSON/CSV/HTML without TensorBoard events")
    return p.parse_args(argv)''')
replace('def run_training(args):\n', '''def run_training(args):
    global ACTIVE_MONITOR
    if args.log_every < 1 or args.train_eval_every < 1 or args.train_eval_images < -1:
        raise ValueError("Invalid monitoring interval or train-eval sample count")
''')
replace('    # Training loop\n', '''    # A separate generator prevents diagnostic evaluation from consuming the
    # training DataLoader's global random sequence.
    count = len(train_names) if args.train_eval_images == -1 else min(args.train_eval_images, len(train_names))
    subset_order = torch.randperm(len(train_names), generator=torch.Generator().manual_seed(args.seed + 913)).tolist()
    train_eval_names = sorted(train_names[i] for i in subset_order[:count])
    train_eval_loader = None
    if train_eval_names:
        train_eval_loader = DataLoader(
            SegmentationDataset(image_dir, mask_dir, train_eval_names, args.crop_size, False),
            batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed + 1913))
    monitor = TrainingMonitor(args, protocol, start_epoch, class_weights, train_eval_names, PROJECT_ROOT)
    ACTIVE_MONITOR = monitor
    print(f"Monitoring enabled; fixed train-eval subset: {len(train_eval_names)} images")

    # Training loop
''')
replace('    for epoch in range(start_epoch, args.epochs + 1):\n        model.train()', '''    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats(device)
        monitor.set_state("training", epoch=epoch)
        train_meter = LossMeter()
        skipped_steps = 0
        train_started = time.perf_counter()
        model.train()''')
replace('        for images, masks, _ in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):',
        '        for batch_index, (images, masks, _) in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"), 1):')
replace('            scaler.scale(loss).backward()', '''            loss_values = numeric_losses(main_ce, main_dice, aux_loss, loss)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()''')
replace('            running_loss += loss.item()', '''            running_loss += loss.item()
            skipped = scaler.get_scale() < scale_before
            skipped_steps += int(skipped)
            train_meter.add(loss_values, images.shape[0])
            monitor.batch(epoch, batch_index, len(train_loader), loss_values, optimizer, scaler, skipped)''')
replace('        # Update scheduler', '''        train_seconds = time.perf_counter() - train_started

        # Update scheduler''')
replace('        # Validation\n', '''        # Validation
        val_started = time.perf_counter()
        val_meter = LossMeter()
        monitor.set_state("validation", epoch=epoch)
''')
replace('                pred = model(images)["out"].argmax(1)\n                cm += confusion_matrix(pred, masks)', '''                val_outputs = model(images)
                pred = val_outputs["out"].argmax(1)
                cm += confusion_matrix(pred, masks)
                values = eval_losses(val_outputs, masks, class_weights)
                if values is not None:
                    val_meter.add(values, len(images))''')
replace('        avg_loss = running_loss / len(train_loader)', '''        avg_loss = running_loss / len(train_loader)
        val_seconds = time.perf_counter() - val_started
        train_eval = None
        if train_eval_loader is not None and (epoch % args.train_eval_every == 0 or epoch == args.epochs):
            monitor.set_state("train_subset_evaluation", epoch=epoch)
            train_eval = evaluate_train_subset(model, train_eval_loader, device, class_weights)
            print(f"  Fixed train-eval mIoU(no augmentation): {train_eval['miou']:.5f}")''')
replace('        # Early stopping only monitors ordinary val. Final TTA selection runs after this loop.', '''        monitor.epoch(
            {"epoch": epoch, "lr": lr_used, "lr_next_epoch": optimizer.param_groups[0]["lr"],
             "val_miou": miou, "best_val_miou": best_miou, "val_per_class_iou": per_class,
             "train_seconds": train_seconds, "val_seconds": val_seconds,
             "train_eval_seconds": train_eval["seconds"] if train_eval else 0.0,
             "epoch_seconds": time.perf_counter() - epoch_started,
             "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
             "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
             "amp_scale": float(scaler.get_scale()), "optimizer_steps_skipped": skipped_steps,
             "improved": improved, "patience_counter": patience_counter,
             "train_images_seen": train_meter.count, "val_images": len(val_ds)},
            train_meter.result(), val_meter.result(), train_eval, cm)

        # Early stopping only monitors ordinary val. Final TTA selection runs after this loop.''')
old=source[source.index('def main(argv=None):'):source.index('\n\nif __name__ == "__main__":')]
replace(old,'''def main(argv=None):
    global ACTIVE_MONITOR
    ACTIVE_MONITOR = None
    args = parse_args(argv)
    try:
        run_training(args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.skip_final_tta:
            print("Final B-TTA selection deferred. Run select_tta.py when ready.")
            ACTIVE_MONITOR.set_state("completed_tta_deferred")
        else:
            ACTIVE_MONITOR.set_state("final_tta_selection")
            result = select_candidates(args.output, args.data_root,
                                       InferenceConfig(batch_size=args.tta_batch_size), device="cuda")
            ACTIVE_MONITOR.final_selection(result)
            ACTIVE_MONITOR.set_state("completed")
    except BaseException as exc:
        if ACTIVE_MONITOR is not None:
            ACTIVE_MONITOR.fail(exc)
        raise
    finally:
        if ACTIVE_MONITOR is not None:
            ACTIVE_MONITOR.close()
''')
compile(source,str(out/'train_monitored.py'),'exec')
(out/'train_monitored.py').write_text(source,encoding='utf-8')
provenance={'base_train_path':str(root/'train.py'),'base_train_sha256':hashlib.sha256((root/'train.py').read_bytes()).hexdigest(),
            'base_snapshot':'baseline_train.py','scope':'Independent monitoring entry; original train.py and uavseg sources unchanged.'}
(out/'implementation_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print('Created monitored entry and read-only baseline snapshot; original train.py unchanged.')