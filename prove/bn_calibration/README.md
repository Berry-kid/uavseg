# BN calibration experiment

This experiment changes only BatchNorm running statistics in a copy of the current epoch-62 best checkpoint. It does not run an optimizer or alter original project files.

## Fixed protocol

- Input: runs/runs/continue-v1/best_tta.pt and its saved split.
- Calibration: all 6,297 training images, one pass, fixed seed 2026, center crop 768, ImageNet normalization, FP32.
- No training masks, validation images, or test images enter calibration.
- All learned parameters are frozen. Dropout is disabled. Only BatchNorm layers use training mode.
- Running statistics are reset and estimated by sample-weighted averaging of batch means and unbiased within-batch variances. This is not an exact pooled global variance estimate.
- Batch size 4; the final five images are split into batches of 3 and 2 so every image is used exactly once and pooled BN never sees a singleton.
- Original and calibrated models are both evaluated on this machine, using identical inputs and settings.
- Ordinary diagnostic evaluation: all 699 validation images plus the saved 256-image training subset, center crop 768, FP32, no augmentation/TTA, original loss formula.
- Final comparison: all 699 full validation images; saved TTA scales 0.75/1/1.25, horizontal flip, tile 768, overlap 192, batch 2, AMP.
- Calibration is fixed before evaluation. No search over calibration seeds, counts, or recipes.
- Original checkpoint and source hashes must match after the run. Every learned parameter and every non-BN-statistics buffer must be exactly unchanged. Independent NumPy and project confusion matrices must agree.

## Run from the project directory

PowerShell (current machine):

    & D:\anaconda\envs\pytorch\python.exe -u -B prove/bn_calibration/compare_bn.py --checkpoint runs/runs/continue-v1/best_tta.pt --data-root "2026-低空图像语义分割赛道-训练集/train/train" --output prove/bn_calibration/run_20260927 --batch-size 4 --workers 2

Output must be a NEW subdirectory inside prove. Existing experiment output is never overwritten.

## Outputs

- config.json: fixed settings, source checkpoint SHA256, software and hardware.
- calibration_names.json: saved training filenames used by calibration.
- input_manifest.json: original source hashes and input size/mtime metadata.
- before_center.json / after_center.json: ordinary loss components and class metrics.
- calibration.json: calibration coverage and parameter-invariance checks.
- tta_partial.npz: paired per-image confusion matrices, saved during evaluation.
- tta_results.json: complete paired TTA metrics.
- bn_calibrated.pt: independent complete model copy, with newly measured scores.
- results.json: final results and original-file preservation checks.
- progress.json / failure.json: progress or failure details.

The copied checkpoint is an experimental candidate, not an automatic replacement for the original. A validation improvement does not establish a test-score improvement; a decline under this one recipe does not rule out every BN-related cause.

