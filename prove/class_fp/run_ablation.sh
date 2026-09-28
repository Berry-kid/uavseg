#!/usr/bin/env bash
set -euo pipefail

FP_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$FP_PROJECT_ROOT"
FP_PYTHON="${FP_PYTHON:-python}"
FP_MODE="${1:-inverse_sqrt}"
case "$FP_MODE" in inverse|inverse_sqrt|none) ;; *) echo "Expected inverse, inverse_sqrt, or none" >&2; exit 2 ;; esac
FP_RUN="${FP_RUN:-prove/class_fp/runs/${FP_MODE}-v1}"

if [[ -z "${FP_DATA_ROOT:-}" ]]; then
    for fp_path in "2026-低空图像语义分割赛道-训练集/train" "2026-低空图像语义分割赛道-训练集/train/train" "datasets/train/train"; do
        if [[ -d "$fp_path/images" && -d "$fp_path/masks" ]]; then
            FP_DATA_ROOT="$fp_path"
            break
        fi
    done
fi
if [[ -z "${FP_DATA_ROOT:-}" || ! -d "$FP_DATA_ROOT/images" || ! -d "$FP_DATA_ROOT/masks" ]]; then
    echo "Set FP_DATA_ROOT to the directory containing images/ and masks/." >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
"$FP_PYTHON" -B -c "import torch, torchvision, numpy, PIL, tqdm; assert torch.cuda.is_available(), 'CUDA unavailable'; assert hasattr(torch.serialization, 'safe_globals'), 'PyTorch safe_globals required'; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0))"
fp_extra=()
if [[ -n "${FP_RESUME:-}" ]]; then
    fp_extra=(--resume "$FP_RESUME")
fi
mkdir -p prove/class_fp/logs
fp_stamp="$(date +%Y%m%d_%H%M%S)"
"$FP_PYTHON" -u -B prove/class_fp/train_fp.py \
    --data-root "$FP_DATA_ROOT" \
    --split-file prove/class_fp/reference/split.json \
    --output "$FP_RUN" \
    --class-weighting "$FP_MODE" \
    --epochs 70 \
    --lr 3e-5 \
    --batch-size 4 \
    --crop-size 768 \
    --backbone resnet101 \
    --workers 4 \
    --seed 2026 \
    --patience 0 \
    --keep-best 3 \
    --tta-batch-size 2 \
    --train-eval-images 256 \
    --train-eval-every 1 \
    "${fp_extra[@]}" 2>&1 | tee "prove/class_fp/logs/${FP_MODE}_${fp_stamp}.log"

"$FP_PYTHON" -B prove/class_fp/compare_runs.py --run "$FP_RUN"

