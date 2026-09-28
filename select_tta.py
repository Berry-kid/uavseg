"""Run or resume final B-TTA candidate selection without retraining."""
from __future__ import annotations
import argparse
from pathlib import Path

from uavseg.inference import InferenceConfig
from uavseg.selection import select_candidates


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    select_candidates(args.run_dir, args.data_root,
                      InferenceConfig(batch_size=args.batch_size, amp=args.amp),
                      device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()

