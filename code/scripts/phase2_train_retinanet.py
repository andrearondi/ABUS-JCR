"""Train one RetinaNet detector (Inv. 2, 9, 10, 14).

8 runs total per the Training Matrix:
  --regime fold --fold {0..4}     (seed DET_FOLD_SEED)   -> retinanet_fold{f}.pt
  --regime full --seed {0,1,2}                           -> retinanet_full_seed{s}.pt

Requires the [2.0] reconciliation to have been done first (constants in
conventions.py (B) must equal the Train-derived values). CUDA required.

Usage (server):
    python scripts/phase2_train_retinanet.py --regime fold --fold 0 \
        --phase1-out /home/maia-user/Andre2/outputs/phase1 --out-root /home/maia-user/Andre2/outputs/phase2
    python scripts/phase2_train_retinanet.py --regime full --seed 0 ...
"""

from __future__ import annotations

import argparse
import sys

from abus_jcr.detect.train import train_detector
from _phase2_common import add_phase2_paths, assert_device, cache_root, load_manifest, load_slice_boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 RetinaNet trainer")
    add_phase2_paths(parser)
    parser.add_argument("--regime", required=True, choices=["fold", "full"])
    parser.add_argument("--fold", type=int, default=None, help="fold id 0..4 (regime=fold)")
    parser.add_argument("--seed", type=int, default=None, help="seed 0/1/2 (regime=full)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if args.regime == "fold" and args.fold is None:
        parser.error("--regime fold requires --fold")
    if args.regime == "full" and args.seed is None:
        parser.error("--regime full requires --seed")
    assert_device(args.device)
    fold_or_seed = args.fold if args.regime == "fold" else args.seed

    manifest = load_manifest(args)
    sb_train = load_slice_boxes(args, "Train")
    sb_val = load_slice_boxes(args, "Validation")

    summary = train_detector(
        regime=args.regime, fold_or_seed=fold_or_seed,
        cache_root=cache_root(args), manifest=manifest,
        slice_boxes_train=sb_train, slice_boxes_val=sb_val,
        out_root=args.out_root, num_workers=args.num_workers, device=args.device,
    )
    print(f"\n**DONE** {summary['run']}: best epoch {summary['best_epoch']} "
          f"(val {summary['best_val_loss']:.4f}), ran {summary['epochs_ran']} epochs")
    print(f"checkpoint = {summary['checkpoint']}")
    print(f"log        = {summary['log']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
