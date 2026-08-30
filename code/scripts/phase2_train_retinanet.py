"""Train one RetinaNet detector (Inv. 2, 9, 10, 14).

8 runs total per the Training Matrix:
  --regime fold --fold {0..4}     (seed DET_FOLD_SEED)   -> retinanet_fold{f}.pt
  --regime full --seed {0,1,2}                           -> retinanet_full_seed{s}.pt

Requires the [2.0] reconciliation to have been done first (constants in
conventions.py (B) must equal the Train-derived values). CUDA required.

Usage (server):
    python scripts/phase2_train_retinanet.py --regime fold --fold 0 \
        --phase1-out $WORK/outputs/phase1 --out-root $WORK/outputs/phase2
    python scripts/phase2_train_retinanet.py --regime full --seed 0 ...
"""

from __future__ import annotations

import argparse
import sys

from abus_jcr import cache as K
from abus_jcr import conventions as C
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
    parser.add_argument("--flip-stack-axis", type=int, default=None, choices=[0, 1],
                        help="in-plane axis of the (C, d0, d1) stack the mirror flip acts on. "
                             "Default = augment.TRAIN_AUGMENT['flip_stack_axis'] (0 = d0 = the "
                             "measured LATERAL axis, DEPLOYED since 2026-08-08). 1 = d1 = the "
                             "measured DEPTH/BEAM axis, which Inv. 13 forbids — it is the "
                             "pre-2026-08-08 default, kept reachable only to reproduce the archived "
                             "arm. See results/AXIS_CHECK.md, runbooks/RB_AUG_FLIP_AB.md and "
                             "RB_FOLD_FLIP.md. Changing this REQUIRES --run-suffix so deployed "
                             "checkpoints stay untouched.")
    parser.add_argument("--run-suffix", default="",
                        help="append to the run name -> checkpoints/<run>_<suffix>/. Use for any "
                             "experimental arm; without it an arm would overwrite a deployed run.")
    args = parser.parse_args()

    if args.regime == "fold" and args.fold is None:
        parser.error("--regime fold requires --fold")
    if args.regime == "full" and args.seed is None:
        parser.error("--regime full requires --seed")
    assert_device(args.device)
    # Refuse a cache that was not built by THIS axis profile, before spending GPU-hours.
    # `cache_dir` is named by `preprocess_hash`, which the profile changes, so a mismatched
    # --phase1-out fails here rather than after loading the dataset (or, worse, succeeding
    # against the wrong substrate). Print the pairing so every training log records it.
    print(f"# axis profile = {C.AXIS_PROFILE} | spacing_storage_mm = {C.SPACING_STORAGE_MM}")
    print(f"# iso cache    = {K.cache_dir(cache_root(args))}")
    K.assert_hash(cache_root(args))
    fold_or_seed = args.fold if args.regime == "fold" else args.seed

    # A non-default augmentation MUST land under its own run name. Without this guard an
    # experimental arm silently overwrites a deployed checkpoint and every downstream number
    # that was derived from it — unrecoverable without a retrain.
    policy = None
    if args.flip_stack_axis is not None:
        from abus_jcr.augment import TRAIN_AUGMENT
        if args.flip_stack_axis != TRAIN_AUGMENT["flip_stack_axis"] and not args.run_suffix:
            parser.error("--flip-stack-axis differs from the deployed default; pass --run-suffix "
                         "so this arm cannot overwrite a deployed checkpoint")
        policy = dict(TRAIN_AUGMENT, flip_stack_axis=int(args.flip_stack_axis))
        print(f"# AUGMENT OVERRIDE: flip_stack_axis={args.flip_stack_axis} "
              f"({'d1 = deployed default' if args.flip_stack_axis == 1 else 'd0 = corrected lateral flip'})")

    manifest = load_manifest(args)
    sb_train = load_slice_boxes(args, "Train")
    sb_val = load_slice_boxes(args, "Validation")

    summary = train_detector(
        regime=args.regime, fold_or_seed=fold_or_seed,
        cache_root=cache_root(args), manifest=manifest,
        slice_boxes_train=sb_train, slice_boxes_val=sb_val,
        out_root=args.out_root, num_workers=args.num_workers, device=args.device,
        augment_policy=policy, run_suffix=args.run_suffix,
    )
    print(f"\n**DONE** {summary['run']}: ran {summary['epochs_ran']} epochs "
          f"(all saved; select on epoch >= {summary['select_min_epoch']})")
    print(f"epochs_dir = {summary['epochs_dir']}")
    print(f"log        = {summary['log']}")
    print(f"NEXT: python scripts/phase2_select_checkpoint.py --run {summary['run']} "
          "(post-hoc linked val CPM -> deployed <run>.pt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
