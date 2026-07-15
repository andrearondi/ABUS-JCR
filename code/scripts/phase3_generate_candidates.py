"""[3.5] Generate the FROZEN candidate pool for a split (Inv. 8, 9, 10, 14).

TRAIN: out-of-fold (``retinanet_fold{f}`` for fold-``f`` volumes) -> one record + one
pred CSV. VAL: the 3 full-train seeds -> one record (3 pools tagged by
``detector_of_origin``) + one pred CSV per seed. TEST path is code-complete but GUARDED:
it refuses to run without ``--phase5-execute`` (Inv. 9 — Test cache is materialised and
Test is touched only in Phase 5).

Writes, under ``<out-root>/candidates/``:
    candidates_<split>.parquet (+ .csv)        # the frozen feature record (Phase 4 input)
    pred_<split>[_full_seed{s}].csv            # official pred CSV(s) consumed by evaluate()

Usage (server, GPU):
    python scripts/phase3_generate_candidates.py --split train --device cuda
    python scripts/phase3_generate_candidates.py --split val   --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from abus_jcr import conventions as C
from abus_jcr.candidates.generate import generate_split
from abus_jcr.candidates.record import write_candidate_record, to_official_pred_csv
from _phase3_common import (add_phase3_paths, assert_device, cache_root, checkpoints_dir,
                            load_manifest, load_official_gt)


def _write_pred_csvs(pool, split: str, out_dir: Path):
    """Train -> one combined pred CSV; Val/Test -> one per detector_of_origin (seed)."""
    written = []
    if split == "train":
        p = out_dir / "pred_train.csv"
        to_official_pred_csv(pool, prob_col="score_max", path=p)
        written.append(p)
    else:
        for det in sorted(pool["detector_of_origin"].unique()):
            sub = pool[pool["detector_of_origin"] == det].reset_index(drop=True)
            p = out_dir / f"pred_{split}_{det}.csv"
            to_official_pred_csv(sub, prob_col="score_max", path=p)
            written.append(p)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.5] generate frozen candidate pool")
    add_phase3_paths(parser)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--op-score-thresh", type=float, default=C.LINK_OP_SCORE_THRESH,
                        help="frozen operating point (default conventions.LINK_OP_SCORE_THRESH; "
                             "set to the [3.4]-calibrated value if conventions not yet updated)")
    parser.add_argument("--phase5-execute", action="store_true",
                        help="required to run --split test (Inv. 9: Test touched only in Phase 5)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.split == "test" and not args.phase5_execute:
        raise SystemExit("Refusing --split test without --phase5-execute (Inv. 9: Test is "
                         "generated only in Phase 5, reusing this exact code).")
    assert_device(args.device)

    manifest = load_manifest(args)
    gt_df = load_official_gt(args, args.split)

    pool = generate_split(
        manifest, cache_root(args), checkpoints_dir(args), args.split, gt_df,
        op_score_thresh=args.op_score_thresh, progress=True)

    out_dir = Path(args.out_root) / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_base = out_dir / f"candidates_{args.split}"
    fmt = write_candidate_record(pool, rec_base)
    preds = _write_pred_csvs(pool, args.split, out_dir)

    n_pos = int((pool["label"] == "pos").sum())
    n_neg = int((pool["label"] == "neg").sum())
    n_ign = int((pool["label"] == "ignore").sum())
    n_vol = pool["public_id"].nunique()
    print(f"\n# [3.5] candidates ({args.split}) — op={args.op_score_thresh}\n")
    print(f"record        = {rec_base}.* ({fmt})")
    print(f"total candidates = {len(pool)} over {n_vol} volumes "
          f"({len(pool)/n_vol:.1f}/vol) across {pool['detector_of_origin'].nunique()} detector pool(s)")
    print(f"labels        = pos {n_pos} / neg {n_neg} / ignore {n_ign}")
    print("pred CSV(s):")
    for p in preds:
        print(f"  {p}")
    print("\nNext: [3.6] baseline FROC reads this record per seed pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
