"""[2.0] Train design-constant probe — the HARD GATE before any training.

Reads the Train iso cache + slice_boxes_Train, computes the design statistics in
iso space (no Val, no zoom proxy), writes ``train_det_stats.json``, and prints
``derive_constants(Train)`` BESIDE the provisional ``conventions.py (B)`` values.
If they differ, update the (B) block to the Train-derived values and record the
change in RESULTS_PHASE_2.md [2.0] before running [2.1]+. This script does not
mutate conventions.py — reconciliation is a human decision (Inv. 9).

Usage (server):
    python scripts/phase2_train_stats.py --phase1-out /home/maia-user/Andre2/outputs/phase1 \
        --out-root /home/maia-user/Andre2/outputs/phase2
Local Val smoke (method check only — NOT a design decision):
    python scripts/phase2_train_stats.py --phase1-out ./_p1_out --out-root ./_p2_out --probe-split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from abus_jcr import conventions as C
from abus_jcr.detect import det_stats as DS
from _phase2_common import add_phase2_paths, cache_root, load_manifest, load_slice_boxes


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _reconcile(derived: dict) -> bool:
    """Print a derived-vs-provisional table; return True iff all fields match."""
    prov = {
        "min_size": C.DET_MIN_SIZE, "max_size": C.DET_MAX_SIZE,
        "image_mean": C.DET_IMAGE_MEAN, "image_std": C.DET_IMAGE_STD,
        "anchor_base_sizes": tuple(C.DET_ANCHOR_BASE_SIZES),
        "anchor_aspect_ratios": tuple(C.DET_ANCHOR_ASPECT_RATIOS),
    }
    tol = {"image_mean": 0.02, "image_std": 0.02}
    print("\n## [2.0] Reconciliation — derive_constants(Train) vs conventions.py (B)\n")
    print(f"{'field':<22}{'Train-derived':<26}{'provisional (B)':<26}{'verdict'}")
    all_match = True
    for k in prov:
        d = derived[k]; p = prov[k]
        if k in tol:
            match = abs(float(d) - float(p)) <= tol[k]
        else:
            match = (tuple(d) if isinstance(d, (list, tuple)) else d) == p
        all_match = all_match and match
        print(f"{k:<22}{_fmt(d):<26}{_fmt(p):<26}{'MATCH' if match else 'DIFFER -> update (B)'}")
    print()
    if all_match:
        print("**GATE PASS** — provisional (B) reproduced by the Train probe; constants stand.")
    else:
        print("**GATE ACTION REQUIRED** — update conventions.py (B) to the Train-derived values above,")
        print("record the change in RESULTS_PHASE_2.md [2.0], THEN run [2.1]+. No training before this.")
    return all_match


def main() -> int:
    parser = argparse.ArgumentParser(description="[2.0] Train design-constant probe (HARD GATE)")
    add_phase2_paths(parser)
    parser.add_argument("--probe-split", default="train", choices=["train", "val"],
                        help="which split to probe; ONLY 'train' sets constants (val is a method smoke)")
    args = parser.parse_args()

    manifest = load_manifest(args)
    split_label = "Train" if args.probe_split == "train" else "Validation"
    slice_boxes = load_slice_boxes(args, split_label)

    if args.probe_split == "val":
        # method-only smoke: relabel val volumes as 'train' locally so the probe runs on them.
        manifest = manifest.copy()
        manifest.loc[manifest["split"] == "val", "split"] = "train"
        print("NOTE: --probe-split val is a METHOD SMOKE (ballpark only); it sets NO design constant.")

    stats = DS.probe_train_stats(cache_root(args), manifest, slice_boxes)
    out_dir = Path(args.out_root) / "stats"
    path = DS.write_stats(stats, out_dir)

    print(f"# [2.0] Train-stats probe -> {path}\n")
    print(f"n_train_volumes        = {stats['n_train_volumes']}")
    print(f"frame d0 (min/med/max) = {stats['frame_d0_min']}/{stats['frame_d0_median']}/{stats['frame_d0_max']}")
    print(f"frame d1 (min/med/max) = {stats['frame_d1_min']}/{stats['frame_d1_median']}/{stats['frame_d1_max']}")
    print(f"n_boxes                = {stats['n_boxes']}")
    print(f"diag p1/p50/p99        = {stats['diag_pct']['1']:.1f}/{stats['diag_pct']['50']:.1f}/{stats['diag_pct']['99']:.1f}")
    print(f"aspect p10/p50/p90     = {stats['aspect_pct']['10']:.3f}/{stats['aspect_pct']['50']:.3f}/{stats['aspect_pct']['90']:.3f}")
    print(f"intensity mean/std     = {stats['intensity_mean']:.4f}/{stats['intensity_std']:.4f} (n={stats['intensity_n_slices']})")
    print(f"components/slice max   = {stats['components_per_slice_max']} (frac>1 = {stats['components_per_slice_frac_gt1']:.3f})")

    all_match = _reconcile(stats["derived"])
    return 0 if (args.probe_split == "val" or all_match) else 3


if __name__ == "__main__":
    sys.exit(main())
