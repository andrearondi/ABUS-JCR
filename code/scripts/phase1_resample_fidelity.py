"""Phase 1 — characterise the iso->native->official IoU ceiling (Inv. 6, exit #2).

For every case in a split: take the native GT mask box (native official space) and
the ISO mask box read from the cache, map the iso box back to native via the
recorded inverse affine, and score official IoU. This is exactly the round-trip
Phase 3 performs to score iso candidates against the native official GT box, so it
runs on the ACTUAL cached data (not a re-resample).

Output: a per-case table, the ceiling distribution (min/median/percentiles and
counts below key thresholds), and a CSV at
<out-root>/resample_fidelity_<split>.csv handed to Phase 3 as its reconstruction-
consistency tolerance. Exits nonzero only if a case falls at/below
RESAMPLE_IOU_FLOOR (the FROC-hit safety margin) — the small-lesion tail ABOVE the
floor is reported, not failed.

Usage:
    python scripts/phase1_resample_fidelity.py --split Train      --out-root /home/maia-user/Andre2/outputs/phase1
    python scripts/phase1_resample_fidelity.py --split Validation --out-root /home/maia-user/Andre2/outputs/phase1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr.conventions import RESAMPLE_IOU_FLOOR, IOU_HIT_THRESHOLD, ISO_SPACING_MM
from abus_jcr.geometry import (
    mask_to_box_storage,
    mask_to_official_box,
    storage_box_to_official,
    iso_storage_to_native_storage,
    iou_official,
)
from abus_jcr.io_nrrd import discover_cases, load_array
from _common import add_split_args, resolve_split_root, split_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 resample-fidelity characterisation")
    add_split_args(parser)
    parser.add_argument("--out-root", default="/home/maia-user/Andre2/outputs/phase1")
    parser.add_argument("--cases", type=int, nargs="+", default=None)
    args = parser.parse_args()

    root = resolve_split_root(args)
    label = split_label(args)
    cache_root = Path(args.out_root) / "cache"
    K.assert_hash(cache_root)  # the iso masks must come from the current cache

    cases = discover_cases(root)
    if args.cases is not None:
        cases = {cid: cases[cid] for cid in args.cases if cid in cases}

    rows = []
    print(f"# Phase 1 resample fidelity — {label} @ {ISO_SPACING_MM} mm ({len(cases)} cases)\n")
    print(f"{'case':>5} {'iou':>7}")
    for cid in sorted(cases):
        mask_native, _ = load_array(cases[cid].mask)
        mask_native = (np.asarray(mask_native) > 0).astype(np.uint8)
        official_native = mask_to_official_box(mask_native)          # native GT box

        mask_iso = np.asarray(K.open_mask(cache_root, cid))
        meta = K.read_meta(cache_root, cid)
        box_native = iso_storage_to_native_storage(mask_to_box_storage(mask_iso), meta)
        iou = iou_official(storage_box_to_official(box_native), official_native)
        rows.append({"volume_id": cid, "iou": iou})
        print(f"{cid:>5} {iou:>7.3f}")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_root) / f"resample_fidelity_{label}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    v = df["iou"].to_numpy()
    print(f"\n## Ceiling distribution — {label} @ {ISO_SPACING_MM} mm ({len(v)} cases) -> {out_csv}")
    print(f"min={v.min():.3f}  p10={np.percentile(v,10):.3f}  median={np.median(v):.3f}  max={v.max():.3f}")
    for thr in (0.85, 0.80, 0.70, RESAMPLE_IOU_FLOOR, IOU_HIT_THRESHOLD):
        print(f"  # cases < {thr:.2f} = {int((v < thr).sum())}")

    below_floor = df[df["iou"] <= RESAMPLE_IOU_FLOOR]
    below_hit = df[df["iou"] <= IOU_HIT_THRESHOLD]
    if len(below_hit):
        print(f"\n**FAIL (catastrophic)** — {len(below_hit)} case(s) at/below the 0.3 hit threshold: "
              f"{list(below_hit['volume_id'])}")
        return 1
    if len(below_floor):
        print(f"\n**FAIL** — {len(below_floor)} case(s) at/below RESAMPLE_IOU_FLOOR "
              f"({RESAMPLE_IOU_FLOOR}): {list(below_floor['volume_id'])}. A perfect candidate losing "
              f"this much IoU signals a coordinate/affine regression — investigate.")
        return 1
    print(f"\n**PASS** — all {len(v)} cases retain IoU > {RESAMPLE_IOU_FLOOR} (FROC-hit safety margin, "
          f"hit threshold {IOU_HIT_THRESHOLD}). Small-lesion ceiling tail reported above is expected "
          f"resampling quantization, handed to Phase 3 as its tolerance input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
