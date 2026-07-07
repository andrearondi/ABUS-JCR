"""Phase 1 — freeze the per-slice 2D GT-box label table for one split (Inv. 11).

Reads the ISO masks from the cache (built by phase1_build_cache.py), derives the
tight per-slice 2D boxes per mask component (never the 3D-box projection), and
writes the frozen table to <out-root>/labels/slice_boxes_<split>.parquet plus a
CSV mirror. Also reports the lesion-bearing vs background slice balance for the
results log.

Usage:
    python scripts/phase1_build_slice_labels.py --split Train      --out-root /home/maia-user/Andre2/outputs/phase1
    python scripts/phase1_build_slice_labels.py --split Validation --out-root /home/maia-user/Andre2/outputs/phase1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr.conventions import SLICE_AXIS
from abus_jcr.io_nrrd import discover_cases
from abus_jcr.slice_labels import build_slice_labels
from _common import add_split_args, resolve_split_root, split_label

DEFAULT_OUT_ROOT = "/home/maia-user/Andre2/outputs/phase1"

_COLS = ["volume_id", "slice_z", "r0", "c0", "r1", "c1", "component_id"]


def _write_table(df: pd.DataFrame, base: Path) -> str:
    """Write Parquet (preferred) + CSV mirror. Falls back to CSV-only if no
    Parquet engine is installed, reporting which artifacts were written."""
    df.to_csv(base.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
        return "parquet+csv"
    except Exception as e:  # pyarrow/fastparquet absent
        print(f"note: Parquet not written ({type(e).__name__}: {e}); CSV mirror only")
        return "csv-only"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 slice-label builder")
    add_split_args(parser)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cases", type=int, nargs="+", default=None,
                        help="optional subset of case ids (default: all in the split)")
    args = parser.parse_args()

    root = resolve_split_root(args)
    label = split_label(args)
    cache_root = Path(args.out_root) / "cache"
    K.assert_hash(cache_root)  # refuse a stale cache

    cases = discover_cases(root)
    if args.cases is not None:
        cases = {cid: cases[cid] for cid in args.cases if cid in cases}
    all_rows = []
    lesion_slices = 0
    total_slices = 0
    for cid in sorted(cases):
        mask_iso = np.asarray(K.open_mask(cache_root, cid))
        rows = build_slice_labels(cid, mask_iso)
        all_rows.extend(rows)
        n_slices = mask_iso.shape[SLICE_AXIS]
        total_slices += n_slices
        lesion_slices += len({r["slice_z"] for r in rows})

    df = pd.DataFrame(all_rows, columns=_COLS)
    out_dir = Path(args.out_root) / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"slice_boxes_{label}"
    fmt = _write_table(df, base)

    background = total_slices - lesion_slices
    print(f"# Phase 1 slice labels — {label} ({len(cases)} cases) -> {base} [{fmt}]\n")
    print(f"total slices           = {total_slices}")
    print(f"lesion-bearing slices  = {lesion_slices}")
    print(f"background slices       = {background}")
    print(f"2D boxes (components)   = {len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
