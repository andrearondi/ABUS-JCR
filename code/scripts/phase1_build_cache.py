"""Phase 1 — materialise the isotropic volume+mask cache for one split (Inv. 6).

Resamples every case in the split ONCE to ISO_SPACING_MM isotropic and writes it
to a hash-named cache directory (see abus_jcr.cache). Train + Validation only;
the Test cache is deferred to Phase 5 (Inv. 9). CPU-only, no GPU.

Usage:
    python scripts/phase1_build_cache.py --split Train      --out-root /home/maia-user/Andre2/outputs/phase1
    python scripts/phase1_build_cache.py --split Validation --out-root /home/maia-user/Andre2/outputs/phase1
    # local Validation:
    python scripts/phase1_build_cache.py --split-root /path/to/Validation --out-root ./_p1_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr.conventions import SLICE_AXIS
from abus_jcr.io_nrrd import discover_cases, load_array
from abus_jcr.preprocess import resample_case, preprocess_hash, iso_shape
from _common import add_split_args, resolve_split_root, split_label

DEFAULT_OUT_ROOT = "/home/maia-user/Andre2/outputs/phase1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 isotropic cache builder")
    add_split_args(parser)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help=f"phase-1 output root (cache goes to <out-root>/cache; default {DEFAULT_OUT_ROOT})")
    parser.add_argument("--force", action="store_true",
                        help="rewrite cases even if the cache dir already exists")
    parser.add_argument("--cases", type=int, nargs="+", default=None,
                        help="optional subset of case ids (default: all in the split)")
    args = parser.parse_args()

    if args.split == "Test" and not args.split_root:
        print("REFUSED — Test cache is deferred to Phase 5 (Inv. 9). Not building here.")
        return 2

    root = resolve_split_root(args)
    label = split_label(args)
    cache_root = Path(args.out_root) / "cache"
    cases = discover_cases(root)
    if args.cases is not None:
        cases = {cid: cases[cid] for cid in args.cases if cid in cases}
    h = preprocess_hash()

    print(f"# Phase 1 cache build — {label} ({len(cases)} cases)")
    print(f"preprocess_hash = {h}")
    print(f"cache dir       = {K.cache_dir(cache_root)}\n")

    total_slices = 0
    for i, cid in enumerate(sorted(cases), 1):
        vol_dir = K.cache_dir(cache_root) / "vol" / f"VOL_{cid}.npy"
        if vol_dir.exists() and not args.force:
            print(f"[{i}/{len(cases)}] case {cid}: exists, skip (use --force to rebuild)")
            continue
        vol, _ = load_array(cases[cid].data)
        mask, _ = load_array(cases[cid].mask)
        vol = np.asarray(vol, dtype=np.uint8)
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        vol_iso, mask_iso, meta = resample_case(vol, mask)
        K.write_case(cache_root, cid, vol_iso, mask_iso, meta)
        nz = vol_iso.shape[SLICE_AXIS]
        total_slices += nz
        assert tuple(meta["iso_shape"]) == iso_shape(vol.shape)
        print(f"[{i}/{len(cases)}] case {cid}: native {vol.shape} -> iso {vol_iso.shape} "
              f"({nz} slices, mask voxels {int(mask_iso.sum())})")

    print(f"\n**DONE** — {label}: cache at {K.cache_dir(cache_root)}")
    print(f"total slices this run = {total_slices}; preprocess_hash = {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
