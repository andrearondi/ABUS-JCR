"""Phase 1 — build the frozen split + stratified k-fold manifest (Inv. 9, 10).

Lists every available case with its official split; Train volumes get a seeded,
B/M-stratified fold in {0..K-1}; val/test get fold = -1. Deterministic and
hash-stable. Written to <out-root>/manifest.csv — the single source of truth for
every later phase.

Usage:
    python scripts/phase1_build_manifest.py --data-root /home/maia-user/Andre2/data --out-root /home/maia-user/Andre2/outputs/phase1
    # local (Validation only present):
    python scripts/phase1_build_manifest.py --data-root /Users/.../Dataset --out-root ./_p1_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from abus_jcr.manifest import build_manifest

DEFAULT_DATA_ROOT = "/home/maia-user/Andre2/data"
DEFAULT_OUT_ROOT = "/home/maia-user/Andre2/outputs/phase1"

# official split-name -> canonical key
_SPLIT_DIRS = {"Train": "train", "Validation": "val", "Test": "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 manifest builder")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"dataset root holding Train/Validation/Test dirs (default {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help=f"phase-1 output root (default {DEFAULT_OUT_ROOT})")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    roots = {}
    for dirname, canon in _SPLIT_DIRS.items():
        p = data_root / dirname
        if p.exists():
            roots[canon] = p
        else:
            print(f"note: {dirname} split not found at {p} — omitted from manifest")

    if not roots:
        print(f"ERROR — no split dirs under {data_root}")
        return 1

    df = build_manifest(roots)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "manifest.csv"
    df.to_csv(out_csv, index=False)

    print(f"# Phase 1 manifest — {len(df)} cases written to {out_csv}\n")
    print("split counts:")
    print(df["split"].value_counts().to_string())
    print("\nfold counts (train only):")
    print(df[df["split"] == "train"]["fold"].value_counts().sort_index().to_string())
    print("\nB/M per fold (train):")
    tr = df[df["split"] == "train"]
    print(tr.groupby(["fold", "label"]).size().to_string())
    print(f"\nmanifest_hash = {df['manifest_hash'].iloc[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
