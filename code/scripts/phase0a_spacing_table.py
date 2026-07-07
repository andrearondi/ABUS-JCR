"""Phase 0a — spacing / shape table.

Per-case storage shape (header-only, no decompress) + injected physical spacing
+ physical extent (mm), and the per-case assertion
``count(d0) >= count(d1) >= count(d2)`` — the voxel-count ordering that
justifies the spacing→axis assignment (finer spacing = more samples). Exit
nonzero if the ordering fails for any case.

Usage:
    python scripts/phase0a_spacing_table.py --split Train
    python scripts/phase0a_spacing_table.py --split-root /path/to/Validation
"""

from __future__ import annotations

import argparse
import sys

from abus_jcr.io_nrrd import discover_cases, read_shape, injected_spacing_storage
from _common import add_split_args, resolve_split_root, split_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0a spacing/shape table")
    add_split_args(parser)
    args = parser.parse_args()

    root = resolve_split_root(args)
    label = split_label(args)
    cases = discover_cases(root)
    spacing = injected_spacing_storage()  # (d0, d1, d2) mm

    print(f"# Spacing / shape table — {label} ({len(cases)} cases)\n")
    print(f"Injected spacing (storage d0,d1,d2) = {spacing} mm "
          f"(NRRD header identity placeholder ignored)\n")
    print("| case | shape (d0,d1,d2) | extent mm (d0,d1,d2) | d0>=d1>=d2 |")
    print("|---|---|---|---|")

    failures = []
    for cid in sorted(cases):
        shape = read_shape(cases[cid].data)
        extent = tuple(round(shape[i] * spacing[i], 1) for i in range(3))
        ordered = shape[0] >= shape[1] >= shape[2]
        if not ordered:
            failures.append((cid, shape))
        print(f"| {cid} | {shape} | {extent} | {'OK' if ordered else 'FAIL'} |")

    print()
    if failures:
        print(f"**FAIL** — {len(failures)} case(s) violate d0>=d1>=d2:")
        for cid, shape in failures:
            print(f"  - case {cid}: shape {shape}")
        return 1
    print(f"**PASS** — count(d0) >= count(d1) >= count(d2) for all {len(cases)} cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
