"""Phase 0a — GT-box double-check (tolerance 0).

For every case in a split: recompute the whole-mask official box and compare it
to the shipped ``bbx_labels.csv`` row. Emits a per-case residual table and an
overall PASS/FAIL. Exit code is nonzero if any case has a nonzero residual, so
the runbook step fails loudly on a corrupted case.

Usage:
    python scripts/phase0a_gt_doublecheck.py --split Train
    python scripts/phase0a_gt_doublecheck.py --split-root /path/to/Validation
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from abus_jcr.io_nrrd import discover_cases, load_array
from abus_jcr.gt_labels import load_gt_documented, to_official_gt, doublecheck_case
from _common import add_split_args, resolve_split_root, split_label

_FIELDS = ["coordX", "coordY", "coordZ", "x_length", "y_length", "z_length"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0a GT-box double-check (tol 0)")
    add_split_args(parser)
    args = parser.parse_args()

    root = resolve_split_root(args)
    label = split_label(args)
    cases = discover_cases(root)
    gt = to_official_gt(load_gt_documented(root / "bbx_labels.csv")).set_index("public_id")

    print(f"# GT-box double-check — {label} ({len(cases)} cases, tolerance 0)\n")
    header = "| case | " + " | ".join(_FIELDS) + " | max |"
    print(header)
    print("|" + "---|" * (len(_FIELDS) + 2))

    failures = []
    max_overall = 0.0
    for cid in sorted(cases):
        mask, _ = load_array(cases[cid].mask)
        residual = doublecheck_case(np.asarray(mask), gt.loc[cid])
        row_max = max(residual.values())
        max_overall = max(max_overall, row_max)
        cells = " | ".join(f"{residual[f]:.3g}" for f in _FIELDS)
        print(f"| {cid} | {cells} | {row_max:.3g} |")
        if row_max > 0:
            failures.append((cid, residual))

    print()
    if failures:
        print(f"**FAIL** — {len(failures)} case(s) with nonzero residual "
              f"(max {max_overall:.6g}):")
        for cid, residual in failures:
            print(f"  - case {cid}: {residual}")
        return 1
    print(f"**PASS** — max residual == 0 across all {len(cases)} cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
