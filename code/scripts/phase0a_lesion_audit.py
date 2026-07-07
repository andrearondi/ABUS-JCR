"""Phase 0a — lesion audit (26-connectivity).

Per-case connected-component count and size distribution, plus split-level raw
and "effective" (>= LESION_MIN_VOXELS) histograms of lesions-per-volume, and a
one-line single-lesion-dominance verdict that gates the relational framing.

Usage:
    python scripts/phase0a_lesion_audit.py --split Train
    python scripts/phase0a_lesion_audit.py --split-root /path/to/Validation
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from abus_jcr.conventions import LESION_MIN_VOXELS
from abus_jcr.io_nrrd import discover_cases, load_array
from abus_jcr.lesions import audit_mask
from _common import add_split_args, resolve_split_root, split_label


def _fmt_hist(counter: Counter) -> str:
    return "{" + ", ".join(f"{k}: {counter[k]}" for k in sorted(counter)) + "}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0a lesion audit (26-conn)")
    add_split_args(parser)
    args = parser.parse_args()

    root = resolve_split_root(args)
    label = split_label(args)
    cases = discover_cases(root)

    print(f"# Lesion audit — {label} ({len(cases)} cases, "
          f"26-conn, floor={LESION_MIN_VOXELS} voxels)\n")
    print("| case | n_raw | n_effective | sizes (voxels) |")
    print("|---|---|---|---|")

    raw_hist: Counter = Counter()
    eff_hist: Counter = Counter()
    multi_cases = []
    for cid in sorted(cases):
        mask, _ = load_array(cases[cid].mask)
        out = audit_mask(np.asarray(mask))
        raw_hist[out["n_components_raw"]] += 1
        eff_hist[out["n_components_effective"]] += 1
        if out["n_components_effective"] > 1:
            multi_cases.append((cid, out["n_components_effective"]))
        sizes = ", ".join(str(s) for s in sorted(out["sizes"], reverse=True))
        print(f"| {cid} | {out['n_components_raw']} | {out['n_components_effective']} | {sizes} |")

    print()
    print(f"- raw lesions/volume histogram: `{_fmt_hist(raw_hist)}`")
    print(f"- effective (>= {LESION_MIN_VOXELS} vox) histogram: `{_fmt_hist(eff_hist)}`")
    n = len(cases)
    single = eff_hist.get(1, 0)
    frac = single / n if n else 0.0
    # "Dominance" is descriptive: single-lesion volumes are the overwhelming
    # majority. It tolerates a small minority of multi-lesion volumes (listed
    # explicitly so nothing is hidden); it is NOT a strict "all single" claim.
    holds = frac >= 0.90 and single == max(eff_hist.values(), default=0)
    verdict = "HOLDS" if holds else "does NOT hold — reconsider the single-lesion assumption"
    print(f"- single-lesion (effective) volumes: {single}/{n} = {frac:.1%}")
    if multi_cases:
        listed = ", ".join(f"case {cid} ({k} lesions)" for cid, k in multi_cases)
        print(f"- multi-lesion (effective) volumes: {len(multi_cases)} — {listed}")
    else:
        print("- multi-lesion (effective) volumes: none")
    print(f"- **single-lesion dominance {verdict}**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
