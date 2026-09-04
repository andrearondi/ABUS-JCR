"""[4.7] Tie-ambiguity BOUNDS for every reported rung — the [MIG-5b] read-out.

**Why this exists.** The tie-break audit on the [MIG-5] smoke (2026-09-04) flagged 2–8 ambiguous
``fp`` values on essentially every rung — the runbook's expectation ("real pools are not
exposed") was wrong: vertical risers (recall rises at constant fp when a threshold step admits a
TP and no FP) are intrinsic to every FROC curve on a 30-lesion pool. What the audit cannot say is
whether any riser sits where the evaluator actually READS.

**What is bounded, exactly.** The vendored ``_get_key_recall`` sorts the swept curve with a
non-stable quicksort and, per key rate, reads ``values[-1]`` of the fp ≤ key side and
``values[0]`` of the fp ≥ key side — so among entries TIED at the bracketing fp *values* the
chosen recall is sort-order-dependent (platform/pandas-build dependent; the 2026-08-30 fixture
measured 0.3841 laptop vs 0.4250 cluster from identical inputs). The interpolated read is
monotone increasing in both endpoint recalls, so substituting the min/max recall within each of
the two tie groups yields hard lower/upper bounds on each key recall, hence on CPM. This script
reports ``cpm_lo``/``cpm_hi`` per (seed, rung) from ``grid.json``'s stored curves.

Inv. 3: the vendored evaluator is read, never modified; this is analysis beside it.

Usage (LOGIN, seconds):
    python scripts/phase4_tie_bound.py --grid $WORK/outputs_iso/phase4/grid/grid.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from abus_jcr import conventions as C


def key_recall_bounds(fp, recall, key_fp) -> tuple:
    """(lo, hi) for the official interpolated recall at ``key_fp`` over all tie orderings.

    Mirrors ``_official_det_score._interpolate_recall_at_fp`` exactly: fp_0 / fp_1 are fp
    *values* (order-independent); only the recall picked within each tied group varies.
    """
    fp = np.asarray(fp, dtype=float)
    rc = np.asarray(recall, dtype=float)
    less, more = fp <= key_fp, fp >= key_fp
    if not less.any():
        return 0.0, 0.0                                   # below the cheapest point: hard 0
    if not more.any():
        m = float(rc.max())                               # above the dearest point: hard max
        return m, m
    fp0 = float(fp[less].max())
    fp1 = float(fp[more].min())
    r0_lo, r0_hi = float(rc[fp == fp0].min()), float(rc[fp == fp0].max())
    r1_lo, r1_hi = float(rc[fp == fp1].min()), float(rc[fp == fp1].max())
    t = (key_fp - fp0) / (fp1 - fp0 + 1e-8)               # in [0, 1): monotone in r0 and r1
    return r0_lo + (r1_lo - r0_lo) * t, r0_hi + (r1_hi - r0_hi) * t


def cpm_bounds(fp, recall, keys=None) -> dict:
    keys = tuple(C.KEY_FP if keys is None else keys)
    lo_hi = [key_recall_bounds(fp, recall, k) for k in keys]
    lo = float(np.mean([a for a, _ in lo_hi]))
    hi = float(np.mean([b for _, b in lo_hi]))
    return {"cpm_lo": lo, "cpm_hi": hi, "spread": hi - lo,
            "keys_affected": int(sum(1 for a, b in lo_hi if b - a > 1e-12))}


def main() -> int:
    ap = argparse.ArgumentParser(description="[MIG-5b] tie-ambiguity CPM bounds from grid.json")
    ap.add_argument("--grid", required=True)
    args = ap.parse_args()
    g = json.load(open(args.grid))

    print(f"# [MIG-5b] tie-ambiguity bounds — official read reproduced with min/max tie choice")
    print(f"  {'seed':<5} {'rung':<10} {'cpm':>8} {'cpm_lo':>8} {'cpm_hi':>8} "
          f"{'spread':>8} {'keys':>5}")
    worst = 0.0
    for seed in sorted(g["per_seed"]):
        for rung in sorted(g["per_seed"][seed]):
            r = g["per_seed"][seed][rung]
            if not r.get("fp"):
                continue
            b = cpm_bounds(r["fp"], r["recall"])
            ok = r["cpm"] >= b["cpm_lo"] - 1e-9 and r["cpm"] <= b["cpm_hi"] + 1e-9
            worst = max(worst, b["spread"])
            print(f"  {seed:<5} {rung:<10} {r['cpm']:>8.4f} {b['cpm_lo']:>8.4f} "
                  f"{b['cpm_hi']:>8.4f} {b['spread']:>8.4f} {b['keys_affected']:>5}"
                  f"{'' if ok else '   <-- REPORTED CPM OUTSIDE BOUNDS: mirror logic wrong, STOP'}")
    lesion_quantum = 1.0 / (30 * len(C.KEY_FP))
    print(f"\n  max spread = {worst:.4f}  (one lesion at one rate = {lesion_quantum:.4f})")
    if worst <= 1e-12:
        print("  VERDICT: no key-rate read touches a tie — every reported CPM is "
              "order-INDEPENDENT; the audit's risers all sit between read points.")
    elif worst < lesion_quantum:
        print("  VERDICT: ambiguity below one lesion-rate quantum — report CPMs with this "
              "bound stated once; no ordering claim in [4.7] can flip inside it "
              "(check the deltas against it).")
    else:
        print("  VERDICT: ambiguity at or above a lesion quantum — before reporting, check "
              "every [4.7] delta against the affected rungs' spreads; a comparison smaller "
              "than the joint spread is not platform-stable. Escalate to the user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
