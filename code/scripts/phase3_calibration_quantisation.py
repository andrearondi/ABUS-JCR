"""[P3U2.CAL-Q] How much of the calibration headroom is threshold QUANTISATION, not miscalibration?

``phase3_baseline_froc.py`` reports ``per_vol_oracle - B0'`` as cross-volume calibration headroom,
and :mod:`abus_jcr.probe.calibration` already carries the caveat that part of it is score SPREAD:
the synthetic assignments sit exactly on the official ``np.arange(0, 1, 0.005)`` threshold grid while
raw ``score_max`` is compressed into a narrow band and shares grid cells. That caveat has never been
measured, so the headroom figure is an upper bound of unknown tightness — and a thesis sentence
resting on it ("the rescorer's value is cross-volume calibration") deserves better.

This script measures the term. It scores the SAME frozen pool (Inv. 8 — re-ranking only) under a
**global monotone rescaling** of ``score_max``: the pooled ordering is preserved exactly, ties stay
ties, and the only change is that each of the seven key FP operating points lands on its own grid
cell. Such a map carries no information the baseline lacks, so

    quantisation      = CPM(global_monotone) - B0'
    calibration       = CPM(per_vol_oracle)  - B0'          (as reported at [3.6])
    calibration NET   = CPM(per_vol_oracle)  - CPM(global_monotone)

and the last line is the part of the gap that no monotone transform of the detector's own score can
reach — i.e. genuine cross-volume miscalibration, which is what the Phase-4 ``lambda * BCE`` term is
argued to attack.

READ-ONLY and CPU-only: it reads the frozen candidate record and the official GT, nothing else. No
cache, no checkpoint, no detector. It re-derives B0', ``volume_neutral`` and ``per_vol_oracle`` in
the same pass **on purpose** — those three must reproduce the recorded [3.6] headroom block, and
that reproduction is the guard that this run read the intended pool.

**It is a diagnostic, never a result and never a selection surface.** ``per_vol_oracle`` uses labels;
``global_monotone`` uses the label-derived key-FP prefixes to place its cuts. Neither is a model, a
deployable ranking, or an input to any decision.

Usage:
    python scripts/phase3_calibration_quantisation.py \
        --out-root /home/maia-user/Andre2/outputs_iso/phase3 \
        --data-root /home/maia-user/Andre2/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.candidates.record import read_candidate_record, to_official_pred_csv
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling
from abus_jcr.probe import calibration as CAL
from _phase3_common import add_phase3_paths, load_official_gt, profile_banner

ROWS = ("score_max", "volume_neutral", "volume_neutral_anchored", "per_vol_oracle", "global_monotone")
RECORDED = ("score_max", "volume_neutral", "per_vol_oracle")   # must reproduce the [3.6] block


def _cpm_of(gt_df: pd.DataFrame, sub: pd.DataFrame, prob) -> tuple:
    s = sub.copy()
    s["_p"] = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0 - 1e-9)
    res = evaluate_froc(gt_df, to_official_pred_csv(s, prob_col="_p"))
    return cpm(res), recall_ceiling(res)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="[P3U2.CAL-Q] split the calibration headroom into quantisation vs miscalibration")
    add_phase3_paths(parser)
    parser.add_argument("--split", default="val", choices=("train", "val"),
                        help="which frozen record to read; default val (the reported substrate)")
    args = parser.parse_args()
    profile_banner()          # Inv. 6: name the substrate that produced this output

    pool = read_candidate_record(Path(args.out_root) / "candidates" / f"candidates_{args.split}")
    gt_df = load_official_gt(args, args.split)
    n_vol = int(gt_df["public_id"].nunique())      # the evaluator's own FP denominator

    print(f"# [P3U2.CAL-Q] calibration headroom: quantisation vs miscalibration "
          f"({args.split}, {n_vol} GT volumes)\n")
    print(f"  key FP rates = {C.KEY_FP}   grid = {CAL.GRID} ({CAL.N_LEVELS} usable levels)")
    print("  global_monotone preserves the POOLED score_max ordering exactly (ties kept as ties);")
    print("  it only moves the seven key-FP operating points onto distinct grid cells.\n")

    per_seed = []
    for det in sorted(pool["detector_of_origin"].unique()):
        sub = pool[pool["detector_of_origin"] == det].reset_index(drop=True)
        # `volume_neutral_anchored` now comes from `assignments` itself (anchored=True, the
        # label-free form). The 2026-08-27 run used anchored="auto", which resolves to the same
        # thing on this pool — no set has a hitting rank-1 candidate in every volume — so the
        # recorded [I3.11] numbers reproduce unchanged.
        probs = CAL.assignments(sub, extra={
            "global_monotone": CAL.global_monotone_probability(sub, n_vol=n_vol)})
        rec = {"detector": det, "n_candidates": int(len(sub))}
        for name in ROWS:
            c, ceil = _cpm_of(gt_df, sub, probs[name].to_numpy(float))
            rec[name] = {"cpm": c, "ceiling": ceil}
        rec["cuts"] = CAL.global_monotone_cuts(sub, n_vol=n_vol)
        print(f"{det}: " + "  ".join(f"{n}={rec[n]['cpm']:.4f}" for n in ROWS))
        kept = [c for c in rec["cuts"] if c["kept"]]
        print(f"    global-monotone operating points kept ({len(kept)} of {len(rec['cuts'])} "
              f"candidate cuts, FP per vol / lesions):")
        print("      " + "  ".join(f"{c['fp_per_vol']:.3f}/{c['hits']}" for c in kept))
        # Inv. 8: re-ranking cannot move the ceiling. If it does, the pool changed under us.
        ceils = {rec[n]["ceiling"] for n in ROWS}
        assert max(ceils) - min(ceils) < 1e-9, f"{det}: ceiling moved across assignments: {ceils}"
        per_seed.append(rec)

    m = {n: np.array([r[n]["cpm"] for r in per_seed]) for n in ROWS}
    ceil = np.array([r["score_max"]["ceiling"] for r in per_seed])
    print(f"\n# MEAN OVER {len(per_seed)} POOLS\n")
    print(f"  {'assignment':>28} {'CPM mean':>10} {'std':>7}   what it isolates")
    what = {"score_max": "the deployed ranking (B0')",
            "volume_neutral": "within-set order kept, cross-volume confidence DISCARDED",
            "volume_neutral_anchored": "  ^ same, read the way a REAL probability column is (see below)",
            "per_vol_oracle": "best per-volume rescaling (uses labels)",
            "global_monotone": "SAME pooled order, key-FP points on distinct grid cells"}
    for n in ROWS:
        print(f"  {n:>28} {m[n].mean():>10.4f} {m[n].std(ddof=0):>7.4f}   {what[n]}")
    print(f"  {'recall ceiling (Inv. 8)':>28} {ceil.mean():>10.4f} {ceil.std(ddof=0):>7.4f}   "
          f"unreachable by any re-ranking")

    quant = m["global_monotone"].mean() - m["score_max"].mean()
    calib = m["per_vol_oracle"].mean() - m["score_max"].mean()
    net = m["per_vol_oracle"].mean() - m["global_monotone"].mean()
    print(f"\n  QUANTISATION term       = global_monotone - B0'      = {quant:+.4f}")
    print(f"  calibration headroom    = per_vol_oracle  - B0'      = {calib:+.4f}   (as reported at [3.6])")
    print(f"  calibration NET of grid = per_vol_oracle  - global_m = {net:+.4f}")
    if abs(calib) > 1e-9:
        print(f"  => {100.0 * quant / calib:.1f} % of the reported calibration headroom is threshold "
              f"quantisation / score spread;\n     {100.0 * net / calib:.1f} % is out of reach of any "
              f"monotone transform of score_max.")
    if quant < 0:
        print("  NOTE quantisation came out NEGATIVE: the baseline's own coarse curve is being "
              "interpolated\n       across a gap in a way that flatters it. Report this as measured; "
              "do not discard it.")
    vn_gap = m["volume_neutral_anchored"].mean() - m["volume_neutral"].mean()
    if abs(vn_gap) > 1e-9:
        print(f"\n  ! volume_neutral, read the way a real probability column is: "
              f"{m['volume_neutral_anchored'].mean():+.4f} "
              f"({vn_gap:+.4f} vs the recorded reading)")
        print("    `volume_neutral` occupies the TOP grid cell, so its swept curve has no empty-set "
              "point, and\n    `_interpolate_recall_at_fp` returns 0 for every key rate below its "
              "cheapest operating point —\n    which costs one FP per set whose rank-1 candidate is "
              "not a hit. A real prediction column never\n    saturates 0.995, so it always has that "
              "anchor, so the unanchored reading is a FLOOR, not the\n    value of the rule.")
        print("    RESOLVED 2026-08-29: the anchored reading is canonical and is reported by "
              "`phase3_baseline_froc`\n    as the `B0-rank` BASELINE with a paired bootstrap against "
              "B0'. The unanchored row is kept as the\n    superseded floor. `volume_neutral_probability`"
              "'s own default is still False so the primitive stays\n    a primitive; `assignments()` "
              "is what carries the reported convention.")

    print("\n  The rows " + ", ".join(RECORDED) + " MUST reproduce the recorded [3.6] headroom block "
          "for this\n  substrate. A mismatch means this run read a different pool — stop and find "
          "out why before reading\n  anything else.")
    print("  global_monotone is a LOWER bound on what a global monotone map achieves: its cuts land "
          "at integer\n  FP totals bracketing each key rate, and the official interpolator then "
          "reads the chord between them.")

    out_dir = Path(args.out_root) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "n_vol": n_vol, "key_fp": list(C.KEY_FP), "per_seed": per_seed,
               "mean": {n: float(m[n].mean()) for n in ROWS},
               "std": {n: float(m[n].std(ddof=0)) for n in ROWS},
               "quantisation_term": float(quant), "calibration_headroom": float(calib),
               "calibration_net_of_grid": float(net)}
    path = out_dir / f"calibration_quantisation_{args.split}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\njson = {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
