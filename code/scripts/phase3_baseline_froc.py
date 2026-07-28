"""[3.6] Baseline FROC ladder B0 — the Phase-4 floor (Inv. 3, 8, 12).

Per Val seed pool (``detector_of_origin == full_seed{s}``): rank the frozen pool by the
detector's own aggregated score (``score_max``), score it through the official oracle
(``eval/froc.evaluate_froc``), and read CPM (``average_recall``) + the recall ceiling
(``max_recall``, Inv. 8). Report mean+-std over the 3 seeds and a volume-level
``bootstrap_cpm_ci`` per seed. Spot-checks that the pred CSV ``probability`` equals the
record ``score_max`` by row-aligned join.

Usage (server or local with the Val record present):
    python scripts/phase3_baseline_froc.py --out-root /home/maia-user/Andre2/outputs/phase3
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
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling, bootstrap_cpm_ci
from abus_jcr.probe import calibration as CAL
from _phase3_common import add_phase3_paths, load_official_gt


def _headroom(sub, gt_df, det: str) -> dict:
    """Decompose the B0'->ceiling gap into cross-volume calibration vs within-set ranking.

    Scores the SAME frozen pool (Inv. 8) under two synthetic probability assignments through the
    official ``evaluate()`` — see :mod:`abus_jcr.probe.calibration`:
      - ``volume_neutral`` : within-set order kept, ALL cross-volume confidence discarded.
      - ``per_vol_oracle`` : the exact upper bound over per-volume monotone rescalings (uses labels
        => a HEADROOM BOUND, never reportable as a result).
    """
    out = {}
    for name, prob in CAL.assignments(sub).items():
        if name == "score_max":
            continue
        s = sub.copy()
        s["_p"] = np.clip(prob.to_numpy(float), 0.0, 1.0 - 1e-9)
        res = evaluate_froc(gt_df, to_official_pred_csv(s, prob_col="_p"))
        out[name] = {"cpm": cpm(res), "ceiling": recall_ceiling(res)}
    hc = CAL.headroom_curve(sub)
    out["curve"] = {k: v for k, v in hc.items() if k != "per_set"}
    print(f"    headroom[{det}]: volume_neutral CPM={out['volume_neutral']['cpm']:.4f}  "
          f"per_vol_oracle CPM={out['per_vol_oracle']['cpm']:.4f}  "
          f"(sets with hit already rank-1 = {hc['n_sets_free']}/{hc['n_sets']}, "
          f"fp_cost p50={hc['fp_cost_p50']:.1f} p90={hc['fp_cost_p90']:.1f} max={hc['fp_cost_max']:.0f})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.6] baseline FROC (B0) over the Val seed pools")
    add_phase3_paths(parser)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-headroom", action="store_true",
                        help="skip the cross-volume-calibration headroom decomposition")
    args = parser.parse_args()

    rec_base = Path(args.out_root) / "candidates" / "candidates_val"
    pool = read_candidate_record(rec_base)
    gt_df = load_official_gt(args, "val")

    seeds = sorted(pool["detector_of_origin"].unique())
    print(f"# [3.6] Baseline FROC B0 (Val, {len(seeds)} seed pools)\n")
    per_seed = []
    for det in seeds:
        sub = pool[pool["detector_of_origin"] == det].reset_index(drop=True)
        pred = to_official_pred_csv(sub, prob_col="score_max")  # row order preserved
        # spot-check row alignment: probability == record score_max
        assert np.allclose(pred["probability"].to_numpy(), sub["score_max"].to_numpy()), \
            f"{det}: pred.probability != record.score_max (row-alignment broken)"

        res = evaluate_froc(gt_df, pred)
        this_cpm = cpm(res)
        ceiling = recall_ceiling(res)
        ci = bootstrap_cpm_ci(gt_df, pred, n_boot=args.n_boot, seed=0)
        rec = {"detector": det, "cpm": this_cpm, "recall_ceiling": ceiling,
               "cpm_ci_lo": ci["lo"], "cpm_ci_hi": ci["hi"], "n_candidates": int(len(sub))}
        print(f"{det}: CPM={this_cpm:.4f}  ceiling(max_recall)={ceiling:.4f}  "
              f"95% CI=[{ci['lo']:.4f}, {ci['hi']:.4f}]  n_cand={len(sub)}")
        if not args.no_headroom:
            rec["headroom"] = _headroom(sub, gt_df, det)
        per_seed.append(rec)

    cpms = np.array([r["cpm"] for r in per_seed])
    ceils = np.array([r["recall_ceiling"] for r in per_seed])
    print(f"\nCPM  mean+-std over {len(seeds)} seeds = {cpms.mean():.4f} +- {cpms.std(ddof=0):.4f}")
    print(f"ceiling mean+-std               = {ceils.mean():.4f} +- {ceils.std(ddof=0):.4f}")
    print("  ^ the recall ceiling is THE key Inv.-8 number; every Phase-4 curve re-ranks this pool.")

    if not args.no_headroom and all("headroom" in r for r in per_seed):
        vn = np.array([r["headroom"]["volume_neutral"]["cpm"] for r in per_seed])
        po = np.array([r["headroom"]["per_vol_oracle"]["cpm"] for r in per_seed])
        print("\n# HEADROOM DECOMPOSITION — same frozen pool (Inv. 8), three probability assignments\n")
        print(f"  {'assignment':>28} {'CPM mean':>10} {'std':>7}   what it isolates")
        print(f"  {'score_max (B0 baseline)':>28} {cpms.mean():>10.4f} {cpms.std(ddof=0):>7.4f}   "
              f"the deployed ranking")
        print(f"  {'volume_neutral':>28} {vn.mean():>10.4f} {vn.std(ddof=0):>7.4f}   "
              f"within-set order kept, cross-volume confidence DISCARDED")
        print(f"  {'per_vol_oracle (BOUND)':>28} {po.mean():>10.4f} {po.std(ddof=0):>7.4f}   "
              f"best possible per-volume rescaling (uses labels)")
        print(f"  {'recall ceiling (Inv. 8)':>28} {ceils.mean():>10.4f} {ceils.std(ddof=0):>7.4f}   "
              f"unreachable by any re-ranking")
        print(f"\n  cross-volume CALIBRATION headroom  = per_vol_oracle - B0  = {po.mean() - cpms.mean():+.4f}")
        print(f"  residual (within-set RANKING) gap   = ceiling - per_vol_oracle = "
              f"{ceils.mean() - po.mean():+.4f}")
        print(f"  volume_neutral - B0                 = {vn.mean() - cpms.mean():+.4f}  "
              f"({'POSITIVE => the detector cross-volume confidence is HARMFUL' if vn.mean() > cpms.mean() else 'negative => it carries real cross-volume signal'})")
        print("  NOTE the synthetic assignments sit exactly on the official 0.005 threshold grid while "
              "raw\n       score_max is compressed into a narrow band, so part of the gap is score SPREAD.")

    out_dir = Path(args.out_root) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"per_seed": per_seed, "cpm_mean": float(cpms.mean()), "cpm_std": float(cpms.std(ddof=0)),
               "ceiling_mean": float(ceils.mean()), "ceiling_std": float(ceils.std(ddof=0))}
    (out_dir / "baseline_froc.json").write_text(json.dumps(payload, indent=2))
    print(f"json = {out_dir / 'baseline_froc.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
