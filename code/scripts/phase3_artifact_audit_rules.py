"""Artifact audit, part 2: metric-native interventions + arrangement-radius sensitivity. READ-ONLY.

Reads the ``signatures_<split>.csv`` written by ``phase3_artifact_audit.py`` and asks two things
the cross-entropy reader of Q3 cannot answer cleanly:

**Q3b — what would perfect mechanism knowledge buy on the metric?** For each mechanism class
and each binary flag, every candidate in that class has its detector score demoted to the bottom
of the ranking (score x 1e-3, order preserved) and the pool is re-scored by the vendored official
evaluator. This is a rule with no parameters: the change in CPM is the ceiling any rescorer could
reach by suppressing that class outright, and the number of TPs demoted is its price.

**Q2b — do the arrangement nulls depend on the pair radii?** The 3-D NMS that reduced the pool
leaves an exclusion zone around every candidate (a same-in-plane tube closer than about a tube
length along the sweep was suppressed), so ``sweep_chain`` at 5 mm can sit BELOW its null by
construction. The chain and stack thresholds are swept beyond the tube length.

No verdict. Numbers only; the reading is written by hand.

Usage:
    export ABUS_AXIS_PROFILE=measured
    PYTHONPATH=. python scripts/phase3_artifact_audit_rules.py --split val \
        --out-root /Volumes/Evo/outputs_iso/phase3 --data-root /Volumes/Evo/data
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.candidates.record import to_official_pred_csv
from abus_jcr.eval.froc import evaluate_froc, cpm as froc_cpm, key_recall
from abus_jcr.probe import artifact_audit as AA
from _phase3_common import add_phase3_paths, load_official_gt, profile_banner

DEMOTE = 1e-3


def _score(g: pd.DataFrame, gt: pd.DataFrame, prob: np.ndarray) -> dict:
    gg = g.copy(); gg["_p"] = np.clip(prob, 0.0, 1.0 - 1e-6)
    pred = to_official_pred_csv(gg, "_p")
    with contextlib.redirect_stderr(io.StringIO()):
        res = evaluate_froc(gt[gt["public_id"].isin(gg["public_id"].unique())], pred)
    return {"cpm": froc_cpm(res), "r_at": {str(k): float(v) for k, v in key_recall(res).items()}}


def interventions(sig: pd.DataFrame, gt: pd.DataFrame, per_detector: bool) -> list:
    groups = sig.groupby("detector_of_origin") if per_detector else [("ALL", sig)]
    rows = []
    for det, g in groups:
        g = g.reset_index(drop=True)
        base = g["score_max"].to_numpy(float)
        b0 = _score(g, gt, base)
        rows.append({"detector": det, "rule": "B0 (no rule)", "n_fp": 0, "n_tp": 0, "n_ign": 0, **b0, "delta": 0.0})
        rules = [(f"demote {m}", g["mechanism"] == m) for m in AA.MECHANISMS if (g["mechanism"] == m).any()]
        rules += [("demote narrow+broad column", g["mechanism"].isin(["narrow_column", "broad_column"])),
                  ("demote dark_above", g["dark_above"].astype(bool)),
                  ("demote reaches_floor", g["reaches_floor"].astype(bool)),
                  ("demote shadow_run>=6mm", g["shadow_run_mm"] >= 6.0),
                  ("demote inside_contrast>=0 (not dark)", g["inside_contrast"] >= 0.0)]
        for name, mask in rules:
            m = mask.to_numpy(bool)
            prob = base.copy(); prob[m] = prob[m] * DEMOTE
            r = _score(g, gt, prob)
            rows.append({"detector": det, "rule": name,
                         "n_fp": int((m & (g["label"] == "neg")).sum()), "n_tp": int((m & (g["label"] == "pos")).sum()),
                         "n_ign": int((m & (g["label"] == "ignore")).sum()), **r, "delta": r["cpm"] - b0["cpm"]})
    return rows


def print_interventions(rows: list):
    print("\n# Q3b — RULE INTERVENTIONS: demote every candidate of a class to the bottom, re-score with the OFFICIAL evaluator")
    print("  (delta = CPM - B0; n_fp / n_tp / n_ign = candidates demoted by label; a class worth suppressing has delta > 0 at n_tp ~ 0)")
    print(f"  {'detector':>12} {'rule':>38} {'n_fp':>5} {'n_tp':>5} {'n_ign':>5} {'CPM':>7} {'delta':>7} | R@0.125 R@0.25 R@0.5 R@1 R@2")
    for r in rows:
        ra = r["r_at"]
        print(f"  {r['detector']:>12} {r['rule']:>38} {r['n_fp']:>5} {r['n_tp']:>5} {r['n_ign']:>5} {r['cpm']:7.4f} {r['delta']:+7.4f} | "
              + " ".join(f"{ra[str(k)]:.3f}" for k in (0.125, 0.25, 0.5, 1, 2)))


def sensitivity(sig: pd.DataFrame, n_perm: int, seed: int) -> list:
    rows = []
    fp = sig[sig["label"] == "neg"]
    for chain_mm, stack_mm in ((5.0, 5.0), (10.0, 8.0), (15.0, 12.0)):
        knobs = dataclasses.replace(AA.Knobs(), chain_mm=chain_mm, stack_depth_mm=stack_mm)
        rng = np.random.default_rng(seed)
        for det, g in fp.groupby("detector_of_origin"):
            zc, zs, ec, es = [], [], [], []
            for vid, gv in g.groupby("public_id"):
                if len(gv) < 5:
                    continue
                p = gv[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float) * knobs.iso_mm
                r = AA.arrangement_null(p, knobs, rng, n_perm=n_perm)
                zc.append(r["same_column"]["z"]); ec.append(r["same_column"]["exceeds"])
                zs.append(r["sweep_chain"]["z"]); es.append(r["sweep_chain"]["exceeds"])
            rows.append({"detector": det, "chain_mm": chain_mm, "stack_mm": stack_mm, "n_vol": len(zc),
                         "column_z_med": float(np.nanmedian(zc)) if zc else np.nan, "column_frac_exceeds": float(np.mean(ec)) if ec else np.nan,
                         "chain_z_med": float(np.nanmedian(zs)) if zs else np.nan, "chain_frac_exceeds": float(np.mean(es)) if es else np.nan})
    return rows


def print_sensitivity(rows: list):
    print("\n# Q2b — ARRANGEMENT RADIUS SENSITIVITY (same_column: |d depth| > stack; sweep_chain: |d sweep| > chain; nulls as before)")
    print(f"  {'detector':>12} {'chain':>6} {'stack':>6} {'n_vol':>5} | {'column z':>9} {'frac>p95':>9} | {'chain z':>9} {'frac>p95':>9}")
    for r in rows:
        print(f"  {r['detector']:>12} {r['chain_mm']:>6g} {r['stack_mm']:>6g} {r['n_vol']:>5} | {r['column_z_med']:9.2f} {r['column_frac_exceeds']:9.2f} | "
              f"{r['chain_z_med']:9.2f} {r['chain_frac_exceeds']:9.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Artifact audit part 2: interventions + radius sensitivity (read-only)")
    add_phase3_paths(ap)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    profile_banner()
    out_dir = Path(args.out_root) / "artifact_audit"
    sig = pd.read_csv(out_dir / f"signatures_{args.split}.csv")
    print(f"\n# ARTIFACT AUDIT — RULES — split={args.split} candidates={len(sig)} from {out_dir / f'signatures_{args.split}.csv'}")
    gt = load_official_gt(args, args.split)
    rows = interventions(sig, gt, per_detector=(args.split == "val"))
    print_interventions(rows)
    sens = sensitivity(sig, args.n_perm, args.seed)
    print_sensitivity(sens)
    (out_dir / f"artifact_audit_rules_{args.split}.json").write_text(json.dumps({"interventions": rows, "sensitivity": sens}, indent=1, default=float))
    print(f"# json -> {out_dir / f'artifact_audit_rules_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
