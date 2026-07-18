"""[3.4'] Calibrate the candidate-generation operating point on VALIDATION (Inv. 2 A2).

With linking FROZEN, sweep ``op_score_thresh`` over
``{0.5,0.3,0.2,0.1,0.05,0.03,0.02,0.01,0.005}`` and, per threshold, measure the linked
3D recall, candidates/volume, AND the linked CPM (official oracle). [P3-UPDATE L2] The
curve is FIRST verified monotone (a superset of detections cannot lower linked recall in a
sound aggregation) — the script FAILS LOUD if not. [P3-UPDATE L5/A2] Then, among thresholds
with linked recall >= ``RECALL_CEILING_FRAC`` * max, pick the one MAXIMISING linked CPM
(ranking-aware), tie-broken toward the smaller pool. If it exceeds ``CANDIDATE_POOL_BUDGET``,
raise the point (or set ``PREFILTER_SCORE_FLOOR``) and RECORD the CPM cost, not just recall.

Efficiency: the detector runs ONCE per (seed, volume) at the sweep minimum; higher
thresholds are obtained by filtering (exact for the frozen NMS regime — see
``_phase3_common.filter_by_score``). Recall/CPM averaged over the 3 full-train seeds.

Usage (server, GPU):
    python scripts/phase3_calibrate_operating_point.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.geometry import iou_official
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official
from abus_jcr.link.aggregate import score_stats
from abus_jcr.eval.froc import evaluate_froc, cpm as _cpm, recall_ceiling as _ceil
from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
from _phase3_common import (add_phase3_paths, assert_device, cache_root, checkpoints_dir,
                            load_manifest, load_official_gt, gt_official_tuple,
                            linked_recall, filter_by_score, load_or_run_detections,
                            monotonicity_violations)

SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]


def _linked_cpm_for(det_by_vid, gt_by_vid, meta_by_vid, gt_used):
    """Linked CPM + ceiling over a seed's volumes (official oracle). Returns (cpm, ceiling)."""
    preds = []
    for vid, det in det_by_vid.items():
        for tube in link_tubes(det):
            off = iso_tube_to_official(tube, meta_by_vid[int(vid)])
            sc = score_stats(tube)["score_max"]
            preds.append({"public_id": int(vid), "coordX": off[0], "coordY": off[1], "coordZ": off[2],
                          "x_length": off[3], "y_length": off[4], "z_length": off[5],
                          "probability": min(float(sc), 0.999999)})
    if not preds:
        return float("nan"), float("nan")
    res = evaluate_froc(gt_used, pd.DataFrame(preds, columns=PRED_COLUMNS))
    return float(_cpm(res)), float(_ceil(res))


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.4'] ranking-aware operating-point calibration (Val)")
    add_phase3_paths(parser)
    parser.add_argument("--ceiling-frac", type=float, default=C.RECALL_CEILING_FRAC)
    parser.add_argument("--allow-nonmonotone", action="store_true",
                        help="do NOT fail on a non-monotone curve (diagnostic escape hatch only)")
    parser.add_argument("--no-cache", action="store_true",
                        help="do not read/write the per-volume detection cache (force recompute)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
    from abus_jcr.detect.retinanet import load_checkpoint

    manifest = load_manifest(args)
    croot = cache_root(args)
    gt_idx = load_official_gt(args, "val").set_index("public_id")
    val_ids = sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])
    meta_by_vid = {v: K.read_meta(croot, v) for v in val_ids}
    gt_by_vid = {v: gt_official_tuple(gt_idx, v) for v in val_ids}
    gt_used = gt_idx.loc[val_ids].reset_index()[GT_COLUMNS]

    op_min = min(SWEEP)
    # Per seed: run the detector ONCE per volume at op_min (cached to disk, so a restart
    # resumes), then sweep every threshold by filtering the cached detections.
    seed_curves = {}
    for s in C.DET_FULL_SEEDS:
        model, _ = load_checkpoint(checkpoints_dir(args) / f"retinanet_full_seed{s}.pt")
        model.to(args.device)
        tag = f"full_seed{s}_op{op_min}"
        det_min = {}
        for k, v in enumerate(val_ids, 1):
            det_min[v] = load_or_run_detections(
                args.out_root, tag, v, model, croot, op_min, args.device,
                use_cache=not args.no_cache)
            print(f"  [detect] seed{s} vol {v} ({k}/{len(val_ids)}): {len(det_min[v])} dets",
                  flush=True)
        del model
        curve = []
        for t in SWEEP:
            det_t = {v: filter_by_score(det_min[v], t) for v in val_ids}
            lr = linked_recall(det_t, gt_by_vid, meta_by_vid)
            cpm_v, ceil_v = _linked_cpm_for(det_t, gt_by_vid, meta_by_vid, gt_used)
            curve.append({"thresh": t, **lr, "cpm": cpm_v, "ceiling": ceil_v})
            print(f"  [sweep] seed{s} thresh={t}: recall={lr['recall']:.4f} "
                  f"cands/vol={lr['cands_per_vol_mean']:.1f} CPM={cpm_v:.4f}", flush=True)
        seed_curves[s] = curve
        print(f"seed {s} done", flush=True)

    # Aggregate over seeds (mean recall + cands/vol + CPM per threshold).
    print(f"\n# [3.4'] Operating-point sweep (Val, {len(val_ids)} vols, mean over "
          f"{len(C.DET_FULL_SEEDS)} seeds)\n")
    print(f"{'thresh':>7} {'recall':>8} {'rec_std':>8} {'cands/vol':>10} {'pool_std':>9} {'CPM':>8}")
    agg = []
    for i, t in enumerate(SWEEP):
        recs = np.array([seed_curves[s][i]["recall"] for s in C.DET_FULL_SEEDS])
        pools = np.array([seed_curves[s][i]["cands_per_vol_mean"] for s in C.DET_FULL_SEEDS])
        cpms = np.array([seed_curves[s][i]["cpm"] for s in C.DET_FULL_SEEDS])
        row = {"thresh": t, "recall_mean": float(recs.mean()), "recall_std": float(recs.std(ddof=0)),
               "cands_per_vol_mean": float(pools.mean()), "cands_per_vol_std": float(pools.std(ddof=0)),
               "cpm_mean": float(np.nanmean(cpms))}
        agg.append(row)
        print(f"{t:>7} {row['recall_mean']:>8.4f} {row['recall_std']:>8.4f} "
              f"{row['cands_per_vol_mean']:>10.1f} {row['cands_per_vol_std']:>9.1f} {row['cpm_mean']:>8.4f}")

    # [P3-UPDATE L2] Monotonicity gate — a superset of detections cannot LOWER linked recall.
    viol = monotonicity_violations([r["thresh"] for r in agg], [r["recall_mean"] for r in agg])
    if viol:
        print("\n*** MONOTONICITY VIOLATION (linked recall dropped as the threshold was lowered) ***")
        for v in viol:
            print(f"    thresh {v['thresh_hi']} -> {v['thresh_lo']}: recall "
                  f"{v['recall_hi']:.4f} -> {v['recall_lo']:.4f}  (drop {v['drop']:.4f})")
        print("This is the fingerprint of an UNSOUND linker (L1 drift caps not effective). STOP —")
        print("do NOT pick an operating point on a broken curve. Investigate link/tubes.py caps.")
        if not args.allow_nonmonotone:
            raise SystemExit(2)

    max_recall = max(r["recall_mean"] for r in agg)
    target = args.ceiling_frac * max_recall
    # [P3-UPDATE L5/A2] ranking-aware: among thresholds with recall >= ceiling_frac*max, pick the
    # one MAXIMISING linked CPM; tie-break toward the smaller pool.
    eligible = [r for r in agg if r["recall_mean"] >= target - 1e-12]
    if not eligible:
        eligible = [max(agg, key=lambda r: r["recall_mean"])]
    chosen = max(eligible, key=lambda r: (r["cpm_mean"], -r["cands_per_vol_mean"]))
    print(f"\nmax linked recall (mean) = {max_recall:.4f}; recall floor "
          f"(>= {args.ceiling_frac:.0%}) = {target:.4f}")
    print(f"RANKING-AWARE OP -> op_score_thresh = {chosen['thresh']} "
          f"(recall {chosen['recall_mean']:.4f}, cands/vol {chosen['cands_per_vol_mean']:.1f}, "
          f"CPM {chosen['cpm_mean']:.4f})")

    budget_note = "within CANDIDATE_POOL_BUDGET"
    if chosen["cands_per_vol_mean"] > C.CANDIDATE_POOL_BUDGET:
        budget_note = (f"EXCEEDS CANDIDATE_POOL_BUDGET ({C.CANDIDATE_POOL_BUDGET}): raise the point one "
                       "step or set PREFILTER_SCORE_FLOOR, and RECORD the CPM cost (not just recall)")
    print(f"pool budget: {budget_note}")
    print("\nACTION: set conventions.LINK_OP_SCORE_THRESH to the chosen thresh and record it in "
          "RESULTS_PHASE_3_UPDATE [3.4'] (recall/CPM/pool trade).")

    out_dir = Path(args.out_root) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"sweep": agg, "per_seed": seed_curves, "max_recall": max_recall,
               "ceiling_frac": args.ceiling_frac, "monotonicity_violations": viol,
               "chosen_thresh": chosen["thresh"], "chosen_recall": chosen["recall_mean"],
               "chosen_cpm": chosen["cpm_mean"], "chosen_cands_per_vol": chosen["cands_per_vol_mean"],
               "budget": C.CANDIDATE_POOL_BUDGET}
    (out_dir / "operating_point.json").write_text(json.dumps(payload, indent=2))

    # recall-vs-threshold + pool figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [r["thresh"] for r in agg]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(ts, [r["recall_mean"] for r in agg], "o-", color="tab:blue", label="linked recall")
        ax1.axhline(target, ls="--", color="tab:blue", alpha=0.5)
        ax1.axvline(chosen["thresh"], ls=":", color="k", alpha=0.6)
        ax1.set_xscale("log"); ax1.set_xlabel("op_score_thresh (log)")
        ax1.set_ylabel("linked 3D recall", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ts, [r["cands_per_vol_mean"] for r in agg], "s-", color="tab:red", label="cands/vol")
        ax2.axhline(C.CANDIDATE_POOL_BUDGET, ls="--", color="tab:red", alpha=0.5)
        ax2.set_ylabel("candidates / volume", color="tab:red")
        ax1.set_title("[3.4'] Val ranking-aware operating point")
        fig.tight_layout(); fig.savefig(out_dir / "operating_point.png", dpi=120); plt.close(fig)
        print(f"figure = {out_dir / 'operating_point.png'}")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")
    print(f"json = {out_dir / 'operating_point.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
