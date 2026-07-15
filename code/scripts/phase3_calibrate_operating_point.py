"""[3.4] Calibrate the candidate-generation operating point on VALIDATION (Inv. 2).

With linking FROZEN, sweep ``op_score_thresh`` over
``{0.5,0.3,0.2,0.1,0.05,0.03,0.02,0.01,0.005}`` and, per threshold, measure the linked
3D recall (fraction of Val GT hit at IoU>0.3 by >=1 candidate) and candidates/volume.
Locate the KNEE (>= 98% of max linked recall) and pick ``LINK_OP_SCORE_THRESH``. If the
knee's candidates/volume exceeds ``CANDIDATE_POOL_BUDGET``, raise the point (or set
``PREFILTER_SCORE_FLOOR``) and RECORD the recall-ceiling cost.

Efficiency: the detector runs ONCE per (seed, volume) at the sweep minimum; higher
thresholds are obtained by filtering (exact for the frozen NMS regime — see
``_phase3_common.filter_by_score``). Recall is averaged over the 3 full-train seeds.

Usage (server, GPU):
    python scripts/phase3_calibrate_operating_point.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr import conventions as C
from _phase3_common import (add_phase3_paths, assert_device, cache_root, checkpoints_dir,
                            load_manifest, load_official_gt, gt_official_tuple,
                            linked_recall, filter_by_score, load_or_run_detections)

SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
KNEE_FRAC = 0.98


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.4] operating-point calibration (Val)")
    add_phase3_paths(parser)
    parser.add_argument("--knee-frac", type=float, default=KNEE_FRAC)
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
            curve.append({"thresh": t, **linked_recall(det_t, gt_by_vid, meta_by_vid)})
            print(f"  [sweep] seed{s} thresh={t}: recall={curve[-1]['recall']:.4f} "
                  f"cands/vol={curve[-1]['cands_per_vol_mean']:.1f}", flush=True)
        seed_curves[s] = curve
        print(f"seed {s} done", flush=True)

    # Aggregate over seeds (mean recall + mean cands/vol per threshold).
    print(f"\n# [3.4] Operating-point sweep (Val, {len(val_ids)} vols, mean over "
          f"{len(C.DET_FULL_SEEDS)} seeds)\n")
    print(f"{'thresh':>7} {'recall':>8} {'rec_std':>8} {'cands/vol':>10} {'pool_std':>9}")
    agg = []
    for i, t in enumerate(SWEEP):
        recs = np.array([seed_curves[s][i]["recall"] for s in C.DET_FULL_SEEDS])
        pools = np.array([seed_curves[s][i]["cands_per_vol_mean"] for s in C.DET_FULL_SEEDS])
        row = {"thresh": t, "recall_mean": float(recs.mean()), "recall_std": float(recs.std(ddof=0)),
               "cands_per_vol_mean": float(pools.mean()), "cands_per_vol_std": float(pools.std(ddof=0))}
        agg.append(row)
        print(f"{t:>7} {row['recall_mean']:>8.4f} {row['recall_std']:>8.4f} "
              f"{row['cands_per_vol_mean']:>10.1f} {row['cands_per_vol_std']:>9.1f}")

    max_recall = max(r["recall_mean"] for r in agg)
    target = args.knee_frac * max_recall
    # Knee = the HIGHEST threshold (smallest pool) whose recall clears the target.
    knee = next((r for r in agg if r["recall_mean"] >= target), agg[-1])
    print(f"\nmax linked recall (mean) = {max_recall:.4f}; knee target "
          f"(>= {args.knee_frac:.0%}) = {target:.4f}")
    print(f"KNEE -> op_score_thresh = {knee['thresh']} "
          f"(recall {knee['recall_mean']:.4f}, cands/vol {knee['cands_per_vol_mean']:.1f})")

    budget_note = "within CANDIDATE_POOL_BUDGET"
    if knee["cands_per_vol_mean"] > C.CANDIDATE_POOL_BUDGET:
        budget_note = (f"EXCEEDS CANDIDATE_POOL_BUDGET ({C.CANDIDATE_POOL_BUDGET}): raise the point "
                       "one step or set PREFILTER_SCORE_FLOOR, and RECORD the recall-ceiling cost")
    print(f"pool budget: {budget_note}")
    print("\nACTION: set conventions.LINK_OP_SCORE_THRESH to the chosen thresh and record it in "
          "RESULTS_PHASE_3 [3.4] (with any pool/recall trade).")

    out_dir = Path(args.out_root) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"sweep": agg, "per_seed": seed_curves, "max_recall": max_recall,
               "knee_frac": args.knee_frac, "chosen_thresh": knee["thresh"],
               "chosen_recall": knee["recall_mean"], "chosen_cands_per_vol": knee["cands_per_vol_mean"],
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
        ax1.axvline(knee["thresh"], ls=":", color="k", alpha=0.6)
        ax1.set_xscale("log"); ax1.set_xlabel("op_score_thresh (log)")
        ax1.set_ylabel("linked 3D recall", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ts, [r["cands_per_vol_mean"] for r in agg], "s-", color="tab:red", label="cands/vol")
        ax2.axhline(C.CANDIDATE_POOL_BUDGET, ls="--", color="tab:red", alpha=0.5)
        ax2.set_ylabel("candidates / volume", color="tab:red")
        ax1.set_title("[3.4] Val operating-point knee")
        fig.tight_layout(); fig.savefig(out_dir / "operating_point.png", dpi=120); plt.close(fig)
        print(f"figure = {out_dir / 'operating_point.png'}")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")
    print(f"json = {out_dir / 'operating_point.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
