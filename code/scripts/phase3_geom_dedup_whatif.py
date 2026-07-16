"""[3.4c] Geometry-keyed dedup what-if (diagnostic; CPU, reads cached detections).

Tests whether the 60x-over-budget candidate pool ([3.4]/[3.4b]: ~9k cands/vol, 99.8%
FP, ~226 near-duplicate tubes per object) can be collapsed to the CANDIDATE_POOL_BUDGET
WITHOUT losing the linked-recall ceiling — the thing the score-based 3D-NMS ([3.4b])
fails at (it keeps high-score FPs over low-score TPs).

Reuses the [3.4] detection cache (no GPU, no re-detect). For each (geometry key x
radius) it collapses each spatial neighbourhood of tubes to ONE representative chosen by
GEOMETRY (tube slice-count / fill-ratio), scores the resulting pool with the OFFICIAL
oracle, and reports candidates/volume, linked recall, CPM, and the recall ceiling.
Baselines: no-dedup, and the score-based centre-distance / IoU-NMS controls (to show
geometry beats score at the same mechanism).

Usage (wherever outputs/phase3/detections_cache/ lives):
    python scripts/phase3_geom_dedup_whatif.py --out-root /home/maia-user/Andre2/outputs/phase3 \
        --data-root /home/maia-user/Andre2/data --seed 0 --op-score-thresh 0.03
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
from abus_jcr.detect import schema as S
from abus_jcr.geometry import iou_official
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official, iso_centre_of_tube
from abus_jcr.link.aggregate import score_stats
from abus_jcr.link.nms import nms_3d
from abus_jcr.link.dedup import dedup_by_centre_distance
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling
from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
from _phase3_common import (add_phase3_paths, cache_root, load_manifest, load_official_gt,
                            gt_official_tuple, det_cache_path, filter_by_score)


def _pred_row(vid, off, score):
    return {"public_id": vid, "coordX": off[0], "coordY": off[1], "coordZ": off[2],
            "x_length": off[3], "y_length": off[4], "z_length": off[5],
            "probability": min(float(score), 0.999999)}


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.4c] geometry-keyed dedup what-if (diagnostic)")
    add_phase3_paths(parser)
    parser.add_argument("--seed", type=int, default=0, help="full-train seed whose cache to read")
    parser.add_argument("--cache-op", type=float, default=0.005,
                        help="operating point the cache was built at in [3.4] (default 0.005)")
    parser.add_argument("--op-score-thresh", type=float, default=0.03,
                        help="operating point to analyse (filters the cached detections)")
    parser.add_argument("--keys", nargs="+", default=["slice_count", "fill_ratio", "score_max"],
                        help="geometry representative keys to sweep (score_max is the control)")
    parser.add_argument("--radii", type=float, nargs="+", default=[5, 10, 15, 20],
                        help="centre-distance suppression radii (iso-vox) to sweep")
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.3],
                        help="score-based IoU-NMS control(s) for comparison")
    parser.add_argument("--budget", type=int, default=getattr(C, "CANDIDATE_POOL_BUDGET", 150),
                        help="candidates/volume target")
    parser.add_argument("--recall-target", type=float, default=0.93,
                        help="minimum acceptable linked recall (near the no-dedup ceiling)")
    args = parser.parse_args()

    manifest = load_manifest(args)
    croot = cache_root(args)
    gt_idx = load_official_gt(args, "val").set_index("public_id")
    val_ids = sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])
    tag = f"full_seed{args.seed}_op{args.cache_op}"
    thr = C.IOU_HIT_THRESHOLD

    # --- pass 1: link once per volume, cache per-tube geometry/centre/official box ---
    vol_data = []
    processed_vids = []
    for vid in val_ids:
        p = det_cache_path(args.out_root, tag, vid)
        if not (p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists()):
            print(f"  (skip vol {vid}: no cached detections at {p})")
            continue
        det = filter_by_score(S.read_detections(p), args.op_score_thresh)
        meta = K.read_meta(croot, vid)
        gt = gt_official_tuple(gt_idx, vid)
        tubes = link_tubes(det)
        centres, offs, smax, slc, fill = [], [], [], [], []
        for tube in tubes:
            st = score_stats(tube)
            centres.append(iso_centre_of_tube(tube))
            offs.append(iso_tube_to_official(tube, meta))
            smax.append(st["score_max"]); slc.append(st["slice_count"]); fill.append(st["fill_ratio"])
        vol_data.append({
            "vid": vid, "gt": gt,
            "centres": np.asarray(centres, float).reshape(-1, 3),
            "offs": [tuple(o) for o in offs],
            "keys": {"slice_count": np.asarray(slc, float),
                     "fill_ratio": np.asarray(fill, float),
                     "score_max": np.asarray(smax, float)},
        })
        processed_vids.append(vid)
    n_vol = len(vol_data)
    if n_vol == 0:
        print("no cached detections found — run [3.4] first (it caches per-volume detections).")
        return 1
    gt_used = gt_idx.loc[processed_vids].reset_index()[GT_COLUMNS]

    def _score(preds):
        if not preds:
            return float("nan"), float("nan")
        res = evaluate_froc(gt_used, pd.DataFrame(preds, columns=PRED_COLUMNS))
        return float(cpm(res)), float(recall_ceiling(res))

    def _eval_keep(keep_fn):
        """keep_fn(vd) -> list of kept indices. Returns (cands/vol, linked recall, CPM, ceiling)."""
        preds, hits, kept = [], 0, []
        for vd in vol_data:
            k = keep_fn(vd)
            kept.append(len(k))
            if any(iou_official(vd["offs"][i], vd["gt"]) > thr for i in k):
                hits += 1
            for i in k:
                preds.append(_pred_row(vd["vid"], vd["offs"][i], vd["keys"]["score_max"][i]))
        c, ceil = _score(preds)
        return float(np.mean(kept)), hits / n_vol, c, ceil

    print(f"# [3.4c] Geometry-keyed dedup what-if (Val, seed {args.seed}, op={args.op_score_thresh}, "
          f"n_vol={n_vol})\n")
    print(f"target: cands/vol <= {args.budget} AND linked recall >= {args.recall_target}\n")
    hdr = f"{'method':<34}{'cands/vol':>10}{'recall':>9}{'CPM':>9}{'ceiling':>9}"
    print(hdr); print("-" * len(hdr))

    rows = {}
    # baseline: keep everything
    base = _eval_keep(lambda vd: list(range(len(vd["offs"]))))
    print(f"{'no dedup (raw pool)':<34}{base[0]:>10.1f}{base[1]:>9.4f}{base[2]:>9.4f}{base[3]:>9.4f}")
    rows["no_dedup"] = base

    # control: score-based IoU-NMS (the [3.4b] approach)
    for iou in args.nms_iou:
        r = _eval_keep(lambda vd, iou=iou: nms_3d(vd["offs"], vd["keys"]["score_max"], iou_thr=iou))
        print(f"{f'score IoU-NMS (iou={iou})':<34}{r[0]:>10.1f}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>9.4f}")
        rows[f"score_iou_nms_{iou}"] = r

    # geometry-keyed centre-distance dedup sweep
    best = None
    print()
    for key in args.keys:
        for radius in args.radii:
            r = _eval_keep(lambda vd, key=key, radius=radius:
                           dedup_by_centre_distance(vd["centres"], vd["keys"][key], radius))
            flag = ""
            if r[0] <= args.budget and r[1] >= args.recall_target:
                flag = "  <- meets target"
                if best is None or r[1] > best[1] or (r[1] == best[1] and r[0] < best[0]):
                    best = (r, key, radius)
            label = f"centre-dist key={key} r={radius:g}"
            print(f"{label:<34}{r[0]:>10.1f}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>9.4f}{flag}")
            rows[f"centredist_{key}_r{radius:g}"] = r

    print()
    if best is not None:
        (r, key, radius) = best
        print(f"VERDICT: geometry dedup MEETS TARGET -> key={key}, radius={radius:g}  "
              f"(cands/vol={r[0]:.1f}, recall={r[1]:.4f}, CPM={r[2]:.4f}, ceiling={r[3]:.4f})")
        print("  -> add this dedup to the aggregation (pre-Inv.4-freeze), re-run [3.4], then [3.5].")
    else:
        print(f"VERDICT: NO (key,radius) meets cands/vol<={args.budget} AND recall>={args.recall_target}.")
        print("  -> pick the best recall/pool trade above and RECORD the recall-ceiling cost, or relax")
        print("     the budget. (score IoU-NMS row shows the score-based fallback.)")

    out_dir = Path(args.out_root) / "dedup_whatif"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "op": args.op_score_thresh, "n_vol": n_vol,
               "budget": args.budget, "recall_target": args.recall_target,
               "rows": {k: {"cands_per_vol": v[0], "recall": v[1], "cpm": v[2], "ceiling": v[3]}
                        for k, v in rows.items()},
               "best": (None if best is None else
                        {"key": best[1], "radius": best[2], "cands_per_vol": best[0][0],
                         "recall": best[0][1], "cpm": best[0][2], "ceiling": best[0][3]})}
    outp = out_dir / f"dedup_whatif_seed{args.seed}_op{args.op_score_thresh}.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"\njson = {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
