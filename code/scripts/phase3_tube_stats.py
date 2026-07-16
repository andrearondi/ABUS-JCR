"""[3.4b] Tube statistics + 3D-NMS what-if (diagnostic; CPU, reads cached detections).

Explains the candidate-pool explosion and quantifies the fix WITHOUT re-running the
detector — it reads the per-volume detections cached by [3.4] at the sweep minimum and
re-links at a chosen operating point. Reports, over the Val volumes of one seed:

  * per-tube distributions: slice_count, z_span, fill_ratio, score_max/mean, official
    box diagonal, depth-anisotropy (ext_d0 / mean(ext_d1, ext_d2));
  * REDUNDANCY: single-linkage clusters of tube centres per volume -> tubes-per-cluster
    (how many near-duplicate tubes sit on the same object);
  * a 3D-NMS WHAT-IF: for each iou_thr, candidates/volume kept and linked 3D recall —
    so the pool/recall trade of adding a dedup step is measured before committing.

Usage (server or local, wherever outputs/phase3/detections_cache/ lives):
    python scripts/phase3_tube_stats.py --out-root /home/maia-user/Andre2/outputs/phase3 \
        --data-root /home/maia-user/Andre2/data --seed 0 --op-score-thresh 0.03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.detect import schema as S
from abus_jcr.geometry import iou_official
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official, iso_centre_of_tube, iso_extents_of_tube
from abus_jcr.link.aggregate import score_stats
from abus_jcr.link.nms import nms_3d
from abus_jcr.probe.fp_structure import _single_linkage_clusters
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling
from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
import pandas as pd
from _phase3_common import (add_phase3_paths, cache_root, load_manifest, load_official_gt,
                            gt_official_tuple, det_cache_path, filter_by_score)


def _pct(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.4b] tube stats + 3D-NMS what-if (diagnostic)")
    add_phase3_paths(parser)
    parser.add_argument("--seed", type=int, default=0, help="full-train seed whose cache to read")
    parser.add_argument("--cache-op", type=float, default=0.005,
                        help="operating point the cache was built at in [3.4] (default 0.005)")
    parser.add_argument("--op-score-thresh", type=float, default=0.03,
                        help="operating point to analyse (filters the cached detections)")
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                        help="3D-NMS iou thresholds for the what-if")
    parser.add_argument("--cluster-radius", type=float, default=C.FP_PROBE_CLUSTER_RADIUS,
                        help="iso-voxel radius for the redundancy clustering")
    args = parser.parse_args()

    manifest = load_manifest(args)
    croot = cache_root(args)
    gt_idx = load_official_gt(args, "val").set_index("public_id")
    val_ids = sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])
    tag = f"full_seed{args.seed}_op{args.cache_op}"

    # per-tube accumulators
    slice_count, z_span, fill_ratio, score_max, score_mean, box_diag, aniso = ([] for _ in range(7))
    tubes_per_cluster = []          # redundancy: tube count per spatial cluster (per volume)
    n_tubes_per_vol, n_tp_tubes_per_vol = [], []
    # 3D-NMS what-if accumulators: iou_thr -> {"kept_per_vol": [], "hit": 0}
    whatif = {t: {"kept": [], "hits": 0} for t in args.nms_iou}
    # prediction rows per condition, to score ACTUAL CPM via the official oracle
    pred_none = []                                     # no NMS (raw pool)
    pred_nms = {t: [] for t in args.nms_iou}
    processed_vids = []
    n_vol = 0

    for vid in val_ids:
        p = det_cache_path(args.out_root, tag, vid)
        if not (p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists()):
            print(f"  (skip vol {vid}: no cached detections at {p})")
            continue
        n_vol += 1
        processed_vids.append(vid)
        det = filter_by_score(S.read_detections(p), args.op_score_thresh)
        meta = K.read_meta(croot, vid)
        gt = gt_official_tuple(gt_idx, vid)
        tubes = link_tubes(det)

        offs, smax, centres = [], [], []
        n_tp = 0
        for tube in tubes:
            st = score_stats(tube)
            off = iso_tube_to_official(tube, meta)
            ext = iso_extents_of_tube(tube)
            slice_count.append(st["slice_count"]); z_span.append(st["z_span"])
            fill_ratio.append(st["fill_ratio"]); score_max.append(st["score_max"])
            score_mean.append(st["score_mean"])
            box_diag.append(float(np.sqrt(off[3] ** 2 + off[4] ** 2 + off[5] ** 2)))
            denom = (ext[1] + ext[2]) / 2.0
            aniso.append(ext[0] / denom if denom > 0 else np.nan)
            offs.append(off); smax.append(st["score_max"])
            centres.append(iso_centre_of_tube(tube))
            if iou_official(off, gt) > C.IOU_HIT_THRESHOLD:
                n_tp += 1
        n_tubes_per_vol.append(len(tubes)); n_tp_tubes_per_vol.append(n_tp)

        # redundancy: how many tubes fall in each spatial cluster
        centres = np.asarray(centres, float) if centres else np.zeros((0, 3))
        if len(centres):
            # union-find cluster ids via the probe helper (returns count); to get sizes we
            # recompute a simple grid: number of clusters -> mean tubes/cluster proxy
            n_clusters = _single_linkage_clusters(centres, args.cluster_radius)
            tubes_per_cluster.append(len(centres) / max(n_clusters, 1))

        # raw-pool predictions (for the no-NMS CPM baseline)
        for off, sc in zip(offs, smax):
            pred_none.append({"public_id": vid, "coordX": off[0], "coordY": off[1], "coordZ": off[2],
                              "x_length": off[3], "y_length": off[4], "z_length": off[5],
                              "probability": min(float(sc), 0.999999)})
        # 3D-NMS what-if (pool size, linked recall, and kept predictions for CPM)
        for t in args.nms_iou:
            keep = nms_3d(offs, smax, iou_thr=t)
            whatif[t]["kept"].append(len(keep))
            if any(iou_official(offs[k], gt) > C.IOU_HIT_THRESHOLD for k in keep):
                whatif[t]["hits"] += 1
            for k in keep:
                pred_nms[t].append({"public_id": vid, "coordX": offs[k][0], "coordY": offs[k][1],
                                    "coordZ": offs[k][2], "x_length": offs[k][3], "y_length": offs[k][4],
                                    "z_length": offs[k][5], "probability": min(float(smax[k]), 0.999999)})

    print(f"# [3.4b] Tube stats (Val, seed {args.seed}, op={args.op_score_thresh}, n_vol={n_vol})\n")
    print(f"candidates/volume: mean={np.mean(n_tubes_per_vol):.1f} median={np.median(n_tubes_per_vol):.0f} "
          f"max={np.max(n_tubes_per_vol):.0f}")
    print(f"TP tubes/volume (hit GT IoU>0.3): mean={np.mean(n_tp_tubes_per_vol):.1f} "
          f"-> FP fraction ~{1 - np.sum(n_tp_tubes_per_vol)/max(np.sum(n_tubes_per_vol),1):.4f}")
    print(f"REDUNDANCY (tubes per spatial cluster, radius {args.cluster_radius} iso-vox): "
          f"mean={np.mean(tubes_per_cluster):.1f} median={np.median(tubes_per_cluster):.1f} "
          f"max={np.max(tubes_per_cluster):.1f}")
    print("  ^ >>1 means many near-duplicate parallel tubes sit on the same object (the pool driver).\n")

    print("per-tube distributions (min / p25 / median / p75 / p95 / max):")
    for name, arr in [("slice_count", slice_count), ("z_span", z_span), ("fill_ratio", fill_ratio),
                      ("score_max", score_max), ("score_mean", score_mean),
                      ("box_diag(offc)", box_diag), ("anisotropy_d0", aniso)]:
        s = _pct(arr)
        if s.get("n"):
            print(f"  {name:<16} {s['min']:8.3f} {s['p25']:8.3f} {s['median']:8.3f} "
                  f"{s['p75']:8.3f} {s['p95']:8.3f} {s['max']:8.3f}")

    # ACTUAL CPM via the official oracle, per condition (this is the deciding number:
    # TP-region redundancy is free, but FP-region redundancy inflates FP/vol -> lowers CPM).
    gt_used = gt_idx.loc[processed_vids].reset_index()[GT_COLUMNS]

    def _cpm_ceiling(rows):
        if not rows:
            return float("nan"), float("nan")
        res = evaluate_froc(gt_used, pd.DataFrame(rows, columns=PRED_COLUMNS))
        return cpm(res), recall_ceiling(res)

    print(f"\n3D-NMS WHAT-IF (dedup; keep highest score_max per cluster) — CPM via official oracle:")
    print(f"{'iou_thr':>8} {'cands/vol':>10} {'linked recall':>14} {'CPM':>8} {'ceiling':>8}")
    cpm_none, ceil_none = _cpm_ceiling(pred_none)
    print(f"{'(none)':>8} {np.mean(n_tubes_per_vol):>10.1f} "
          f"{np.sum([1 for h in n_tp_tubes_per_vol if h>0])/max(n_vol,1):>14.4f} "
          f"{cpm_none:>8.4f} {ceil_none:>8.4f}")
    whatif_out = {"none": {"cands_per_vol": float(np.mean(n_tubes_per_vol)),
                           "cpm": float(cpm_none), "ceiling": float(ceil_none)}}
    for t in args.nms_iou:
        kept = np.mean(whatif[t]["kept"]); rec = whatif[t]["hits"] / max(n_vol, 1)
        c, ceil = _cpm_ceiling(pred_nms[t])
        whatif_out[str(t)] = {"cands_per_vol": float(kept), "recall": float(rec),
                              "cpm": float(c), "ceiling": float(ceil)}
        print(f"{t:>8} {kept:>10.1f} {rec:>14.4f} {c:>8.4f} {ceil:>8.4f}")
    print("  ^ recall/ceiling should hold; CPM should RISE as FP-region duplicates are removed.")

    out_dir = Path(args.out_root) / "tube_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "op": args.op_score_thresh, "n_vol": n_vol,
               "cands_per_vol_mean": float(np.mean(n_tubes_per_vol)),
               "tubes_per_cluster_mean": float(np.mean(tubes_per_cluster)),
               "dist": {name: _pct(arr) for name, arr in
                        [("slice_count", slice_count), ("z_span", z_span), ("fill_ratio", fill_ratio),
                         ("score_max", score_max), ("score_mean", score_mean),
                         ("box_diag", box_diag), ("anisotropy_d0", aniso)]},
               "cpm_none": float(cpm_none), "nms_whatif": whatif_out}
    (out_dir / f"tube_stats_seed{args.seed}_op{args.op_score_thresh}.json").write_text(json.dumps(payload, indent=2))
    print(f"\njson = {out_dir / f'tube_stats_seed{args.seed}_op{args.op_score_thresh}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
