"""[Step-0] Operating-point + ranking probe for ONE arbitrary checkpoint (diagnostic; GPU).

Motivation: [3.4c] showed the candidate pool cannot be deduped to budget without losing
recall — the fix must be upstream (the detector's score can't threshold to a small,
well-ranked pool). Before committing to a detector RETRAIN, this cheaply tests whether a
different, ranking-selected checkpoint (e.g. the archived AP-selected seed0, epoch 9)
already thresholds to a SMALLER and/or BETTER-RANKED pool than the deployed CPM-proxy
checkpoint (epoch 4) — WITHOUT any retraining.

Runs the given checkpoint through the IDENTICAL frozen Phase-3 path as [3.4]/[3.4b]
(same per-slice NMS + cap via ``load_or_run_detections``, same linking, same official
oracle), then reports per operating threshold: linked 3D recall, candidates/volume, and
the baseline CPM + recall ceiling. Directly comparable to the deployed-seed0 columns in
RESULTS_PHASE_3 [3.4] (recall/pool) and [3.4b] (CPM 0.2095 @ 0.03).

Usage (server, GPU):
    python scripts/phase3_step0_checkpoint_probe.py \
        --checkpoint /home/maia-user/Andre2/outputs/_archive/phase2_ap_selection_seed0/retinanet_full_seed0.pt \
        --label ap_seed0 --device cuda
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
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling
from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
from _phase3_common import (add_phase3_paths, assert_device, cache_root, load_manifest,
                            load_official_gt, gt_official_tuple, load_or_run_detections,
                            filter_by_score)

SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
KNEE_FRAC = 0.98


def _eval_threshold(det_by_vid, gt_by_vid, meta_by_vid, gt_used):
    """Link each volume's detections, then return (recall, cands/vol, CPM, ceiling)."""
    n_hit, pools, preds = 0, [], []
    for vid, det in det_by_vid.items():
        tubes = link_tubes(det)                    # frozen linking params (conventions defaults)
        pools.append(len(tubes))
        gt = gt_by_vid[int(vid)]
        meta = meta_by_vid[int(vid)]
        hit = False
        for tube in tubes:
            off = iso_tube_to_official(tube, meta)
            sc = score_stats(tube)["score_max"]
            preds.append({"public_id": int(vid), "coordX": off[0], "coordY": off[1], "coordZ": off[2],
                          "x_length": off[3], "y_length": off[4], "z_length": off[5],
                          "probability": min(float(sc), 0.999999)})
            if iou_official(off, gt) > C.IOU_HIT_THRESHOLD:
                hit = True
        n_hit += int(hit)
    recall = n_hit / max(len(det_by_vid), 1)
    pool_mean = float(np.mean(pools)) if pools else float("nan")
    if preds:
        res = evaluate_froc(gt_used, pd.DataFrame(preds, columns=PRED_COLUMNS))
        cpm_v, ceil_v = float(cpm(res)), float(recall_ceiling(res))
    else:
        cpm_v, ceil_v = float("nan"), float("nan")
    return recall, pool_mean, cpm_v, ceil_v


def main() -> int:
    parser = argparse.ArgumentParser(description="[Step-0] single-checkpoint operating-point + ranking probe")
    add_phase3_paths(parser)
    parser.add_argument("--checkpoint", required=True, help="path to the .pt checkpoint to probe")
    parser.add_argument("--label", required=True, help="short label for the cache tag + output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-cache", action="store_true", help="force re-detect (ignore cache)")
    parser.add_argument("--op-min", type=float, default=min(SWEEP),
                        help="detector run threshold; higher sweep points are obtained by filtering")
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

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model, cfg = load_checkpoint(ckpt)
    model.to(args.device)
    print(f"# [Step-0] checkpoint probe: {ckpt}")
    print(f"  label={args.label}  best_epoch={cfg.get('best_epoch')}  "
          f"selection_metric={cfg.get('selection_metric')}  "
          f"best_val_ap={cfg.get('best_val_ap')}  best_val_cpm_proxy={cfg.get('best_val_cpm_proxy')}\n")

    tag = f"probe_{args.label}_op{args.op_min}"
    det_min = {}
    for k, v in enumerate(val_ids, 1):
        det_min[v] = load_or_run_detections(args.out_root, tag, v, model, croot, args.op_min,
                                            args.device, use_cache=not args.no_cache)
        print(f"  [detect] {args.label} vol {v} ({k}/{len(val_ids)}): {len(det_min[v])} dets", flush=True)
    del model

    print(f"\n# [Step-0] Val sweep (n_vol={len(val_ids)}) — compare to deployed seed0 in [3.4]/[3.4b]\n")
    hdr = f"{'thresh':>7} {'recall':>8} {'cands/vol':>10} {'CPM':>8} {'ceiling':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for t in SWEEP:
        det_t = {v: filter_by_score(det_min[v], t) for v in val_ids}
        rec, pool, cpm_v, ceil_v = _eval_threshold(det_t, gt_by_vid, meta_by_vid, gt_used)
        rows.append({"thresh": t, "recall": rec, "cands_per_vol": pool, "cpm": cpm_v, "ceiling": ceil_v})
        print(f"{t:>7} {rec:>8.4f} {pool:>10.1f} {cpm_v:>8.4f} {ceil_v:>8.4f}", flush=True)

    max_rec = max(r["recall"] for r in rows)
    target = KNEE_FRAC * max_rec
    knee = next((r for r in rows if r["recall"] >= target), rows[-1])
    print(f"\nmax linked recall = {max_rec:.4f}; knee (>= {KNEE_FRAC:.0%} = {target:.4f}) -> "
          f"thresh={knee['thresh']}  cands/vol={knee['cands_per_vol']:.1f}  CPM={knee['cpm']:.4f}")
    print("READ: vs deployed seed0 (epoch 4) — knee was 0.03 @ ~10723 cands/vol, CPM 0.2095. A smaller")
    print("cands/vol at similar recall AND/OR a higher CPM here means ranking-aware SELECTION alone helps")
    print("(lighter fix); if it looks the same, the recipe — matcher/negatives/training — must change.")

    out_dir = Path(args.out_root) / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "label": args.label, "cfg_best_epoch": cfg.get("best_epoch"),
               "selection_metric": cfg.get("selection_metric"), "sweep": rows,
               "knee": knee, "max_recall": max_rec}
    outp = out_dir / f"step0_{args.label}.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"\njson = {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
