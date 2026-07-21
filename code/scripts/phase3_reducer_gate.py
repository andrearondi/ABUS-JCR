"""[P3U2.4f-2] Reducer gate on seed0 — relax-then-reduce, WITHOUT re-flooding (the load-bearing de-risk).

[P3U.4f] showed no linker-suppression config reaches recall >= 0.90 within the pool budget: `contain_off`
recovers the distinct over-trimmed lesions (110/121/122) at recall 0.933 but pool_MAX 1720 (> budget). This
gate tests the Update-2 fix: RELAX the in-plane containment (recover the TPs) and then apply a MEMBERSHIP-ONLY
3D NMS (`link.nms.reduce_pool_3dnms`, keyed by score_max, coordinates unchanged) to collapse the 3D duplicates
the relaxation re-introduces — WITHOUT touching the depth-collinear low-IoU FP trains the Phase-4 geometry
axis (A) needs. It reuses the deployed seed0's CACHED detections (the [P3U.4b] probe cache; no retrain).

For each (containment, 3D-NMS-IoU) config it reports, at the deployment op:
  - linked recall (n_vol with a SURVIVING tube clearing IOU_HIT_THRESHOLD) — the recoverable ceiling,
  - post-NMS cands/vol MEAN and MAX — the pool cost (MAX vs RESCORER_POOL_BUDGET is the gate),
  - per-target hit flags for the lesions the FROZEN (contain=0.80, nms=off) config misses (auto-detected).
Pick the CHEAPEST config with recall >= --target-recall at pool_MAX <= budget; set LINK_CONTAINMENT_THRESH +
LINK_3DNMS_IOU PROVISIONALLY, re-run the unit tests, then freeze formally at [P3U2.7] on the fold detectors.

Self-checks (printed): (contain=0.80, nms=off) must reproduce the frozen recall 0.833 @ ~388; (contain=off,
nms=off) must reproduce the [P3U.4f] `all_off` recovery (recall 0.933 @ pool_MAX 1720).

Usage (server; CPU-fast if the [P3U.4b] cache is warm, else GPU to re-detect):
    python scripts/phase3_reducer_gate.py \
        --checkpoint /home/maia-user/Andre2/outputs/phase2/checkpoints/retinanet_full_seed0.pt \
        --label p3u_seed0_ep6 --op-thresh 0.03 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.geometry import iou_official
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official
from abus_jcr.link.aggregate import score_stats
from abus_jcr.link.nms import reduce_pool_3dnms
from _phase3_common import (add_phase3_paths, assert_device, cache_root, load_manifest,
                            load_official_gt, gt_official_tuple, load_or_run_detections,
                            filter_by_score)

CONTAINMENTS = [0.80, 0.90, 0.95, 1.0]      # 1.0 == containment OFF
NMS_IOUS = [None, 0.5, 0.4, 0.3, 0.2]       # None == 3D NMS OFF (pre-Update-2)


def _cfg_label(contain: float, nms) -> str:
    c = "off" if contain >= 1.0 else f"{contain:.2f}"
    n = "off" if nms is None else f"{nms:.2f}"
    return f"contain={c},nms={n}"


def _link_eval(det_by_vid, gt_by_vid, meta_by_vid, contain: float, nms, score_floor: float = 0.0):
    """(recall, pool_mean, pool_max, {vid: hit}) at the fixed op, post-(score-floor then 3D-NMS)."""
    hits, pools = {}, []
    for vid, raw in det_by_vid.items():
        tubes = link_tubes(raw, containment_thresh=contain)
        meta = meta_by_vid[vid]; gt = gt_by_vid[vid]
        offs = [iso_tube_to_official(t, meta) for t in tubes]
        scs = [float(score_stats(t)["score_max"]) for t in tubes]
        if score_floor > 0.0:                                # LUNA-style tail trim BEFORE 3D NMS
            keep0 = [i for i, sc in enumerate(scs) if sc >= score_floor]
            offs = [offs[i] for i in keep0]; scs = [scs[i] for i in keep0]
        kept = reduce_pool_3dnms(offs, scs, iou_thr=nms)     # membership-only; None -> all
        pools.append(len(kept))
        hits[vid] = any(iou_official(offs[i], gt) > C.IOU_HIT_THRESHOLD for i in kept)
    n = max(len(det_by_vid), 1)
    pool = np.asarray(pools, dtype=float)
    return (sum(hits.values()) / n, float(pool.mean()), float(pool.max()), hits)


def main() -> int:
    parser = argparse.ArgumentParser(description="[P3U2.4f-2] reducer gate (relax containment + 3D NMS)")
    add_phase3_paths(parser)
    parser.add_argument("--checkpoint", required=True, help="deployed seed0 <run>.pt")
    parser.add_argument("--label", required=True,
                        help="cache label; use the [P3U.4b] label (p3u_seed0_ep6) to REUSE its cache")
    parser.add_argument("--op-thresh", type=float, default=C.DET_SELECT_OP_THRESH,
                        help="operating point to analyse (default DET_SELECT_OP_THRESH; the recall peak)")
    parser.add_argument("--detect-op", type=float, default=0.005,
                        help="detector run threshold (default 0.005 to match the [P3U.4b] cache)")
    parser.add_argument("--score-floor", type=float, default=float(C.PREFILTER_SCORE_FLOOR),
                        help="LUNA-style per-candidate score_max floor applied to ALL configs before 3D NMS "
                             "(default conventions.PREFILTER_SCORE_FLOOR; pick from the candidate-diagnostics floor sweep)")
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="acceptance recall for the recommendation (default 0.90; aim 0.93)")
    parser.add_argument("--budget", type=float, default=float(C.RESCORER_POOL_BUDGET),
                        help=f"max cands/vol for the recommendation (default RESCORER_POOL_BUDGET={C.RESCORER_POOL_BUDGET})")
    parser.add_argument("--no-cache", action="store_true", help="force re-detect (ignore cache)")
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

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model, _cfg = load_checkpoint(ckpt)
    model.to(args.device)
    print(f"# [P3U2.4f-2] reducer gate: {ckpt}")
    print(f"  label={args.label}  op_thresh={args.op_thresh}  hit_iou={C.IOU_HIT_THRESHOLD}  "
          f"budget(RESCORER_POOL_BUDGET)={args.budget:.0f}  target_recall={args.target_recall:.2f}\n")

    tag = f"probe_{args.label}_op{args.detect_op}"                # matches phase3_step0_checkpoint_probe cache
    det_by_vid = {}
    for k, v in enumerate(val_ids, 1):
        det_min = load_or_run_detections(args.out_root, tag, v, model, croot, args.detect_op,
                                         args.device, use_cache=not args.no_cache)
        det_by_vid[v] = filter_by_score(det_min, args.op_thresh)
        print(f"  [detect] vol {v} ({k}/{len(val_ids)}): raw={len(det_by_vid[v])} @op{args.op_thresh}", flush=True)
    del model

    if args.score_floor > 0.0:
        print(f"  (LUNA-style score_max floor = {args.score_floor} applied to all configs before 3D NMS)\n")
    results = {}
    for contain in CONTAINMENTS:
        for nms in NMS_IOUS:
            results[_cfg_label(contain, nms)] = _link_eval(det_by_vid, gt_by_vid, meta_by_vid,
                                                           contain, nms, score_floor=args.score_floor)

    # targets = the lesions the FROZEN (contain=0.80, nms=off) config misses (auto-detected).
    frozen_key = _cfg_label(0.80, None)
    _r0, _pm0, _px0, frozen_hits = results[frozen_key]
    targets = sorted(v for v, h in frozen_hits.items() if not h)

    print(f"# reducer sweep @ op={args.op_thresh} (n_val={len(val_ids)}) — targets (frozen-missed) = {targets}\n")
    tcol = "".join(f"{('v'+str(v)):>7}" for v in targets)
    hdr = f"{'config':>22} {'recall':>7} {'pool_mean':>10} {'pool_max':>9}  {'budget?':>8}{tcol}"
    print(hdr); print("-" * len(hdr))
    for contain in CONTAINMENTS:
        for nms in NMS_IOUS:
            label = _cfg_label(contain, nms)
            rec, pmean, pmax, hits = results[label]
            ok = "OK" if pmax <= args.budget else "OVER"
            tflag = "".join(f"{('Y' if hits[v] else '.'):>7}" for v in targets)
            print(f"{label:>22} {rec:>7.3f} {pmean:>10.1f} {pmax:>9.0f}  {ok:>8}{tflag}", flush=True)

    # self-checks (reproduce Update-1 recorded points).
    sc_frozen = results[frozen_key]
    sc_alloff = results[_cfg_label(1.0, None)]
    print("\nSELF-CHECKS (vs Update-1 [P3U.4c]/[P3U.4f]):")
    print(f"  (contain=0.80, nms=off) -> recall {sc_frozen[0]:.3f} @ pool_mean {sc_frozen[1]:.1f} "
          f"(expect ~0.833 @ ~388)")
    print(f"  (contain=off,  nms=off) -> recall {sc_alloff[0]:.3f} @ pool_MAX {sc_alloff[2]:.0f} "
          f"(expect ~0.933 @ ~1720)")

    # recommendation: cheapest (min pool_mean) config clearing target recall AND pool_MAX <= budget.
    elig = [(label, r) for label, r in results.items()
            if r[0] >= args.target_recall - 1e-9 and r[2] <= args.budget]
    print("\nRECOMMENDATION:")
    if elig:
        best = min(elig, key=lambda kv: kv[1][1])            # smallest pool_mean
        rec, pmean, pmax, _h = best[1]
        print(f"  cheapest config reaching recall >= {args.target_recall:.2f} within budget {args.budget:.0f}: "
              f"'{best[0]}' — recall {rec:.3f}, pool mean {pmean:.1f} / max {pmax:.0f}.")
        print("  SET conventions.LINK_CONTAINMENT_THRESH + LINK_3DNMS_IOU to this config, re-run the unit "
              "tests ([P3U2.0a]), re-verify the selector ([P3U2.4e]), then freeze at [P3U2.7] on the fold detectors.")
    else:
        within = [r for r in results.values() if r[2] <= args.budget]
        best_recall = max((r[0] for r in within), default=float("nan"))
        print(f"  NO config reaches recall >= {args.target_recall:.2f} within budget {args.budget:.0f} "
              f"(best in-budget recall = {best_recall:.3f}).")
        print("  → the recovery costs pool beyond budget, OR the residual misses are not linker-recoverable. "
              "ESCALATE (spec Open esc. #2): accept a ceiling < target and document the residual lesion(s), "
              "OR raise RESCORER_POOL_BUDGET one step, OR reconsider the Phase-4 B1 pre-filter (reopens Inv. 8).")

    out_dir = Path(args.out_root) / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "label": args.label, "op_thresh": args.op_thresh,
               "budget": args.budget, "target_recall": args.target_recall, "targets": targets,
               "containments": CONTAINMENTS, "nms_ious": [("off" if n is None else n) for n in NMS_IOUS],
               "configs": {label: {"recall": r[0], "pool_mean": r[1], "pool_max": r[2],
                                   "target_hits": {str(v): bool(r[3][v]) for v in targets}}
                           for label, r in results.items()}}
    outp = out_dir / f"reducer_gate_{args.label}.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"\njson = {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
