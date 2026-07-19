"""[P3U.4f] Linker-knob ablation on seed0 — recover the 0.833 ceiling WITHOUT re-flooding (diagnostic).

[P3U.4d] proved the 5 missed val lesions are LINKER-side, not a detector wall: 3 are over-trimmed by
suppression (containment/drift caps), 1 (vol 116) fragments on sparse z-coverage, 1 (vol 124) is the
marginal case. The naive fix — turn suppression OFF (the [P3U.4d] `perm` setting) — recovers them but
risks re-introducing the very FP-duplicate flood (9,023 cands/vol, 226 dup/object) this cycle added
containment (L4) + drift caps (L1) to kill. That trade was NEVER measured: [P3U.4d] reported no pool cost.

This step measures it. On the deployed seed0's CACHED detections (reuses the [P3U.4b] op=0.005 cache; no
retrain), it re-links every Val volume under each candidate linker config and reports, all at the
deployment op (--op-thresh 0.03):
  - linked recall (n_vol with a tube clearing IOU_HIT_THRESHOLD) — the recoverable ceiling,
  - cands/vol MEAN and MAX — the pool cost (MAX vs CANDIDATE_POOL_BUDGET is the budget gate),
  - per-target hit flags for the lesions the FROZEN config misses (auto-detected — no hardcoding),
so you can pick the CHEAPEST config that reaches recall >= --target-recall at pool <= budget, set it
PROVISIONALLY, re-run [P3U.4c], then freeze it formally at [P3U.7] on the fold detectors.

Axes swept (one-at-a-time ablation + a few bundles):
  suppression : drift caps {frozen 182/342, 2x wider, OFF}; containment {0.80 frozen, 0.90, 0.95, OFF}
  continuity  : max_z_gap {1 frozen, 2, 3}; link_iou {0.30 frozen, 0.20}   (for the fragmented vol 116)

Usage (server, GPU only if the cache is cold; else CPU-fast):
    python scripts/phase3_linker_ablation.py \
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
from _phase3_common import (add_phase3_paths, assert_device, cache_root, load_manifest,
                            load_official_gt, gt_official_tuple, load_or_run_detections,
                            filter_by_score)


def _configs():
    """(label, link_tubes kwargs override) — frozen baseline first, then one-axis ablations + bundles."""
    z = C.LINK_MAX_Z_GAP
    caps_off = {"max_tube_zspan": None, "max_centroid_drift": None}
    caps_2x = {"max_tube_zspan": 2 * C.LINK_MAX_TUBE_ZSPAN, "max_centroid_drift": 2 * C.LINK_MAX_CENTROID_DRIFT}
    return [
        ("frozen", {}),
        # --- suppression axis ---
        ("caps_2x", dict(caps_2x)),
        ("caps_off", dict(caps_off)),
        ("contain_0.90", {"containment_thresh": 0.90}),
        ("contain_0.95", {"containment_thresh": 0.95}),
        ("contain_off", {"containment_thresh": 1.0}),
        # --- continuity axis (targets the fragmented vol 116) ---
        ("zgap_2", {"max_z_gap": z + 1}),
        ("zgap_3", {"max_z_gap": z + 2}),
        ("link_iou_0.20", {"link_iou": 0.20}),
        # --- bundles ---
        ("caps_off+contain_0.90", {**caps_off, "containment_thresh": 0.90}),
        ("continuity(zgap_3+iou0.20)", {"max_z_gap": z + 2, "link_iou": 0.20}),
        ("recovery(caps_off+cont0.90+zgap_3+iou0.20)",
         {**caps_off, "containment_thresh": 0.90, "max_z_gap": z + 2, "link_iou": 0.20}),
        ("all_off (=[P3U.4d] perm)", {**caps_off, "containment_thresh": 1.0}),
    ]


def _link_eval(det_by_vid, gt_by_vid, meta_by_vid, override):
    """Per-config: (recall, pool_mean, pool_max, {vid: hit}, {vid: best_iou}) at the fixed op."""
    hits, best_iou, pools = {}, {}, []
    for vid, raw in det_by_vid.items():
        tubes = link_tubes(raw, **override)
        pools.append(len(tubes))
        gt = gt_by_vid[vid]; meta = meta_by_vid[vid]
        bi = 0.0
        for tube in tubes:
            bi = max(bi, float(iou_official(iso_tube_to_official(tube, meta), gt)))
        best_iou[vid] = bi
        hits[vid] = bi > C.IOU_HIT_THRESHOLD
    n = max(len(det_by_vid), 1)
    pool = np.asarray(pools, dtype=float)
    return (sum(hits.values()) / n, float(pool.mean()), float(pool.max()), hits, best_iou)


def main() -> int:
    parser = argparse.ArgumentParser(description="[P3U.4f] linker-knob ablation (recall vs pool cost)")
    add_phase3_paths(parser)
    parser.add_argument("--checkpoint", required=True, help="deployed seed0 <run>.pt")
    parser.add_argument("--label", required=True,
                        help="cache label; use the [P3U.4b] label (p3u_seed0_ep6) to REUSE its cache")
    parser.add_argument("--op-thresh", type=float, default=C.DET_SELECT_OP_THRESH,
                        help="operating point to analyse (default DET_SELECT_OP_THRESH; the recall peak)")
    parser.add_argument("--detect-op", type=float, default=0.005,
                        help="detector run threshold (default 0.005 to match the [P3U.4b] cache)")
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="acceptance recall for the recommendation (default 0.90)")
    parser.add_argument("--budget", type=float, default=float(C.CANDIDATE_POOL_BUDGET),
                        help=f"max cands/vol for the recommendation (default CANDIDATE_POOL_BUDGET={C.CANDIDATE_POOL_BUDGET})")
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
    print(f"# [P3U.4f] linker-knob ablation: {ckpt}")
    print(f"  label={args.label}  op_thresh={args.op_thresh}  hit_iou={C.IOU_HIT_THRESHOLD}  "
          f"budget={args.budget:.0f}  frozen(caps={C.LINK_MAX_TUBE_ZSPAN}/{C.LINK_MAX_CENTROID_DRIFT}, "
          f"contain={C.LINK_CONTAINMENT_THRESH}, zgap={C.LINK_MAX_Z_GAP}, link_iou={C.LINK_IOU}, "
          f"min_len={C.LINK_MIN_TUBE_LEN})\n")

    tag = f"probe_{args.label}_op{args.detect_op}"                # matches phase3_step0_checkpoint_probe cache
    det_by_vid = {}
    for k, v in enumerate(val_ids, 1):
        det_min = load_or_run_detections(args.out_root, tag, v, model, croot, args.detect_op,
                                         args.device, use_cache=not args.no_cache)
        det_by_vid[v] = filter_by_score(det_min, args.op_thresh)
        print(f"  [detect] vol {v} ({k}/{len(val_ids)}): raw={len(det_by_vid[v])} @op{args.op_thresh}", flush=True)
    del model

    configs = _configs()
    results = {}
    for label, override in configs:
        results[label] = _link_eval(det_by_vid, gt_by_vid, meta_by_vid, override)

    # targets = the lesions the FROZEN config misses (auto-detected, no hardcoding).
    _rec0, _pm0, _px0, frozen_hits, _bi0 = results["frozen"]
    targets = sorted(v for v, h in frozen_hits.items() if not h)

    print(f"# ablation @ op={args.op_thresh} (n_val={len(val_ids)}) — targets (frozen-missed) = {targets}\n")
    tcol = "".join(f"{('v'+str(v)):>7}" for v in targets)
    hdr = f"{'config':>42} {'recall':>7} {'pool_mean':>10} {'pool_max':>9}  {'budget?':>8}{tcol}"
    print(hdr); print("-" * len(hdr))
    for label, _ov in configs:
        rec, pmean, pmax, hits, bi = results[label]
        ok = "OK" if pmax <= args.budget else "OVER"
        tflag = "".join(f"{('Y' if hits[v] else '.'):>7}" for v in targets)
        print(f"{label:>42} {rec:>7.3f} {pmean:>10.1f} {pmax:>9.0f}  {ok:>8}{tflag}", flush=True)

    # recommendation: cheapest (min pool_mean) config that clears target recall AND the per-vol budget.
    elig = [(label, results[label]) for label, _ in configs
            if results[label][0] >= args.target_recall - 1e-9 and results[label][2] <= args.budget]
    print("\nRECOMMENDATION:")
    if elig:
        best = min(elig, key=lambda kv: kv[1][1])            # smallest pool_mean
        rec, pmean, pmax, _h, _b = best[1]
        print(f"  cheapest config reaching recall >= {args.target_recall:.2f} within budget {args.budget:.0f}: "
              f"'{best[0]}' — recall {rec:.3f}, pool mean {pmean:.1f} / max {pmax:.0f}.")
        print("  SET the matching conventions.LINK_* to this config, re-run [P3U.0a], then re-run [P3U.4c] to "
              "confirm the ceiling recovers on the full sweep. Freeze formally at [P3U.7] on the fold detectors.")
    else:
        within = [(label, results[label]) for label, _ in configs if results[label][2] <= args.budget]
        best_recall = max((r[1][0] for r in within), default=float("nan"))
        print(f"  NO config reaches recall >= {args.target_recall:.2f} within budget {args.budget:.0f} "
              f"(best in-budget recall = {best_recall:.3f}).")
        print("  → the recovery costs pool beyond budget, OR the residual misses (e.g. vol 124, weak coverage) "
              "are not linker-recoverable. Re-check per-target flags above; consider raising --budget one step "
              "(Inv. amendment, escalate) or accepting a ceiling < target and documenting the residual lesions.")

    out_dir = Path(args.out_root) / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "label": args.label, "op_thresh": args.op_thresh,
               "budget": args.budget, "target_recall": args.target_recall, "targets": targets,
               "frozen": {"caps": [C.LINK_MAX_TUBE_ZSPAN, C.LINK_MAX_CENTROID_DRIFT],
                          "containment": C.LINK_CONTAINMENT_THRESH, "max_z_gap": C.LINK_MAX_Z_GAP,
                          "link_iou": C.LINK_IOU, "min_tube_len": C.LINK_MIN_TUBE_LEN},
               "configs": {label: {"recall": r[0], "pool_mean": r[1], "pool_max": r[2],
                                   "target_hits": {str(v): bool(r[3][v]) for v in targets},
                                   "target_best_iou": {str(v): r[4][v] for v in targets}}
                           for label, r in results.items()}}
    outp = out_dir / f"linker_ablation_{args.label}.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"\njson = {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
