"""[3.3] Light Train-split validation + FREEZE of the linking params (Inv. 4).

At a provisional operating point, over a handful of Train folds (OOF detectors), sweep
each of ``(LINK_IOU, LINK_MAX_Z_GAP, LINK_MIN_TUBE_LEN)`` +/- one step and report
linked-3D recall + candidates/volume. The defaults are FROZEN (Inv. 4) — this confirms
they sit on a FLAT region (small recall/pool sensitivity), it does NOT tune per detector.
The detector is run ONCE per volume; every param setting re-links the cached detections.

**LOG the confirmed values into RESULTS_PHASE_3.md [3.3]** — Phase 6 reuses them unmodified.

Usage (server, GPU):
    python scripts/phase3_freeze_linking.py --folds 0 1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from abus_jcr import cache as K
from abus_jcr import conventions as C
from _phase2_common import load_slice_boxes
from _phase3_common import (add_phase3_paths, assert_device, cache_root, checkpoints_dir,
                            load_manifest, load_official_gt, gt_official_tuple, linked_recall,
                            load_or_run_detections, derive_link_caps)


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.3] freeze linking params (Train, OOF)")
    add_phase3_paths(parser)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1],
                        help="Train folds to validate over (OOF detectors); default 0 1")
    parser.add_argument("--op-score-thresh", type=float, default=C.LINK_OP_SCORE_THRESH,
                        help="provisional operating point for the param check")
    parser.add_argument("--no-cache", action="store_true",
                        help="do not read/write the per-volume detection cache (force recompute)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
    from abus_jcr.detect.retinanet import load_checkpoint

    manifest = load_manifest(args)
    croot = cache_root(args)
    gt_idx = load_official_gt(args, "train").set_index("public_id")

    # Run each OOF detector once per its fold's volumes; cache detections + gt + meta.
    det_by_vid, gt_by_vid, meta_by_vid = {}, {}, {}
    for f in args.folds:
        ckpt = checkpoints_dir(args) / f"retinanet_fold{f}.pt"
        model, _ = load_checkpoint(ckpt)
        model.to(args.device)
        vids = sorted(int(v) for v in manifest[(manifest["split"] == "train")
                                               & (manifest["fold"] == f)]["volume_id"])
        tag = f"fold{f}_op{args.op_score_thresh}"
        for k, vid in enumerate(vids, 1):
            det_by_vid[vid] = load_or_run_detections(
                args.out_root, tag, vid, model, croot, args.op_score_thresh, args.device,
                use_cache=not args.no_cache)
            gt_by_vid[vid] = gt_official_tuple(gt_idx, vid)
            meta_by_vid[vid] = K.read_meta(croot, vid)
            print(f"  [detect] fold{f} vol {vid} ({k}/{len(vids)}): {len(det_by_vid[vid])} dets",
                  flush=True)
        del model

    # [P3-UPDATE L1] Derive the Train-GT drift caps FIRST (they bound every link below).
    caps = derive_link_caps(load_slice_boxes(args, "Train"))
    print("# [3.3'] Train-GT drift-cap derivation (P3-UPDATE L1; SET these in conventions.py, re-run [3.0])")
    print(f"  zspan_p99={caps['zspan_p99']:.1f} -> LINK_MAX_TUBE_ZSPAN = {caps['LINK_MAX_TUBE_ZSPAN']} "
          f"(x{caps['zspan_safety']})")
    print(f"  inplane_extent_p99={caps['inplane_extent_p99']:.1f} -> "
          f"LINK_MAX_CENTROID_DRIFT = {caps['LINK_MAX_CENTROID_DRIFT']} (x{caps['drift_safety']})\n")

    # Use the just-derived caps for this validation run even if conventions still holds None,
    # so the sweep reflects the bounded linker that will actually be frozen.
    cap_kw = dict(max_tube_zspan=caps["LINK_MAX_TUBE_ZSPAN"],
                  max_centroid_drift=caps["LINK_MAX_CENTROID_DRIFT"],
                  containment_thresh=C.LINK_CONTAINMENT_THRESH)

    base = dict(link_iou=C.LINK_IOU, max_z_gap=C.LINK_MAX_Z_GAP, min_tube_len=C.LINK_MIN_TUBE_LEN)
    # [P3-UPDATE L3 / P3U2] widen min_tube_len {2..12}: [P3U2.diag] §6 showed TP slice_count p10=18 vs FP
    # p90=13, so a higher min-len trims short FP tubes at (near-)zero recall cost. Pick the largest recall-neutral.
    sweeps = {
        "link_iou": [round(C.LINK_IOU - 0.1, 3), C.LINK_IOU, round(C.LINK_IOU + 0.1, 3)],
        "max_z_gap": [max(0, C.LINK_MAX_Z_GAP - 1), C.LINK_MAX_Z_GAP, C.LINK_MAX_Z_GAP + 1],
        "min_tube_len": [2, 3, 4, 5, 6, 8, 10, 12],
    }

    print(f"# [3.3'] Linking-param validation (folds={args.folds}, n_vol={len(det_by_vid)}, "
          f"op={args.op_score_thresh}; caps + containment {C.LINK_CONTAINMENT_THRESH} ON)\n")
    ref = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **base, **cap_kw)
    print(f"DEFAULTS {base}: recall={ref['recall']:.4f} "
          f"cands/vol mean={ref['cands_per_vol_mean']:.1f} median={ref['cands_per_vol_median']:.1f}\n",
          flush=True)

    results = {"defaults": {**base, **ref}, "caps": caps, "sweeps": {}, "ablations": {}}
    for param, values in sweeps.items():
        print(f"-- sweep {param} (others at default) --")
        results["sweeps"][param] = []
        for v in values:
            kw = dict(base); kw[param] = v
            r = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **kw, **cap_kw)
            tag = "  <- default" if v == base[param] else ""
            print(f"    {param}={v:<5} recall={r['recall']:.4f} "
                  f"cands/vol={r['cands_per_vol_mean']:.1f}{tag}", flush=True)
            results["sweeps"][param].append({"value": v, **r})
        print()

    # [P3-UPDATE L1/L4] ablations: caps OFF vs ON, containment OFF vs ON (pool + recall effect).
    print("-- ablation: drift caps + containment (recall should hold; pool should DROP) --")
    variants = {
        "caps OFF, containment OFF": dict(max_tube_zspan=None, max_centroid_drift=None, containment_thresh=1.0),
        "caps ON,  containment OFF": dict(**{**cap_kw, "containment_thresh": 1.0}),
        "caps OFF, containment ON ": dict(max_tube_zspan=None, max_centroid_drift=None,
                                          containment_thresh=C.LINK_CONTAINMENT_THRESH),
        "caps ON,  containment ON ": dict(**cap_kw),
    }
    for name, kw in variants.items():
        r = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **base, **kw)
        print(f"    {name}: recall={r['recall']:.4f} cands/vol={r['cands_per_vol_mean']:.1f}", flush=True)
        results["ablations"][name.strip()] = r
    print()

    # [P3U2 3.C] 3D-NMS freeze sweep on the fold detectors — confirm the seed0-chosen LINK_3DNMS_IOU is
    # recall-neutral here (recall holds; pool drops) before the formal freeze. Runs at the current
    # (possibly relaxed) LINK_CONTAINMENT_THRESH so it reflects the deployed relax-then-reduce config.
    print("-- [P3U2] 3D-NMS sweep (recall should hold; cands/vol should DROP toward RESCORER_POOL_BUDGET) --")
    results["nms_3d_sweep"] = []
    for nms in [None, 0.5, 0.4, 0.3, 0.2]:
        r = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **base, **cap_kw, nms_iou=nms)
        tag = "  <- current LINK_3DNMS_IOU" if nms == C.LINK_3DNMS_IOU else ""
        print(f"    nms_iou={('off' if nms is None else nms):<4} recall={r['recall']:.4f} "
              f"cands/vol mean={r['cands_per_vol_mean']:.1f} (budget {C.RESCORER_POOL_BUDGET}){tag}", flush=True)
        results["nms_3d_sweep"].append({"nms_iou": ("off" if nms is None else nms), **r})
    print()

    # [P3U2] PREFILTER_SCORE_FLOOR sweep on the FOLD detectors — the score floor is frozen on train (Inv. 4),
    # but it was PICKED on seed0-val ([P3U2.diag]). Fold detectors (80-vol, weaker) may have lower TP scores,
    # so a floor tuned to seed0 can over-cut them. This sweep shows fold recall vs floor: FREEZE the LARGEST
    # floor that stays recall-neutral ON THE FOLDS (not just on seed0). Runs at the deployed containment/caps.
    print("-- [P3U2] score-floor sweep on the FOLD detectors (recall must hold on folds; pool should DROP) --")
    results["score_floor_sweep"] = []
    for floor in [0.0, 0.04, 0.05, 0.06, 0.08, 0.10]:
        r = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **base, **cap_kw, score_floor=floor)
        tag = "  <- current PREFILTER_SCORE_FLOOR" if floor == C.PREFILTER_SCORE_FLOOR else ""
        print(f"    floor={floor:<5} recall={r['recall']:.4f} "
              f"cands/vol mean={r['cands_per_vol_mean']:.1f} (budget {C.RESCORER_POOL_BUDGET}){tag}", flush=True)
        results["score_floor_sweep"].append({"score_floor": floor, **r})
    print()

    # [P3-UPDATE L4] LINK_NMS_THRESH sweep note: nms is applied INSIDE the detector, so a true sweep
    # needs re-detection. Re-run this script with a conventions.LINK_NMS_THRESH edit (0.5/0.6/0.7) and
    # --no-cache to compare; the cache tag encodes op only, so change the tag or pass --no-cache.

    out_dir = Path(args.out_root) / "linking"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "freeze_linking.json").write_text(json.dumps(results, indent=2))
    print("FREEZE (Inv. 4; SET in conventions.py, log into RESULTS_PHASE_3_UPDATE [3.3'], reuse Phase 6):")
    print(f"  LINK_IOU={C.LINK_IOU} LINK_MAX_Z_GAP={C.LINK_MAX_Z_GAP} "
          f"LINK_MIN_TUBE_LEN=<largest recall-neutral from the sweep above>")
    print(f"  LINK_MAX_TUBE_ZSPAN={caps['LINK_MAX_TUBE_ZSPAN']} "
          f"LINK_MAX_CENTROID_DRIFT={caps['LINK_MAX_CENTROID_DRIFT']} "
          f"LINK_CONTAINMENT_THRESH={C.LINK_CONTAINMENT_THRESH}")
    print(f"  LINK_NMS_THRESH=<recall-neutral min from the nms re-runs> "
          f"LINK_DETECTIONS_PER_IMG={C.LINK_DETECTIONS_PER_IMG}")
    print(f"json = {out_dir / 'freeze_linking.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
