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
from _phase3_common import (add_phase3_paths, assert_device, cache_root, checkpoints_dir,
                            load_manifest, load_official_gt, gt_official_tuple, linked_recall)


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.3] freeze linking params (Train, OOF)")
    add_phase3_paths(parser)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1],
                        help="Train folds to validate over (OOF detectors); default 0 1")
    parser.add_argument("--op-score-thresh", type=float, default=C.LINK_OP_SCORE_THRESH,
                        help="provisional operating point for the param check")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
    from abus_jcr.detect.retinanet import load_checkpoint
    from abus_jcr.detect.infer import run_detector_on_volume

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
        for vid in vids:
            det_by_vid[vid] = run_detector_on_volume(
                model, croot, vid, score_thresh=args.op_score_thresh,
                nms_thresh=C.LINK_NMS_THRESH, detections_per_img=C.LINK_DETECTIONS_PER_IMG,
                device=args.device)
            gt_by_vid[vid] = gt_official_tuple(gt_idx, vid)
            meta_by_vid[vid] = K.read_meta(croot, vid)
        del model

    base = dict(link_iou=C.LINK_IOU, max_z_gap=C.LINK_MAX_Z_GAP, min_tube_len=C.LINK_MIN_TUBE_LEN)
    sweeps = {
        "link_iou": [round(C.LINK_IOU - 0.1, 3), C.LINK_IOU, round(C.LINK_IOU + 0.1, 3)],
        "max_z_gap": [max(0, C.LINK_MAX_Z_GAP - 1), C.LINK_MAX_Z_GAP, C.LINK_MAX_Z_GAP + 1],
        "min_tube_len": [max(1, C.LINK_MIN_TUBE_LEN - 1), C.LINK_MIN_TUBE_LEN, C.LINK_MIN_TUBE_LEN + 1],
    }

    print(f"# [3.3] Linking-param validation (folds={args.folds}, n_vol={len(det_by_vid)}, "
          f"op={args.op_score_thresh})\n")
    ref = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **base)
    print(f"DEFAULTS {base}: recall={ref['recall']:.4f} "
          f"cands/vol mean={ref['cands_per_vol_mean']:.1f} median={ref['cands_per_vol_median']:.1f}\n")

    results = {"defaults": {**base, **ref}, "sweeps": {}}
    for param, values in sweeps.items():
        print(f"-- sweep {param} (others at default) --")
        results["sweeps"][param] = []
        for v in values:
            kw = dict(base); kw[param] = v
            r = linked_recall(det_by_vid, gt_by_vid, meta_by_vid, **kw)
            tag = "  <- default" if v == base[param] else ""
            print(f"    {param}={v:<5} recall={r['recall']:.4f} "
                  f"cands/vol={r['cands_per_vol_mean']:.1f}{tag}")
            results["sweeps"][param].append({"value": v, **r})
        print()

    out_dir = Path(args.out_root) / "linking"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "freeze_linking.json").write_text(json.dumps(results, indent=2))
    print("FROZEN (Inv. 4; log into RESULTS_PHASE_3 [3.3] and reuse for Phase 6):")
    print(f"  LINK_IOU={C.LINK_IOU} LINK_MAX_Z_GAP={C.LINK_MAX_Z_GAP} "
          f"LINK_MIN_TUBE_LEN={C.LINK_MIN_TUBE_LEN}")
    print(f"  LINK_NMS_THRESH={C.LINK_NMS_THRESH} LINK_DETECTIONS_PER_IMG={C.LINK_DETECTIONS_PER_IMG}")
    print(f"json = {out_dir / 'freeze_linking.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
