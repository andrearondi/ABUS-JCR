"""[3.x] Phase-3 cost — linking + candidate-generation latency per volume.

Times, per Val volume (one full-train detector): detector inference, tube linking, and
tube->box aggregation, plus the resulting pool size. Persisted for the Phase-5 cost
table. Detector inference dominates; linking/aggregation is the Phase-3-added overhead.

Usage (server, GPU):
    python scripts/phase3_cost.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official, iso_centre_of_tube, iso_extents_of_tube
from abus_jcr.link.aggregate import score_stats
from _phase3_common import (add_phase3_paths, assert_device, cache_root, checkpoints_dir,
                            load_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.x] Phase-3 linking/candidate cost")
    add_phase3_paths(parser)
    parser.add_argument("--seed", type=int, default=0, help="full-train seed checkpoint to time")
    parser.add_argument("--volumes", type=int, nargs="+", default=None,
                        help="Val volumes to time (default: all val)")
    parser.add_argument("--op-score-thresh", type=float, default=C.LINK_OP_SCORE_THRESH)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
    from abus_jcr.detect.retinanet import load_checkpoint
    from abus_jcr.detect.infer import run_detector_on_volume

    manifest = load_manifest(args)
    croot = cache_root(args)
    val_ids = args.volumes or sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])
    model, _ = load_checkpoint(checkpoints_dir(args) / f"retinanet_full_seed{args.seed}.pt")
    model.to(args.device)

    per_vol = []
    for vid in val_ids:
        t0 = time.perf_counter()
        det_df = run_detector_on_volume(
            model, croot, vid, score_thresh=args.op_score_thresh, nms_thresh=C.LINK_NMS_THRESH,
            detections_per_img=C.LINK_DETECTIONS_PER_IMG, device=args.device)
        t1 = time.perf_counter()
        tubes = link_tubes(det_df)
        t2 = time.perf_counter()
        meta = K.read_meta(croot, vid)
        for tube in tubes:
            iso_tube_to_official(tube, meta); iso_centre_of_tube(tube)
            iso_extents_of_tube(tube); score_stats(tube)
        t3 = time.perf_counter()
        per_vol.append({"volume_id": vid, "n_det": int(len(det_df)), "n_tubes": len(tubes),
                        "detect_s": t1 - t0, "link_s": t2 - t1, "aggregate_s": t3 - t2})
        print(f"  vol {vid}: {len(det_df)} dets, {len(tubes)} tubes "
              f"(detect {t1-t0:.2f}s, link {t2-t1:.2f}s, agg {t3-t2:.2f}s)", flush=True)

    def col(k):
        return np.array([r[k] for r in per_vol], dtype=float)

    print(f"# [3.x] Phase-3 cost (seed {args.seed}, {len(val_ids)} Val vols, op={args.op_score_thresh})\n")
    print(f"{'metric':<16}{'mean':>10}{'median':>10}{'max':>10}")
    for k, label in [("detect_s", "detector s/vol"), ("link_s", "linking s/vol"),
                     ("aggregate_s", "aggregate s/vol"), ("n_det", "raw dets/vol"),
                     ("n_tubes", "candidates/vol")]:
        c = col(k)
        print(f"{label:<16}{c.mean():>10.3f}{np.median(c):>10.3f}{c.max():>10.3f}")

    out_dir = Path(args.out_root) / "cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "op_score_thresh": args.op_score_thresh,
               "per_volume": per_vol,
               "detect_s_mean": float(col("detect_s").mean()),
               "link_s_mean": float(col("link_s").mean()),
               "aggregate_s_mean": float(col("aggregate_s").mean()),
               "candidates_per_vol_mean": float(col("n_tubes").mean())}
    (out_dir / "phase3_cost.json").write_text(json.dumps(summary, indent=2))
    print(f"\njson = {out_dir / 'phase3_cost.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
