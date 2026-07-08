"""[2.x] Exit-check dump: schema detections + overlays + per-slice 2D recall.

Loads a full-train checkpoint, runs ``run_detector_on_volume`` at the permissive
``DET_DIAG_*`` knobs over the chosen Val volumes, writes schema-valid detections,
renders overlay PNGs (detection boxes + mask contour) for several lesion slices,
and prints the **per-slice 2D recall** at ``DET_PER_SLICE_RECALL`` (a GT box is
recalled iff some detection with ``score >= 0.05`` has 2D IoU ``> 0.30``). This is
a diagnostic foreshadowing the 3D recall ceiling — NOT the Phase-3 operating point.

Usage (server):
    python scripts/phase2_dump_detections.py --checkpoint .../retinanet_full_seed0.pt \
        --phase1-out .../outputs/phase1 --out-root .../outputs/phase2 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.detect import schema as S
from abus_jcr.detect.infer import run_detector_on_volume
from abus_jcr.detect.retinanet import load_checkpoint
from abus_jcr.detect.slice_det_dataset import boxes_halfopen_for
from _phase2_common import add_phase2_paths, cache_root, load_manifest, load_slice_boxes


def iou_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D IoU of two half-open boxes ``(x1, y1, x2, y2)``."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def per_slice_recall(det_df, slice_boxes_df, volume_ids, score_thresh, iou_thresh):
    """Fraction of GT boxes (over ``volume_ids``) recalled by some detection on the same slice."""
    hits, total = 0, 0
    for vid in volume_ids:
        gt_v = slice_boxes_df[slice_boxes_df["volume_id"] == vid]
        for z in sorted(gt_v["slice_z"].unique()):
            gts = boxes_halfopen_for(slice_boxes_df, vid, int(z))
            dets = det_df[(det_df["volume_id"] == vid) & (det_df["slice_z"] == z)
                          & (det_df["score"] >= score_thresh)]
            dboxes = dets[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            for g in gts:
                total += 1
                if any(iou_2d(g, d) > iou_thresh for d in dboxes):
                    hits += 1
    return hits, total, (hits / total if total else float("nan"))


def _render_overlays(cache_root_p, vid, det_df, slice_boxes_df, out_dir, n_slices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from abus_jcr.slice_dataset import get_stack

    mask = np.asarray(K.open_mask(cache_root_p, vid))
    gt_v = slice_boxes_df[slice_boxes_df["volume_id"] == vid]
    lesion_zs = sorted(gt_v["slice_z"].unique())[:n_slices]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for z in lesion_zs:
        stack = get_stack(cache_root_p, vid, int(z))
        frame = stack[C.C_CHANNELS // 2]  # centre slice
        msl = np.take(mask, int(z), axis=C.SLICE_AXIS)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(frame, cmap="gray", origin="upper")  # d0 depth vertical, d1 lateral horizontal
        if msl.any():
            ax.contour(msl, levels=[0.5], colors="lime", linewidths=1.0)  # mask contour (Inv. figure rule)
        dets = det_df[(det_df["volume_id"] == vid) & (det_df["slice_z"] == z)]
        for _, r in dets.iterrows():
            ax.add_patch(plt.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1,
                                       fill=False, edgecolor="red", linewidth=0.8))
        ax.set_title(f"vol {vid} z={z}: {len(dets)} det")
        ax.axis("off")
        p = out_dir / f"overlay_vol{vid}_z{int(z):03d}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 detection dump (exit check)")
    add_phase2_paths(parser)
    parser.add_argument("--checkpoint", required=True, help="full-train checkpoint (e.g. seed 0)")
    parser.add_argument("--volumes", type=int, nargs="+", default=None,
                        help="Val volumes for recall (default: all val)")
    parser.add_argument("--overlay-volume", type=int, default=100)
    parser.add_argument("--overlay-slices", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = load_manifest(args)
    sb_val = load_slice_boxes(args, "Validation")
    val_ids = args.volumes or sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])

    model, cfg = load_checkpoint(args.checkpoint)
    model.to(args.device)

    croot = cache_root(args)
    det_dir = Path(args.out_root) / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    all_dets = []
    for vid in val_ids:
        df = run_detector_on_volume(
            model, croot, vid,
            score_thresh=C.DET_DIAG_SCORE_THRESH, nms_thresh=C.DET_DIAG_NMS_THRESH,
            detections_per_img=C.DET_DIAG_DETECTIONS_PER_IMG, device=args.device)
        S.write_detections(df, det_dir / f"detections_val{vid}")
        all_dets.append(df)
    det_all = pd.concat(all_dets, ignore_index=True) if all_dets else S.empty_detections()

    hits, total, recall = per_slice_recall(
        det_all, sb_val, val_ids,
        C.DET_PER_SLICE_RECALL["score_thresh"], C.DET_PER_SLICE_RECALL["iou_thresh"])

    fig_dir = Path(args.out_root) / "figures"
    overlays = _render_overlays(croot, args.overlay_volume, det_all, sb_val, fig_dir, args.overlay_slices)

    print(f"# [2.x] Detection dump (checkpoint={Path(args.checkpoint).name}, best_epoch={cfg.get('best_epoch')})\n")
    print(f"val volumes scored     = {len(val_ids)}")
    print(f"total detections       = {len(det_all)}")
    print(f"per-slice 2D recall    = {hits}/{total} = {recall:.4f} "
          f"(score>={C.DET_PER_SLICE_RECALL['score_thresh']}, IoU>{C.DET_PER_SLICE_RECALL['iou_thresh']})")
    print("  ^ diagnostic foreshadowing the 3D recall ceiling; NOT the Phase-3 operating point.")
    print(f"overlays               = {len(overlays)} PNGs under {fig_dir}")
    for p in overlays:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
