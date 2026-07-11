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
from abus_jcr.detect import diagnostics as DG
from abus_jcr.detect import schema as S
from abus_jcr.detect.infer import run_detector_on_volume
from abus_jcr.detect.retinanet import load_checkpoint
from _phase2_common import add_phase2_paths, assert_device, cache_root, load_manifest, load_slice_boxes


def _rect(ax, box, color, lw, zorder):
    import matplotlib.pyplot as plt

    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               fill=False, edgecolor=color, linewidth=lw, zorder=zorder))


def _render_overlays(cache_root_p, vid, det_df, slice_boxes_df, out_dir, n_slices):
    """Overlay per lesion slice. GT = tight bbox around the 2D mask (green).

    Colour key: all detections red; the 5 most-confident yellow; the single
    best-IoU detection cyan, drawn last (topmost, uncovered). The caption shows
    the best IoU + that box's confidence; a right-side panel lists the 5
    most-confident detections with their IoU vs the GT bbox.
    """
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
        gt_box = DG.tight_bbox_from_mask(msl)  # tight bbox around the 2D GT mask

        dets = det_df[(det_df["volume_id"] == vid) & (det_df["slice_z"] == z)]
        boxes = dets[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        scores = dets["score"].to_numpy(dtype=float)
        ann = DG.overlay_annotations(boxes, scores, gt_box, top_k=5)
        ious, best_idx, top_idx = ann["ious"], ann["best_idx"], ann["top_idx"]
        top_set = set(top_idx)

        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.imshow(frame, cmap="gray", origin="upper")  # d0 depth vertical, d1 lateral horizontal
        if msl.any():
            ax.contour(msl, levels=[0.5], colors="lime", linewidths=1.0, zorder=5)  # mask contour
        if gt_box is not None:
            _rect(ax, gt_box, "lime", 1.0, zorder=5)  # tight GT bbox (dashed handled below)
            ax.patches[-1].set_linestyle("--")

        # layer order: red (others) < yellow (top-5) < cyan (best, topmost)
        for i in range(len(boxes)):
            if i == best_idx or i in top_set:
                continue
            _rect(ax, boxes[i], "red", 0.6, zorder=2)
        for i in top_idx:
            if i == best_idx:
                continue
            _rect(ax, boxes[i], "yellow", 1.3, zorder=4)
        if best_idx >= 0:
            _rect(ax, boxes[best_idx], "cyan", 2.0, zorder=6)

        if best_idx >= 0:
            ax.set_title(f"vol {vid} z={z}: {len(boxes)} det | best IoU={ious[best_idx]:.3f} "
                         f"(conf={scores[best_idx]:.3f})", fontsize=9)
        else:
            ax.set_title(f"vol {vid} z={z}: {len(boxes)} det (no GT/det)", fontsize=9)
        ax.axis("off")

        # right-side panel: 5 most-confident detections, conf + IoU vs GT bbox
        lines = ["top-5 by conf (yellow;", "  cyan = best IoU):"]
        for rank, i in enumerate(top_idx, 1):
            tag = " <best" if i == best_idx else ""
            lines.append(f"#{rank} conf={scores[i]:.3f} IoU={ious[i]:.3f}{tag}")
        if not top_idx:
            lines.append("(no detections)")
        ax.text(1.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

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
    parser.add_argument("--overlay-volume", type=int, nargs="+", default=[100],
                        help="one or more volumes to render overlays for (default: 100)")
    parser.add_argument("--overlay-slices", type=int, default=6)
    parser.add_argument("--score-thresh", type=float, default=C.DET_DIAG_SCORE_THRESH,
                        help="diagnostic score threshold for inference + recall (default DET_DIAG_SCORE_THRESH); "
                             "lower it (e.g. 0.01) to probe whether missed lesions are recoverable")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
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
            score_thresh=args.score_thresh, nms_thresh=C.DET_DIAG_NMS_THRESH,
            detections_per_img=C.DET_DIAG_DETECTIONS_PER_IMG, device=args.device)
        S.write_detections(df, det_dir / f"detections_val{vid}")
        all_dets.append(df)
    det_all = pd.concat(all_dets, ignore_index=True) if all_dets else S.empty_detections()

    score_thr = args.score_thresh
    iou_thr = C.DET_PER_SLICE_RECALL["iou_thresh"]
    hits, total, recall = DG.gt_recall(det_all, sb_val, val_ids, score_thr, iou_thr)
    rep = DG.recall_breakdown(det_all, sb_val, val_ids, score_thresh=score_thr,
                              iou_threshs=(0.1, 0.2, 0.3))
    pv = DG.per_volume_recall(det_all, sb_val, val_ids, score_thresh=score_thr, iou_thresh=iou_thr)
    missed = DG.missed_lesion_detail(det_all, sb_val, val_ids, score_thresh=score_thr, iou_thresh=iou_thr)

    fig_dir = Path(args.out_root) / "figures"
    overlays = []
    for ov in args.overlay_volume:
        overlays += _render_overlays(croot, ov, det_all, sb_val, fig_dir, args.overlay_slices)

    fr = rep["fire_rate"]
    print(f"# [2.x] Detection dump (checkpoint={Path(args.checkpoint).name}, best_epoch={cfg.get('best_epoch')})\n")
    print(f"val volumes scored     = {len(val_ids)}")
    print(f"total detections       = {len(det_all)}")
    print(f"per-slice 2D recall    = {hits}/{total} = {recall:.4f} "
          f"(score>={score_thr}, IoU>{iou_thr})")
    print("  ^ diagnostic foreshadowing the 3D recall ceiling; NOT the Phase-3 operating point.\n")
    hsc = sorted(pv["hit_slice_counts"])
    median_hit_slices = hsc[len(hsc) // 2] if hsc else 0
    print(f"per-LESION recall      = {pv['vols_with_hit']}/{pv['n_vols']} = {pv['recall']:.4f} "
          f"(volume hit on >=1 slice at IoU>{iou_thr}; 2D same-slice, NO linking)")
    print("  ^ correlated proxy for the 3D ceiling, not a bound; linked 3D recall is Phase 3.")
    print(f"  hit-slices per lesion: median={median_hit_slices}, min={hsc[0] if hsc else 0}, "
          f"max={hsc[-1] if hsc else 0}  (>1 => plausibly 3D-linkable)")
    n_zero = sum(1 for c in pv["hit_slice_counts"] if c == 0)
    print(f"  lesions never hit (0 slices) = {n_zero}/{pv['n_vols']}")
    if missed:
        print("  missed lesions (why the ceiling? small diag => intrinsic; best_iou~0.25 => recoverable):")
        for m in sorted(missed, key=lambda x: x["max_gt_diag"]):
            print(f"    vol {m['volume_id']:<4} n_gt={m['n_gt_boxes']:<4} max_diag={m['max_gt_diag']:6.1f}px "
                  f"best_iou={m['best_iou']:.3f} fired_on={m['fired_frac']:.2f} of GT slices")
    print()
    print("-- diagnostics (why is recall what it is?) --------------------------------")
    print(f"lesion-slice fire-rate = {fr['fired']}/{fr['lesion_slices']} = {fr['rate']:.4f} "
          f"(>=1 detection on the slice, IoU-agnostic)")
    print("recall vs IoU threshold (localisation tightness):")
    for thr in (0.1, 0.2, 0.3):
        print(f"    IoU>{thr:<4} : {rep['by_iou'][thr]:.4f}")
    print("recall vs GT box size (diag px; small = specks, large = real lesions) @ IoU>0.3:")
    for label, b in rep["by_size"].items():
        print(f"    diag {label:<10} n={b['n']:<5} recall={b['recall']:.4f}")
    print("---------------------------------------------------------------------------")
    print(f"overlays               = {len(overlays)} PNGs under {fig_dir}")
    for p in overlays:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
