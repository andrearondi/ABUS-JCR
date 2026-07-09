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
            score_thresh=C.DET_DIAG_SCORE_THRESH, nms_thresh=C.DET_DIAG_NMS_THRESH,
            detections_per_img=C.DET_DIAG_DETECTIONS_PER_IMG, device=args.device)
        S.write_detections(df, det_dir / f"detections_val{vid}")
        all_dets.append(df)
    det_all = pd.concat(all_dets, ignore_index=True) if all_dets else S.empty_detections()

    score_thr = C.DET_PER_SLICE_RECALL["score_thresh"]
    iou_thr = C.DET_PER_SLICE_RECALL["iou_thresh"]
    hits, total, recall = DG.gt_recall(det_all, sb_val, val_ids, score_thr, iou_thr)
    rep = DG.recall_breakdown(det_all, sb_val, val_ids, score_thresh=score_thr,
                              iou_threshs=(0.1, 0.2, 0.3))
    pv = DG.per_volume_recall(det_all, sb_val, val_ids, score_thresh=score_thr, iou_thresh=iou_thr)
    missed = DG.missed_lesion_detail(det_all, sb_val, val_ids, score_thresh=score_thr, iou_thresh=iou_thr)

    fig_dir = Path(args.out_root) / "figures"
    overlays = _render_overlays(croot, args.overlay_volume, det_all, sb_val, fig_dir, args.overlay_slices)

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
