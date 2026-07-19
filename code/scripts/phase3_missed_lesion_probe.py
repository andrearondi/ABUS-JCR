"""[P3U.4d] Attribute each MISSED val lesion: detector-never-fired vs linker-killed (diagnostic; GPU only to detect).

The Stage-1 probe ([P3U.4b]/[P3U.4c]) shows linked recall SATURATES at 0.833 — 5 of 30 val
lesions are never recovered at any operating point, and lowering ``op`` makes recall WORSE
(runaway/containment eats TPs). That single number cannot say WHETHER the 5 misses are a
detector/anchor limit (the lesion produces no usable boxes) or a linker artifact (the boxes
exist but linking/suppression/reconstruction drops them). This script answers that — cheaply,
on the deployed seed0 checkpoint, reusing the CACHED detections from [P3U.4b] (no retrain).

For each Validation volume (one official GT lesion each) it computes, all through the IDENTICAL
frozen detect->link->official path used everywhere else:

  1. DETECTOR COVERAGE (linking-independent). Using the Phase-1 iso GT slice-boxes
     (``labels/slice_boxes_Validation``; same half-open ``x=d1, y=d0`` frame as detections),
     count the GT slices on which at least one raw detection reaches 2D IoU >= --overlap-iou
     with the lesion's per-slice box. ``n_cov_slices >= LINK_MIN_TUBE_LEN`` == the detector laid
     down the raw material a tube needs. ``n_cov_slices == 0`` == the detector never fired on the
     lesion -> an anchor/detector wall D1-D6 cannot move.
  2. FROZEN linked max IoU_official over tubes (the deployed linker) -> the actual hit/miss.
  3. PERMISSIVE linked max IoU_official (containment OFF, drift caps OFF, min_tube_len=2) -> does
     relaxing the linker's SUPPRESSION recover the lesion? If yes, the miss is a linker knob, not
     the detector.

Classification per missed lesion:
  HIT               frozen linker already clears IOU_HIT_THRESHOLD (the 25).
  LINKER-SUPPRESSED frozen misses but PERMISSIVE hits -> containment/caps/min_len suppressed it. FIXABLE.
  LINKER-LOST       detector covered >= min_len slices, neither linker hits, but a tube forms near the
                    lesion (permissive max IoU in a grey band) -> seeding/reconstruction granularity. FIXABLE-ish.
  DETECTOR-WALL     detector coverage < min_len slices AND permissive max IoU ~ 0 -> the detector never
                    produced usable boxes. NOT a linker fix -> escalate Inv.-1 (anchor grid) if these are
                    the small-diag tail.

Prints a per-lesion table + a MISS-ATTRIBUTION summary and the size (official extent) split of
missed vs hit lesions, then writes ``outputs/phase3/step0_probe/missed_<label>.json``.

Usage (server, GPU — reuses the [P3U.4b] op=0.005 detection cache if --label matches):
    python scripts/phase3_missed_lesion_probe.py \
        --checkpoint /home/maia-user/Andre2/outputs/phase2/checkpoints/retinanet_full_seed0.pt \
        --label p3u_seed0_ep6 --op-thresh 0.03 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.geometry import iou_official
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official
from _phase2_common import load_slice_boxes
from _phase3_common import (add_phase3_paths, assert_device, cache_root, load_manifest,
                            load_official_gt, gt_official_tuple, load_or_run_detections,
                            filter_by_score)

# grey band: a tube whose hull overlaps the GT this much but < IOU_HIT is "near but geometry misses".
GREY_BAND = 0.20


def _iou_2d_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Half-open 2D IoU of one ``box`` (4,) vs ``boxes`` (m, 4) — matches the detect/link contract."""
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(ix2 - ix1, 0.0, None); ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / union, 0.0)


def _gt_slice_boxes_by_vid(args):
    """{vid: {slice_z: ndarray(k,4) half-open x1,y1,x2,y2}} from ``slice_boxes_Validation`` or None."""
    try:
        df = load_slice_boxes(args, "Validation")
    except FileNotFoundError:
        return None
    need = {"volume_id", "slice_z", "r0", "c0", "r1", "c1"}
    if not need.issubset(df.columns):
        return None
    out: dict = {}
    for vid, grp in df.groupby("volume_id", sort=False):
        by_z: dict = {}
        for z, zg in grp.groupby("slice_z", sort=False):
            # inclusive (r0,c0,r1,c1) -> half-open (x1=c0, y1=r0, x2=c1+1, y2=r1+1); x=d1(col), y=d0(row)
            boxes = np.stack([zg["c0"].to_numpy(dtype=float), zg["r0"].to_numpy(dtype=float),
                              zg["c1"].to_numpy(dtype=float) + 1.0, zg["r1"].to_numpy(dtype=float) + 1.0], axis=1)
            by_z[int(z)] = boxes
        out[int(vid)] = by_z
    return out


def _detector_coverage(raw: pd.DataFrame, gt_by_z: dict, overlap_iou: float):
    """(n_cov_slices, best_2d_iou, n_gt_slices) — GT slices with a raw det clearing ``overlap_iou``."""
    if not gt_by_z:
        return 0, 0.0, 0
    n_cov, best = 0, 0.0
    for z, gt_boxes in gt_by_z.items():
        dz = raw[raw["slice_z"] == z]
        if len(dz) == 0:
            continue
        det_boxes = dz[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        slice_best = 0.0
        for gb in gt_boxes:
            ious = _iou_2d_vec(gb, det_boxes)
            if ious.size:
                slice_best = max(slice_best, float(ious.max()))
        best = max(best, slice_best)
        if slice_best >= overlap_iou:
            n_cov += 1
    return n_cov, best, len(gt_by_z)


def _max_linked_iou(raw: pd.DataFrame, gt, meta, *, permissive: bool) -> float:
    """Max IoU_official over the volume's tubes; ``permissive`` turns OFF containment + drift caps."""
    if permissive:
        tubes = link_tubes(raw, min_tube_len=2, max_tube_zspan=None,
                           max_centroid_drift=None, containment_thresh=1.0)
    else:
        tubes = link_tubes(raw)                     # frozen conventions
    best = 0.0
    for tube in tubes:
        best = max(best, float(iou_official(iso_tube_to_official(tube, meta), gt)))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="[P3U.4d] missed-lesion attribution (detector vs linker)")
    add_phase3_paths(parser)
    parser.add_argument("--checkpoint", required=True, help="deployed <run>.pt to attribute (e.g. seed0 ep6)")
    parser.add_argument("--label", required=True,
                        help="cache label; use the [P3U.4b] label (e.g. p3u_seed0_ep6) to REUSE its cache")
    parser.add_argument("--op-thresh", type=float, default=C.DET_SELECT_OP_THRESH,
                        help="operating point to analyse (default DET_SELECT_OP_THRESH; the recall peak)")
    parser.add_argument("--detect-op", type=float, default=0.005,
                        help="detector run threshold (default 0.005 to match [P3U.4b] cache); filtered up to --op-thresh")
    parser.add_argument("--overlap-iou", type=float, default=0.10,
                        help="min 2D IoU for a raw det to 'cover' a GT slice (default 0.10)")
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
    gt_slices = _gt_slice_boxes_by_vid(args)
    if gt_slices is None:
        print("WARNING: labels/slice_boxes_Validation not found (or missing r0/c0/r1/c1) — "
              "detector-coverage column disabled; attribution falls back to permissive-link only.\n")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model, cfg = load_checkpoint(ckpt)
    model.to(args.device)
    print(f"# [P3U.4d] missed-lesion attribution: {ckpt}")
    print(f"  label={args.label}  op_thresh={args.op_thresh}  detect_op={args.detect_op}  "
          f"overlap_iou={args.overlap_iou}  hit_iou={C.IOU_HIT_THRESHOLD}  min_tube_len={C.LINK_MIN_TUBE_LEN}\n")

    tag = f"probe_{args.label}_op{args.detect_op}"            # matches phase3_step0_checkpoint_probe cache
    rows = []
    for k, v in enumerate(val_ids, 1):
        det_min = load_or_run_detections(args.out_root, tag, v, model, croot, args.detect_op,
                                         args.device, use_cache=not args.no_cache)
        raw = filter_by_score(det_min, args.op_thresh)
        gt = gt_by_vid[v]; meta = meta_by_vid[v]
        n_cov, best2d, n_gt_sl = _detector_coverage(raw, (gt_slices or {}).get(v), args.overlap_iou)
        froz = _max_linked_iou(raw, gt, meta, permissive=False)
        perm = _max_linked_iou(raw, gt, meta, permissive=True)
        rows.append({"vid": v, "n_raw": int(len(raw)), "n_gt_slices": n_gt_sl,
                     "n_cov_slices": n_cov, "best_2d_iou": best2d,
                     "frozen_iou": froz, "perm_iou": perm,
                     "x_len": gt[3], "y_len": gt[4], "z_len": gt[5]})
        print(f"  [detect] vol {v} ({k}/{len(val_ids)}): raw={len(raw)} @op{args.op_thresh}", flush=True)
    del model

    hit_iou = C.IOU_HIT_THRESHOLD
    min_len = C.LINK_MIN_TUBE_LEN
    have_cov = gt_slices is not None

    def classify(r):
        if r["frozen_iou"] > hit_iou:
            return "HIT"
        if r["perm_iou"] > hit_iou:
            return "LINKER-SUPPRESSED"
        if have_cov and r["n_cov_slices"] >= min_len:
            return "LINKER-LOST"
        if r["perm_iou"] >= GREY_BAND:
            return "LINKER-LOST"
        return "DETECTOR-WALL"

    for r in rows:
        r["inplane_diag"] = math.sqrt(r["x_len"] ** 2 + r["y_len"] ** 2)
        r["klass"] = classify(r)

    order = {"DETECTOR-WALL": 0, "LINKER-LOST": 1, "LINKER-SUPPRESSED": 2, "HIT": 3}
    rows.sort(key=lambda r: (order[r["klass"]], r["vid"]))

    print(f"\n# per-lesion attribution (Val n={len(rows)})\n")
    hdr = (f"{'vid':>5} {'class':>18} {'n_raw':>7} {'cov/gtsl':>9} {'best2d':>7} "
           f"{'froz_iou':>9} {'perm_iou':>9} {'x_len':>6} {'y_len':>6} {'z_len':>6} {'ipdiag':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cov = f"{r['n_cov_slices']}/{r['n_gt_slices']}" if have_cov else "n/a"
        print(f"{r['vid']:>5} {r['klass']:>18} {r['n_raw']:>7} {cov:>9} {r['best_2d_iou']:>7.3f} "
              f"{r['frozen_iou']:>9.3f} {r['perm_iou']:>9.3f} {r['x_len']:>6.1f} {r['y_len']:>6.1f} "
              f"{r['z_len']:>6.1f} {r['inplane_diag']:>7.1f}", flush=True)

    counts = {kls: sum(1 for r in rows if r["klass"] == kls) for kls in order}
    n_hit = counts["HIT"]
    missed = [r for r in rows if r["klass"] != "HIT"]
    hits = [r for r in rows if r["klass"] == "HIT"]
    print(f"\n# MISS-ATTRIBUTION (n_val={len(rows)}, hit={n_hit}, recall={n_hit/len(rows):.4f})")
    for kls in order:
        if kls != "HIT":
            print(f"  {kls:>18}: {counts[kls]}")
    if missed and hits:
        md = np.array([r["inplane_diag"] for r in missed]); hd = np.array([r["inplane_diag"] for r in hits])
        print(f"\n  in-plane diag (official units) — MISSED: mean {md.mean():.1f} "
              f"[{md.min():.1f}, {md.max():.1f}]   HIT: mean {hd.mean():.1f} [{hd.min():.1f}, {hd.max():.1f}]")
        print("  (if MISSED lesions are the small tail AND classed DETECTOR-WALL -> Inv.-1 anchor limit, escalate)")

    wall = counts["DETECTOR-WALL"]
    print("\nVERDICT:")
    if wall == 0 and missed:
        print("  every miss is LINKER-side (suppressed/lost) -> the detector covers all lesions; "
              "fix the linker (containment/caps/min_len/seeding), re-probe [P3U.4c], THEN run [P3U.5].")
    elif wall:
        print(f"  {wall} miss(es) are a DETECTOR-WALL (no usable boxes) -> a linker fix CANNOT recover them. "
              "STOP before [P3U.5]; escalate Inv.-1 (anchor grid / anchor_min_base / finer pyramid level) "
              "if these are the small-diag tail.")
    else:
        print("  no misses (recall 1.0 at this op) — unexpected vs the sweep; re-check --op-thresh.")

    out_dir = Path(args.out_root) / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "label": args.label, "op_thresh": args.op_thresh,
               "detect_op": args.detect_op, "overlap_iou": args.overlap_iou, "hit_iou": hit_iou,
               "min_tube_len": min_len, "have_coverage": have_cov, "counts": counts,
               "recall": n_hit / len(rows) if rows else float("nan"), "rows": rows}
    outp = out_dir / f"missed_{args.label}.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"\njson = {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
