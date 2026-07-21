"""[P3U2.diag] Deep candidate-pool diagnostics on cached seed0 — understand the 0.833/pool wall.

Reuses the deployed seed0 (epoch 6) + the [P3U.4b] `p3u_seed0_ep6` detection cache (no retrain). For the
Val set, at a fixed op and a chosen containment, it links tubes, reconstructs each to BOTH iso and official
boxes, and reports everything needed to diagnose the recall/pool wall:

  1. pool & TP/FP summary (how many 3D candidates are TP: official IoU > 0.3),
  2. confidence separation — detector score_max on TP tubes vs FP tubes (the key: is the FP tail low-score?),
  3. SCORE-FLOOR SWEEP — LUNA/NoduleSAT per-candidate score_max floor: recall vs pool at each floor,
  4. IoU distribution + RECONSTRUCTION LOSS (iso_iou vs official_iou; the 0.4 mm resampling ceiling, [3.2]),
  5. clustering / redundancy of the pool,
  6. candidate size stats (box_diag, z_span, extents) TP vs FP,
  7. tube-geometry features (3.D) TP vs FP — do they discriminate?,
  8. missed-lesion deep-dive — GT size, recon ceiling, did the detector fire (how often / how confident),
     best candidate iso vs official IoU (linker-limited vs reconstruction-limited).

Usage (server; CPU-fast if the [P3U.4b] cache is warm, else GPU to re-detect):
    python scripts/phase3_candidate_diagnostics.py \
        --checkpoint /home/maia-user/Andre2/outputs/phase2/checkpoints/retinanet_full_seed0.pt \
        --label p3u_seed0_ep6 --op-thresh 0.03 --containment off --device cuda
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
from abus_jcr.geometry import iou_official, storage_box_to_official, mask_to_box_storage
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import gt_reconstruction_consistency
from abus_jcr.probe.candidate_diag import (build_candidate_frame, score_floor_sweep, separability,
                                           tp_fp_split_stats, cluster_counts, _pct)
from _phase3_common import (add_phase3_paths, assert_device, cache_root, load_manifest,
                            load_official_gt, gt_official_tuple, load_or_run_detections,
                            filter_by_score)

FLOORS = [0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]


def _detector_fire_on_lesion(raw: pd.DataFrame, gt_iso_storage) -> dict:
    """Did the detector fire inside the GT's iso footprint? (x=d1, y=d0, half-open.)"""
    min_d0, min_d1, min_z, max_d0, max_d1, max_z = gt_iso_storage
    gx1, gx2 = min_d1, max_d1 + 1
    gy1, gy2 = min_d0, max_d0 + 1
    gt_n_slices = int(max_z - min_z + 1)
    inz = raw[(raw["slice_z"] >= min_z) & (raw["slice_z"] <= max_z)]
    if len(inz) == 0:
        return {"n_fire": 0, "n_slices_fired": 0, "gt_n_slices": gt_n_slices, "max_score": 0.0}
    ix1 = np.maximum(inz["x1"].to_numpy(), gx1); iy1 = np.maximum(inz["y1"].to_numpy(), gy1)
    ix2 = np.minimum(inz["x2"].to_numpy(), gx2); iy2 = np.minimum(inz["y2"].to_numpy(), gy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    hit = inter > 0
    fired = inz[hit]
    return {"n_fire": int(hit.sum()),
            "n_slices_fired": int(fired["slice_z"].nunique()) if len(fired) else 0,
            "gt_n_slices": gt_n_slices,
            "max_score": float(fired["score"].max()) if len(fired) else 0.0}


def _fmt_pct(d: dict) -> str:
    return (f"n={d['n']:>5}  mean={d['mean']:.3f}  p10={d['p10']:.3f}  p50={d['p50']:.3f}  "
            f"p90={d['p90']:.3f}  max={d['p100']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="[P3U2.diag] deep candidate-pool diagnostics")
    add_phase3_paths(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, help="reuse the [P3U.4b] label (p3u_seed0_ep6)")
    parser.add_argument("--op-thresh", type=float, default=C.DET_SELECT_OP_THRESH)
    parser.add_argument("--detect-op", type=float, default=0.005)
    parser.add_argument("--containment", default="off",
                        help="'off' (1.0; full recoverable pool) or a float like 0.80/0.95")
    parser.add_argument("--cluster-radius", type=float, default=float(C.FP_PROBE_CLUSTER_RADIUS))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
    from abus_jcr.detect.retinanet import load_checkpoint

    contain = 1.0 if str(args.containment).lower() == "off" else float(args.containment)
    manifest = load_manifest(args)
    croot = cache_root(args)
    gt_idx = load_official_gt(args, "val").set_index("public_id")
    val_ids = sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model, _cfg = load_checkpoint(ckpt)
    model.to(args.device)
    print(f"# [P3U2.diag] candidate diagnostics: {ckpt}")
    print(f"  label={args.label}  op={args.op_thresh}  containment={contain}  hit_iou={C.IOU_HIT_THRESHOLD}\n")

    tag = f"probe_{args.label}_op{args.detect_op}"
    frames, per_vol, recon_rows, missed_detail = [], [], [], []
    for k, v in enumerate(val_ids, 1):
        meta = K.read_meta(croot, v)
        mask_iso = np.asarray(K.open_mask(croot, v))
        gt_official = gt_official_tuple(gt_idx, v)
        gt_iso_storage = mask_to_box_storage((mask_iso > 0).astype(np.uint8))
        gt_iso_official = storage_box_to_official(gt_iso_storage)
        recon_self = float(gt_reconstruction_consistency((mask_iso > 0).astype(np.uint8), gt_official, meta))

        det_min = load_or_run_detections(args.out_root, tag, v, model, croot, args.detect_op,
                                         args.device, use_cache=not args.no_cache)
        raw = filter_by_score(det_min, args.op_thresh)
        tubes = link_tubes(raw, containment_thresh=contain)
        fr = build_candidate_frame(v, tubes, gt_official, gt_iso_official, meta)
        frames.append(fr)

        gt_diag = float(np.sqrt(sum(e ** 2 for e in gt_official[3:6])))
        n_tp = int(fr["is_tp"].sum())
        best_off = float(fr["official_iou"].max()) if len(fr) else 0.0
        best_iso = float(fr["iso_iou"].max()) if len(fr) else 0.0
        per_vol.append({"public_id": v, "n_cand": len(fr), "n_tp": n_tp, "recon_self_iou": recon_self,
                        "gt_diag": gt_diag, "best_official_iou": best_off, "best_iso_iou": best_iso})
        recon_rows.append(recon_self)
        if n_tp == 0:
            fire = _detector_fire_on_lesion(raw, gt_iso_storage)
            missed_detail.append({"public_id": v, "gt_diag": gt_diag,
                                  "gt_ext": [round(float(e), 1) for e in gt_official[3:6]],
                                  "recon_self_iou": round(recon_self, 3),
                                  "best_official_iou": round(best_off, 3), "best_iso_iou": round(best_iso, 3),
                                  **fire})
        print(f"  [vol {v} ({k}/{len(val_ids)})] cands={len(fr):>4} TP={n_tp} best_off_iou={best_off:.3f} "
              f"best_iso_iou={best_iso:.3f} recon_self={recon_self:.3f}", flush=True)
    del model

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    n_vol = len(val_ids)
    pv = pd.DataFrame(per_vol)

    # 1. POOL & TP/FP SUMMARY
    n_cand = len(frame); n_tp = int(frame["is_tp"].sum()); n_fp = n_cand - n_tp
    recall = int((pv["n_tp"] > 0).sum()) / n_vol
    print("\n" + "=" * 78 + "\n# 1. POOL & TP/FP SUMMARY\n")
    print(f"  candidates total={n_cand}  ({n_cand/n_vol:.1f}/vol)   TP={n_tp} ({n_tp/n_vol:.2f}/vol)   "
          f"FP={n_fp} ({n_fp/n_vol:.1f}/vol)")
    print(f"  linked recall = {recall:.3f} ({int((pv['n_tp']>0).sum())}/{n_vol})   "
          f"volumes with 0 TP (MISSED) = {int((pv['n_tp']==0).sum())}")

    # 2. CONFIDENCE SEPARATION
    print("\n# 2. CONFIDENCE SEPARATION — detector score_max on TP vs FP tubes\n")
    sc = tp_fp_split_stats(frame, "score_max")
    print(f"  TP score_max: {_fmt_pct(sc['TP'])}")
    print(f"  FP score_max: {_fmt_pct(sc['FP'])}")
    sep = separability(frame, "score_max")
    print(f"  separability: frac_FP_below_TP_median={sep['frac_fp_below_tp_median']:.3f}  "
          f"best_balacc={sep['best_balacc']:.3f} @ thresh={sep['best_thresh']:.3f}  "
          f"(min TP score_max = {sep['tp_min']:.3f})")

    # 3. SCORE-FLOOR SWEEP (the LUNA-style prefilter answer)
    print("\n# 3. SCORE-FLOOR SWEEP — per-candidate score_max floor (recall vs pool)\n")
    sweep = score_floor_sweep(frame, FLOORS, n_vol)
    print(f"  {'floor':>6} {'recall':>7} {'pool_mean':>10} {'pool_max':>9} {'n_tp_kept':>10}")
    for _, r in sweep.iterrows():
        flag = "  <= budget" if r["pool_max"] <= C.RESCORER_POOL_BUDGET else ""
        print(f"  {r['floor']:>6.2f} {r['recall']:>7.3f} {r['pool_mean']:>10.1f} "
              f"{int(r['pool_max']):>9d} {int(r['n_tp_kept']):>10d}{flag}")

    # 4. IoU + RECONSTRUCTION LOSS
    print("\n# 4. IoU DISTRIBUTION + RECONSTRUCTION LOSS (iso vs official; 0.4 mm resampling ceiling)\n")
    print(f"  candidate official_iou (all): {_fmt_pct(_pct(frame['official_iou'].to_numpy()))}")
    print(f"  best-per-vol official_iou   : {_fmt_pct(_pct(pv['best_official_iou'].to_numpy()))}")
    print(f"  best-per-vol iso_iou        : {_fmt_pct(_pct(pv['best_iso_iou'].to_numpy()))}")
    tp_recon = frame[frame["is_tp"]]["recon_loss"].to_numpy()
    print(f"  recon_loss (iso-official) on TP cands: {_fmt_pct(_pct(tp_recon))}")
    print(f"  GT self-reconstruction IoU (per-lesion ceiling): {_fmt_pct(_pct(np.array(recon_rows)))}")
    n_recon_marginal = int((pv["recon_self_iou"] < 0.5).sum())
    print(f"  lesions with recon_self_iou < 0.50 (resampling near-blocker): {n_recon_marginal}/{n_vol}")

    # 5. CLUSTERING
    print(f"\n# 5. CLUSTERING / REDUNDANCY (single-linkage @ r={args.cluster_radius:.0f} iso vox)\n")
    ncls, reds = [], []
    for v in val_ids:
        sub = frame[frame["public_id"] == v]
        if len(sub) == 0:
            continue
        centres = sub[["coordX", "coordY", "coordZ"]].to_numpy()
        nc, npnt, red = cluster_counts(centres, args.cluster_radius)
        ncls.append(nc); reds.append(red)
    print(f"  mean distinct clusters/vol = {np.mean(ncls):.1f}   mean redundancy (cands/cluster) = "
          f"{np.mean(reds):.2f}   (pool {n_cand/n_vol:.1f}/vol)")

    # 6. SIZE STATS
    print("\n# 6. CANDIDATE SIZE — TP vs FP\n")
    for col in ["box_diag", "z_span", "slice_count", "fill_ratio"]:
        st = tp_fp_split_stats(frame, col)
        print(f"  {col:>11}: TP {_fmt_pct(st['TP'])}")
        print(f"  {'':>11}  FP {_fmt_pct(st['FP'])}")

    # 7. TUBE-GEOMETRY FEATURES (3.D) TP vs FP
    print("\n# 7. TUBE-GEOMETRY FEATURES (3.D) — TP vs FP means (do they discriminate?)\n")
    for col in ["centroid_jitter", "area_cv", "area_peak_pos", "area_monotonicity", "score_std", "score_mean"]:
        st = tp_fp_split_stats(frame, col)
        print(f"  {col:>17}: TP mean={st['TP']['mean']:.3f}  FP mean={st['FP']['mean']:.3f}")

    # 8. MISSED-LESION DEEP-DIVE
    print("\n# 8. MISSED-LESION DEEP-DIVE (volumes with 0 TP candidate)\n")
    if not missed_detail:
        print("  (none — every volume has a TP candidate)")
    else:
        hdr = (f"  {'vid':>5} {'gt_diag':>8} {'recon':>6} {'fired_sl/gt':>12} {'n_fire':>7} "
               f"{'maxscore':>9} {'best_off':>9} {'best_iso':>9}  verdict")
        print(hdr)
        for m in missed_detail:
            fired = f"{m['n_slices_fired']}/{m['gt_n_slices']}"
            if m["max_score"] < args.op_thresh + 1e-9 and m["n_fire"] == 0:
                verdict = "DETECTOR-SILENT"
            elif m["best_iso_iou"] > C.IOU_HIT_THRESHOLD >= m["best_official_iou"]:
                verdict = "RECONSTRUCTION-LOST (iso>0.3, official<0.3)"
            elif m["best_iso_iou"] <= C.IOU_HIT_THRESHOLD:
                verdict = "LINKER/DETECTOR-LIMITED (iso<=0.3)"
            else:
                verdict = "LINKER-SUPPRESSED"
            print(f"  {m['public_id']:>5} {m['gt_diag']:>8.1f} {m['recon_self_iou']:>6.3f} {fired:>12} "
                  f"{m['n_fire']:>7} {m['max_score']:>9.3f} {m['best_official_iou']:>9.3f} "
                  f"{m['best_iso_iou']:>9.3f}  {verdict}")

    out_dir = Path(args.out_root) / "step0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(ckpt), "label": args.label, "op": args.op_thresh, "containment": contain,
               "n_vol": n_vol, "n_cand": n_cand, "n_tp": n_tp, "recall": recall,
               "confidence_separation": {"score_max_TP": sc["TP"], "score_max_FP": sc["FP"], **sep},
               "score_floor_sweep": sweep.to_dict(orient="records"),
               "per_vol": pv.to_dict(orient="records"),
               "missed": missed_detail,
               "recon_self_iou": {"values": recon_rows}}
    outp = out_dir / f"candidate_diag_{args.label}.json"
    outp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\njson = {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
