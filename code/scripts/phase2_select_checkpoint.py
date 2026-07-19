"""[D3] Post-hoc checkpoint selection on the true linked 3D val CPM (Inv. 2 A1; GPU).

``train.py`` saves every epoch to ``checkpoints/<run>/epoch{e:02d}.pt`` and selects
nothing. This script runs each saved epoch ``>= DET_SELECT_MIN_EPOCH`` through the
IDENTICAL frozen Phase-3 detect->link->oracle path (same NMS/cap/linking as [3.4]) at the
fixed reference operating point ``DET_SELECT_OP_THRESH``, computes the Inv.-3 linked CPM
(``average_recall`` via the official oracle), picks the argmax (ties -> later epoch), and
copies the winner to the deployed byte-stable ``checkpoints/<run>.pt`` that Phase 3 loads.
Reports the winner's CPM with a volume-level bootstrap CI (Inv. 12) and the neighbouring
epochs' CPMs (selection-stability flag).

Val set per run:
  - retinanet_full_seed{s}: the 30 Validation volumes.
  - retinanet_fold{f}:      the held-out Train fold (manifest.fold == f) — its OOF eval set.

Usage (server, GPU):
    python scripts/phase2_select_checkpoint.py --run retinanet_full_seed0 --device cuda
    python scripts/phase2_select_checkpoint.py --run retinanet_fold0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.geometry import iou_official
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official
from abus_jcr.link.aggregate import score_stats
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling, bootstrap_cpm_ci
from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
from abus_jcr.detect.select import select_epoch, selection_stability
from _phase3_common import (add_phase3_paths, assert_device, cache_root, load_manifest,
                            load_official_gt, gt_official_tuple, load_or_run_detections)


def _run_val_ids_and_gt(args, manifest, run: str):
    """(val_ids, gt_split_name) for a run: fold -> held-out Train fold; full -> Validation."""
    if run.startswith("retinanet_fold"):
        f = int(run.split("fold")[-1])
        ids = sorted(int(v) for v in manifest[(manifest["split"] == "train")
                                              & (manifest["fold"] == f)]["volume_id"])
        return ids, "train"
    if run.startswith("retinanet_full_seed"):
        ids = sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])
        return ids, "val"
    raise SystemExit(f"unrecognised run {run!r} (expected retinanet_fold<f> or retinanet_full_seed<s>)")


def _linked_cpm(det_by_vid, gt_by_vid, meta_by_vid, gt_used):
    """Link -> reconstruct -> official oracle. Returns (cpm, ceiling, recall, cands/vol, pred_df)."""
    n_hit, pools, preds = 0, [], []
    for vid, det in det_by_vid.items():
        tubes = link_tubes(det)                     # frozen linking params (conventions)
        pools.append(len(tubes))
        gt = gt_by_vid[int(vid)]
        meta = meta_by_vid[int(vid)]
        hit = False
        for tube in tubes:
            off = iso_tube_to_official(tube, meta)
            sc = score_stats(tube)["score_max"]
            preds.append({"public_id": int(vid), "coordX": off[0], "coordY": off[1], "coordZ": off[2],
                          "x_length": off[3], "y_length": off[4], "z_length": off[5],
                          "probability": min(float(sc), 0.999999)})
            if iou_official(off, gt) > C.IOU_HIT_THRESHOLD:
                hit = True
        n_hit += int(hit)
    recall = n_hit / max(len(det_by_vid), 1)
    pool_mean = float(np.mean(pools)) if pools else float("nan")
    pred_df = pd.DataFrame(preds, columns=PRED_COLUMNS)
    if len(pred_df):
        res = evaluate_froc(gt_used, pred_df)
        return float(cpm(res)), float(recall_ceiling(res)), recall, pool_mean, pred_df
    return float("nan"), float("nan"), recall, pool_mean, pred_df


def main() -> int:
    parser = argparse.ArgumentParser(description="[D3] post-hoc checkpoint selection (linked val CPM)")
    add_phase3_paths(parser)
    parser.add_argument("--run", required=True,
                        help="retinanet_fold<f> or retinanet_full_seed<s>")
    parser.add_argument("--op-thresh", type=float, default=C.DET_SELECT_OP_THRESH)
    parser.add_argument("--min-epoch", type=int, default=C.DET_SELECT_MIN_EPOCH,
                        help="earliest epoch eligible for SELECTION (earlier epochs are shown but not picked)")
    parser.add_argument("--eval-from", type=int, default=0,
                        help="earliest epoch to EVALUATE + print (default 0, for completeness); "
                             "epochs < --min-epoch are marked (diag) and excluded from selection")
    parser.add_argument("--cpm-tol", type=float, default=C.DET_SELECT_CPM_TOL,
                        help="CPM within this of the max is a tie -> break on highest recall ceiling")
    parser.add_argument("--no-cache", action="store_true", help="force re-detect (ignore cache)")
    parser.add_argument("--no-deploy", action="store_true",
                        help="report + write the selection JSON but do NOT overwrite the deployed <run>.pt "
                             "(verification-only; e.g. [P3U.4e] confirming the ceiling-aware pick)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert_device(args.device)
    from abus_jcr.detect.retinanet import load_checkpoint

    manifest = load_manifest(args)
    croot = cache_root(args)
    run = args.run
    val_ids, gt_split = _run_val_ids_and_gt(args, manifest, run)
    gt_idx = load_official_gt(args, gt_split).set_index("public_id")
    meta_by_vid = {v: K.read_meta(croot, v) for v in val_ids}
    gt_by_vid = {v: gt_official_tuple(gt_idx, v) for v in val_ids}
    gt_used = gt_idx.loc[val_ids].reset_index()[GT_COLUMNS]

    epochs_dir = Path(args.phase2_out) / "checkpoints" / run
    epoch_ckpts = sorted(epochs_dir.glob("epoch*.pt"))
    if not epoch_ckpts:
        raise SystemExit(f"no per-epoch checkpoints under {epochs_dir} (run phase2_train_retinanet first)")

    print(f"# [D3] post-hoc selection for {run} — linked val CPM @ op={args.op_thresh} "
          f"(select epoch>={args.min_epoch}, show from {args.eval_from}, cpm_tol={args.cpm_tol}, "
          f"n_val={len(val_ids)}, gt={gt_split})\n")
    hdr = f"{'epoch':>6} {'CPM':>8} {'ceiling':>8} {'recall':>8} {'cands/vol':>10} {'val_loss':>9} {'val_ap':>7} {'sel?':>5}"
    print(hdr); print("-" * len(hdr))

    epoch_cpms, epoch_ceilings, rows = {}, {}, {}
    for ck in epoch_ckpts:
        e = int(ck.stem.replace("epoch", ""))
        if e < args.eval_from:
            continue
        model, cfg = load_checkpoint(ck)
        model.to(args.device)
        tag = f"select_{run}_e{e:02d}_op{args.op_thresh}"
        det_by_vid = {}
        for v in val_ids:
            det_by_vid[v] = load_or_run_detections(
                args.out_root, tag, v, model, croot, args.op_thresh, args.device,
                use_cache=not args.no_cache)
        del model
        cpm_v, ceil_v, rec, pool, _pred = _linked_cpm(det_by_vid, gt_by_vid, meta_by_vid, gt_used)
        eligible = e >= args.min_epoch
        rows[e] = {"cpm": cpm_v, "ceiling": ceil_v, "recall": rec, "cands_per_vol": pool,
                   "val_loss": cfg.get("val_loss"), "val_ap": cfg.get("val_ap"), "eligible": eligible}
        if eligible:                     # only epochs >= min_epoch enter the selection
            epoch_cpms[e] = cpm_v
            epoch_ceilings[e] = ceil_v
        vl = cfg.get("val_loss"); va = cfg.get("val_ap")
        print(f"{e:>6} {cpm_v:>8.4f} {ceil_v:>8.4f} {rec:>8.4f} {pool:>10.1f} "
              f"{(vl if vl is not None else float('nan')):>9.4f} "
              f"{(va if va is not None else float('nan')):>7.4f} "
              f"{'  ' if eligible else 'diag':>5}", flush=True)

    # Ceiling-aware selection (A1 revised): among CPM-ties (within cpm_tol), highest ceiling, then earliest.
    best_e = select_epoch(epoch_cpms, args.min_epoch, epoch_ceilings, args.cpm_tol)
    spread, top = selection_stability(epoch_cpms, args.min_epoch)

    # Bootstrap CI on the winner (Inv. 12): re-run its detections to build the pred_df.
    model, cfg = load_checkpoint(epochs_dir / f"epoch{best_e:02d}.pt")
    model.to(args.device)
    tag = f"select_{run}_e{best_e:02d}_op{args.op_thresh}"
    det_by_vid = {v: load_or_run_detections(args.out_root, tag, v, model, croot, args.op_thresh,
                                            args.device, use_cache=not args.no_cache) for v in val_ids}
    del model
    _cpm, _ceil, _rec, _pool, pred_df = _linked_cpm(det_by_vid, gt_by_vid, meta_by_vid, gt_used)
    ci = bootstrap_cpm_ci(gt_used, pred_df)

    max_cpm = max(epoch_cpms.values()) if epoch_cpms else float("nan")
    tie_note = ""
    if epoch_cpms and rows[best_e]["cpm"] < max_cpm - 1e-9:
        argmax_e = max(epoch_cpms, key=lambda e: (epoch_cpms[e], e))
        tie_note = (f"  [ceiling-tie-break: epoch {argmax_e} had the bare-max CPM {max_cpm:.4f} but "
                    f"ceiling {rows[argmax_e]['ceiling']:.4f} < epoch {best_e}'s {rows[best_e]['ceiling']:.4f}]")
    print(f"\nSELECTED epoch {best_e}: linked val CPM = {rows[best_e]['cpm']:.4f} "
          f"[{ci['lo']:.4f}, {ci['hi']:.4f}] (95% boot), ceiling {rows[best_e]['ceiling']:.4f}, "
          f"cands/vol {rows[best_e]['cands_per_vol']:.1f}{tie_note}")
    print(f"top-3 by CPM: {[(e, round(c, 4)) for e, c in top]}  (spread {spread:.4f})")
    if spread > 0.05:
        print("  ^ WIDE spread across near-tied epochs — the 30-lesion val CPM barely resolves them; "
              "flag low selection resolution in the report.")

    # Deploy: copy the winning epoch to the byte-stable <run>.pt Phase 3 loads.
    deployed = Path(args.phase2_out) / "checkpoints" / f"{run}.pt"
    if args.no_deploy:
        print(f"\n[--no-deploy] would deploy epoch{best_e:02d}.pt -> {deployed} (NOT copied; verification-only)")
    else:
        shutil.copyfile(epochs_dir / f"epoch{best_e:02d}.pt", deployed)
        print(f"\ndeployed -> {deployed}  (from epoch{best_e:02d}.pt)")

    out_dir = Path(args.out_root) / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run": run, "op_thresh": args.op_thresh, "min_epoch": args.min_epoch,
               "eval_from": args.eval_from, "cpm_tol": args.cpm_tol,
               "gt_split": gt_split, "n_val": len(val_ids), "per_epoch": rows,
               "selected_epoch": best_e, "selected_cpm": rows[best_e]["cpm"],
               "selected_ceiling": rows[best_e]["ceiling"],
               "ci": {"lo": ci["lo"], "hi": ci["hi"], "point": ci["point"]},
               "top3": top, "spread": spread,
               "deployed": None if args.no_deploy else str(deployed)}
    (out_dir / f"select_{run}.json").write_text(json.dumps(payload, indent=2))
    print(f"json = {out_dir / f'select_{run}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
