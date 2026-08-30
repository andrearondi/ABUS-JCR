"""[4.3] Pretrain the shared 3D encoder + B1Head per seed — **this run IS rung B1**.

Trains ``CandidateEncoder + B1Head`` end to end on the **train** pool only (Inv. 10), with
crop augmentation applied by re-extracting from the iso cache: a mirror along **d0 = the
MEASURED LATERAL axis**, plus centre jitter. **Never d1 (depth/beam)** — Inv. 13. Every epoch
is saved; the deployed epoch is chosen **post-hoc on val CPM** through the official oracle on
the PAIRED seed pool ``full_seed{r}`` (Inv. 14) — never on val loss.

An **early read** on exit check 4 is printed here: this seed's B1 val CPM against **this
seed's B0**, which is MEASURED on the pool being held (the same record ranked by
``score_max``), never read from a constant — a hard-coded floor survives a substrate
promotion while its meaning does not. The BINDING exit check is on the [4.6] B1's mean over
the 3 replicas, at [4.7].

If B1 lands at or below B0 there are **two** live explanations and they need different
remedies, so do not jump to the fallback:

1. the encoder/token/crop pipeline is broken — then the per-candidate discrimination will be
   far below the pool's measured single-feature ceiling (balanced accuracy **0.811** on the
   promoted val pool, ``[F.9]`` §1), and ``--encoder small_cnn_3d`` (spec Open escalation #2)
   is the pre-registered response;
2. B0 is simply a strong ranking on this substrate — the promoted detector's ``score_max``
   separates TP/FP at Cliff's δ **0.713** (was 0.600) and already carries **39.8 %** of the
   volume-trust signal (was 18.8 %, ``[F.8]``). A per-candidate classifier with no volume
   context can legitimately fail to beat it, which is a finding about B0, not a broken
   pipeline, and it makes B2−B1 *more* interesting rather than less.

The printed line reports B1's balanced accuracy alongside the CPM so the two can be told
apart at a glance.

Usage:
    python scripts/phase4_pretrain_encoder.py --seed 0 --device cuda \\
        --phase1-out ... --phase3-out ... --out-root ...
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from abus_jcr import conventions as C
from abus_jcr.rescore.crops import open_crop_cache
from abus_jcr.rescore.datasets import CropDataset
from abus_jcr.rescore.encoder import build_encoder
from abus_jcr.probe.pool_diag import best_balacc
from abus_jcr.rescore.evaluate import evaluate_variant
from abus_jcr.rescore.objective import record_lesion_weights
from abus_jcr.rescore.setmodel import B1Rescorer
from abus_jcr.rescore.train import pretrain_encoder, write_epoch_table

from _phase4_common import (add_phase4_paths, assert_device, build_features, cache_root,
                            crops_dir, dump_json, encoder_dir, gt_for_pool, iso_shape_map,
                            label_codes, loader_kwargs, load_gt, load_record, rest_blocks,
                            val_pool_for_seed)


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.3] pretrain the shared encoder (= rung B1)")
    add_phase4_paths(ap)
    ap.add_argument("--seed", type=int, required=True, choices=list(C.RESC_SEEDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=C.RESC_ENC_EPOCHS)
    ap.add_argument("--batch", type=int, default=C.RESC_ENC_BATCH)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--encoder", default=C.RESC_ENCODER,
                    choices=[C.RESC_ENCODER, C.RESC_ENCODER_FALLBACK])
    ap.add_argument("--crop-side", type=float, default=None,
                    help="[4.2] control only: force a FIXED ROI side instead of the adaptive one")
    ap.add_argument("--b1-hidden", type=int, default=256,
                    help="B1 MLP width; [4.5] later re-sizes it to match the selected set module")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="bootstrap draws for the per-epoch val CI (the final grid uses 1000)")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    assert_device(args.device)

    rec_tr = load_record(args, "train")
    rec_va_all = load_record(args, "val")
    rec_va = val_pool_for_seed(rec_va_all, args.seed)
    gt_va = gt_for_pool(load_gt(args, "val"), rec_va)

    # B0 for THIS seed, measured on the pool actually held (one oracle call). Never a
    # constant: see conventions.py 4 (H). This is also the Inv.-8 ceiling for every epoch.
    b0 = evaluate_variant(rec_va, rec_va["score_max"].to_numpy(float), gt_va,
                          seed_tag=f"B0_seed{args.seed}", n_boot=0)
    print(f"# B0 (this seed, measured): CPM {b0['cpm']:.4f}, ceiling {b0['ceiling']:.4f} "
          f"-- the floor B1 must clear, and the cap no rung can exceed (Inv. 8)")
    iso_tr = iso_shape_map(args, rec_tr)
    iso_va = iso_shape_map(args, rec_va)

    # non-appearance blocks only: the appearance block IS what the encoder is learning
    rest = rest_blocks(C.RESC_TOKEN_BLOCKS)
    Ztr, names, stats = build_features(rec_tr, None, rest, iso_tr, stats=None)   # fit on TRAIN
    Zva, _, _ = build_features(rec_va, None, rest, iso_va, stats=stats)          # apply unchanged
    print(f"# rest-token width = {Ztr.shape[1]} ({', '.join(rest)})")

    if args.crop_side is None:
        crops_va_all, _ = open_crop_cache(crops_dir(args), "val")
        va_rows = rec_va_all.index[rec_va_all["detector_of_origin"]
                                   == f"full_seed{args.seed}"].to_numpy()
        crops_va = crops_va_all[va_rows]    # (n_seed_rows, 48, 48, 48), only this seed's rows
    else:
        # [4.2] fixed-crop control: the cache holds ADAPTIVE crops, so re-extract un-augmented
        print(f"# [4.2] control: fixed ROI side {args.crop_side} iso vox (no crop cache)")
        ds_va = CropDataset(rec_va, np.zeros((len(rec_va), 1)), np.zeros(len(rec_va)),
                            cache_root=cache_root(args), source="volume", augment=False,
                            fixed_side=args.crop_side)
        crops_va = np.stack([ds_va.crop(i) for i in range(len(rec_va))]).astype(np.float16)

    ds_tr = CropDataset(rec_tr, Ztr, label_codes(rec_tr), cache_root=cache_root(args),
                        source="volume", augment=True, seed=args.seed,
                        fixed_side=args.crop_side)
    loader = torch.utils.data.DataLoader(ds_tr.torch_dataset(), batch_size=args.batch,
                                         shuffle=True, drop_last=False,
                                         **loader_kwargs(args.num_workers))

    d_app = int(C.RESC_D_APP)
    d_in = d_app + Ztr.shape[1]

    # a FACTORY: pretrain_encoder seeds before invoking it, so the weights cannot inherit this
    # process's RNG state (rescore/train.resolve_model — worth 0.021 CPM when got backwards)
    def model_factory():
        return (build_encoder(args.encoder, d_app=d_app),
                B1Rescorer(d_in=d_in, d_model=128, hidden=args.b1_hidden, depth=2))

    _e, _h = model_factory()
    n_enc = sum(p.numel() for p in _e.parameters())
    n_head = sum(p.numel() for p in _h.parameters())
    del _e, _h
    print(f"# encoder {args.encoder}: {n_enc/1e6:.2f} M params; B1Head: {n_head/1e3:.1f} k params")
    print(f"# [4.2c] objective: gamma={C.RESC_FOCAL_GAMMA} alpha={args.alpha} "
          f"per_lesion_weights={C.RESC_PER_LESION_WEIGHTS}")

    # Per-epoch probability columns, kept so the CI can be computed ONCE, after selection.
    #
    # Until 2026-08-31 every epoch ran a full `--n-boot` (default 200) bootstrap. Measured on
    # job 17425385: ~2.9 s per draw => ~10 min per epoch, ~5 h per arm, with the GPU idle
    # throughout (which is also what the NSC efficiency reaper was killing). And 29 of the 30
    # intervals were discarded unread: `rescore.train.pretrain_encoder` selects with
    # `select_epoch_by_val_cpm`, whose docstring calls `val_cpm` "the ONLY selection signal".
    #
    # `bootstrap_cpm_ci` is a pure function of (gt, pred, n_boot, seed) — pinned by
    # tests/test_paired_bootstrap.py — so deferring it reproduces the selected epoch's interval
    # exactly. Nothing reported moves; ~29/30 of the evaluation cost disappears.
    probs_by_epoch: dict = {}

    def evaluate_epoch(epoch: int, enc, head) -> dict:
        enc.eval(); head.eval()
        with torch.no_grad():
            embs = []
            for s in range(0, len(rec_va), 64):
                x = torch.from_numpy(np.ascontiguousarray(
                    crops_va[s:s + 64][:, None], dtype=np.float32)).to(args.device)
                embs.append(enc(x).cpu().numpy())
            emb = np.concatenate(embs, axis=0)
            feats = torch.from_numpy(np.concatenate([emb, Zva], axis=1).astype(np.float32))
            logits = head(feats.to(args.device)[None]).squeeze(0)
            prob = torch.sigmoid(logits).clamp(0.0, 1.0 - C.RESC_PROB_EPS).cpu().numpy()
        probs_by_epoch[int(epoch)] = prob
        res = evaluate_variant(rec_va, prob, gt_va, seed_tag=f"B1_seed{args.seed}",
                               n_boot=0)
        # Per-candidate discrimination, alongside the ranking metric. If B1 fails to clear B0
        # this is what distinguishes "the pipeline is broken" (balacc far below the pool's
        # measured single-feature ceiling of 0.811, [F.9] §1) from "B0 is simply strong".
        lab = rec_va["label"].to_numpy()
        ba, _thr = best_balacc(prob[lab == "pos"], prob[lab == "neg"])
        return {"val_cpm": res["cpm"], "val_ceiling": res["ceiling"],
                "val_ci_lo": res["ci"]["lo"], "val_ci_hi": res["ci"]["hi"],
                "val_balacc": float(ba)}

    out_dir = encoder_dir(args, args.seed)
    row_w = record_lesion_weights(rec_tr) if C.RESC_PER_LESION_WEIGHTS else None
    result = pretrain_encoder(model_factory, loader, evaluate_epoch, out_dir, seed=args.seed,
                              epochs=args.epochs, lr=args.lr, alpha=args.alpha,
                              device=args.device, row_weights=row_w)

    # The one CI that is actually reported: the selected epoch's, computed now that we know
    # which epoch that is. Same pred, same seed, same n_boot as the old per-epoch call, so the
    # interval is identical to what the discarded-29-of-30 version would have printed.
    sel = int(result["selected_epoch"])
    best = result["epochs"][sel]
    if args.n_boot > 0:
        print(f"# CI for the SELECTED epoch only ({args.n_boot} draws; the other "
              f"{len(result['epochs']) - 1} epochs were selected on val_cpm, which needs none)",
              flush=True)
        ci_res = evaluate_variant(rec_va, probs_by_epoch[sel], gt_va,
                                  seed_tag=f"B1_seed{args.seed}", n_boot=args.n_boot)
        best["val_ci_lo"], best["val_ci_hi"] = ci_res["ci"]["lo"], ci_res["ci"]["hi"]
        # `pretrain_encoder` already wrote the table with NaN bounds; rewrite it so the
        # CSV and pretrain_report.json cannot disagree about the selected epoch.
        write_epoch_table(result["epochs"], out_dir)
    print(f"\n# [4.3] seed {args.seed}: selected epoch {result['selected_epoch']} "
          f"(val CPM {best['val_cpm']:.4f}, CI [{best['val_ci_lo']:.4f}, {best['val_ci_hi']:.4f}], "
          f"ceiling {best['val_ceiling']:.4f})")
    passed = best["val_cpm"] > b0["cpm"]
    print(f"# EXIT CHECK 4 (early read, this seed): B1 val CPM {best['val_cpm']:.4f} vs "
          f"MEASURED B0 {b0['cpm']:.4f} -> {'PASS' if passed else 'BELOW B0'}")
    print(f"#   B1 per-candidate balanced accuracy = {best['val_balacc']:.3f} "
          f"(pool single-feature ceiling on the promoted val pool: 0.811, [F.9] §1)")
    if not passed:
        print("#   TWO explanations, different remedies — do NOT jump to the fallback:")
        print("#     (a) balacc well below ~0.81 => the encoder/token/crop pipeline is broken; "
              "the pre-registered response is --encoder small_cnn_3d (spec Open escalation #2);")
        print("#     (b) balacc at/near ~0.81 => B0 is simply a strong ranking on this substrate "
              "(score_max delta 0.713, 39.8% of the volume-trust signal, [F.8]/[F.9]). That is a "
              "finding about B0, not a broken pipeline.")
    print("#   (the BINDING exit check is on the [4.6] B1's MEAN over the 3 replicas, at [4.7])")

    dump_json({"seed": args.seed, "encoder": args.encoder, "d_in": d_in,
               "rest_blocks": list(rest), "rest_names": names,
               "standardiser": {"mean": stats["mean"], "std": stats["std"]},
               "encoder_params": n_enc, "head_params": n_head,
               "b1_hidden": args.b1_hidden, "alpha": args.alpha,
               "selected_epoch": result["selected_epoch"],
               "selected_ckpt": result["selected_ckpt"],
               "selection_rule": result["selection_rule"],
               "epochs": result["epochs"]},
              out_dir / "pretrain_report.json")
    np.savez(out_dir / "standardiser.npz", mean=stats["mean"], std=stats["std"],
             names=np.array(names, dtype=object))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
