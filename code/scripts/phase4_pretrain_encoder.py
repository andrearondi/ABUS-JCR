"""[4.3] Pretrain the shared 3D encoder + B1Head per seed — **this run IS rung B1**.

Trains ``CandidateEncoder + B1Head`` end to end on the **train** pool only (Inv. 10), with
crop augmentation (d1 flip + centre jitter, Inv. 13) applied by re-extracting from the iso
cache. Every epoch is saved; the deployed epoch is chosen **post-hoc on val CPM** through the
official oracle on the PAIRED seed pool ``full_seed{r}`` (Inv. 14) — never on val loss.

Exit check 4 is evaluated here: **B1 val CPM (mean over 3 replicas) must exceed B0 = 0.5567.**
A B1 at or below the floor means the encoder/token/crop pipeline is broken, not that
rescoring fails — stop and diagnose, then consider the pre-registered
``--encoder small_cnn_3d`` fallback (spec Open escalation #2).

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
from abus_jcr.rescore.evaluate import evaluate_variant
from abus_jcr.rescore.setmodel import B1Rescorer
from abus_jcr.rescore.train import pretrain_encoder

from _phase4_common import (add_phase4_paths, assert_device, build_features, cache_root,
                            crops_dir, dump_json, encoder_dir, gt_for_pool, iso_shape_map,
                            label_codes, load_gt, load_record, rest_blocks, val_pool_for_seed)


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
                                         shuffle=True, num_workers=args.num_workers,
                                         drop_last=False)

    d_app = int(C.RESC_D_APP)
    d_in = d_app + Ztr.shape[1]
    enc = build_encoder(args.encoder, d_app=d_app)
    head = B1Rescorer(d_in=d_in, d_model=128, hidden=args.b1_hidden, depth=2)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"# encoder {args.encoder}: {n_enc/1e6:.2f} M params; B1Head: {n_head/1e3:.1f} k params")

    def evaluate_epoch(epoch: int) -> dict:
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
        res = evaluate_variant(rec_va, prob, gt_va, seed_tag=f"B1_seed{args.seed}",
                               n_boot=args.n_boot)
        return {"val_cpm": res["cpm"], "val_ceiling": res["ceiling"],
                "val_ci_lo": res["ci"]["lo"], "val_ci_hi": res["ci"]["hi"]}

    out_dir = encoder_dir(args, args.seed)
    result = pretrain_encoder(enc, head, loader, evaluate_epoch, out_dir, seed=args.seed,
                              epochs=args.epochs, lr=args.lr, alpha=args.alpha,
                              device=args.device)

    best = result["epochs"][result["selected_epoch"]]
    print(f"\n# [4.3] seed {args.seed}: selected epoch {result['selected_epoch']} "
          f"(val CPM {best['val_cpm']:.4f}, CI [{best['val_ci_lo']:.4f}, {best['val_ci_hi']:.4f}], "
          f"ceiling {best['val_ceiling']:.4f})")
    print(f"# EXIT CHECK 4 (this seed): B1 val CPM {best['val_cpm']:.4f} vs B0 {C.RESC_B0_CPM} "
          f"-> {'PASS' if best['val_cpm'] > C.RESC_B0_CPM else 'FAIL — diagnose the pipeline'}")
    print("#   (the exit check is on the MEAN over the 3 replicas; run all seeds before deciding)")

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
