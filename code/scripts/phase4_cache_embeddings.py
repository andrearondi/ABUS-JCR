"""[4.4] Freeze the encoder and cache the un-augmented appearance embeddings ``a_i``.

Reads the [4.3]-selected encoder checkpoint for seed ``r`` and writes row-aligned
``emb_{split}_seed{r}.npy`` for train and val. **Un-augmented, no TTA** (Inv. 13).

This is what makes the ablation honest: every set-module variant (B2/A1/A2/FULL) reads the
SAME ``a_i``, so B2-vs-B1, A1-vs-B2 and FULL-vs-B2 isolate the set module and never the
encoder — and each set-module run then costs seconds instead of GPU-hours.

Embeddings are computed for **every** val row (not just seed ``r``'s) so the array is
literally row-aligned to the record; the extra work is a few seconds on an A6000 and it
removes a whole class of off-by-one join bugs.

Usage:
    python scripts/phase4_cache_embeddings.py --seed 0 --device cuda --out-root ...
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from abus_jcr import conventions as C
from abus_jcr.rescore.crops import open_crop_cache
from abus_jcr.rescore.encoder import build_encoder

from _phase4_common import (add_phase4_paths, assert_device, crops_dir, dump_json,
                            emb_path, emb_report_path, encoder_dir, load_record)


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.4] cache the frozen encoder embeddings")
    add_phase4_paths(ap)
    ap.add_argument("--seed", type=int, required=True, choices=list(C.RESC_SEEDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--ckpt", default=None,
                    help="override the [4.3]-selected checkpoint (default: read the report)")
    args = ap.parse_args()
    assert_device(args.device)

    edir = encoder_dir(args, args.seed)
    import json
    report = json.loads((edir / "pretrain_report.json").read_text())
    ckpt_path = args.ckpt or report["selected_ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"# seed {args.seed}: encoder from {ckpt_path} (epoch {ckpt['epoch']}, "
          f"{ckpt.get('encoder_name', C.RESC_ENCODER)})")

    enc = build_encoder(ckpt.get("encoder_name", C.RESC_ENCODER), d_app=C.RESC_D_APP)
    enc.load_state_dict(ckpt["encoder"])
    enc = enc.to(args.device).eval()
    for p in enc.parameters():                 # FROZEN from here on
        p.requires_grad_(False)

    written = {}
    for split in args.splits:
        rec = load_record(args, split)
        crops, meta = open_crop_cache(crops_dir(args), split)
        if int(meta["n_rows"]) != len(rec):
            raise SystemExit(f"crop cache has {meta['n_rows']} rows, record has {len(rec)}; "
                             f"rebuild [4.1] against this record")
        out = np.zeros((len(rec), C.RESC_D_APP), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(rec), args.batch):
                x = torch.from_numpy(np.ascontiguousarray(
                    crops[s:s + args.batch][:, None], dtype=np.float32)).to(args.device)
                out[s:s + args.batch] = enc(x).cpu().numpy()
                if (s // args.batch) % 20 == 0:
                    print(f"  {split}: {s}/{len(rec)}", flush=True)
        path = emb_path(args, split, args.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
        written[split] = {"path": str(path), "shape": list(out.shape),
                          "mean_abs": float(np.abs(out).mean()),
                          "frac_zero": float((out == 0).mean())}
        print(f"# wrote {path} {out.shape} (mean|a| {written[split]['mean_abs']:.4f}, "
              f"zero frac {written[split]['frac_zero']:.3f})")

    # The report shares `emb_path`'s suffix (see `emb_report_path`): two checkpoints of one seed
    # must leave two records, or the canonical arrays end up documented as the alternate epoch's.
    dump_json({"seed": args.seed, "encoder_ckpt": str(ckpt_path),
               "encoder_epoch": int(ckpt["epoch"]), "crop_hash": meta["crop_hash"],
               "preprocess_hash": meta["preprocess_hash"], "augmented": False,
               "emb_suffix": getattr(args, "emb_suffix", "") or "",
               "splits": written},
              emb_report_path(args, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
