"""[4.10] Cost record — params / GFLOPs / latency (do-not-drift #16, exit check 12).

Encoder cost at ``(1, 1, 48, 48, 48)``; set-module cost at the PROMOTED pool's val **median
(85)**, val **max (292)** and overall **max (509)** set sizes (``cost.COSTED_SET_SIZES``);
per-set rescoring latency (mean ± std over 50 timed forwards); per-volume crop-extraction
latency. Appended next to the recorded Phase-3 detector cost ([P3U3.5]: 6.688 s/vol detector
+ 0.364 s/vol linking + 0.019 s/vol aggregation, 36.34 M params / 22.64 GFLOPs) so the
Phase-5 cost table reads end to end.

⚠ ``[F.11e]`` re-measured the detector at **16.7 s/vol** on the promoted arm and flagged it
as **machine-load noise, to be re-measured for the Phase-5 cost table**. The 6.688 s/vol
figure above is the [P3U3.5] measurement and is the one to quote until then; the
architecture is unchanged, so params/GFLOPs do not move either way.

Usage:
    python scripts/phase4_cost.py --seed 0 --device cuda --out-root ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.cache import open_vol
from abus_jcr.rescore.cost import (COSTED_SET_SIZES, crop_extraction_cost, encoder_cost,
                                   setmodule_cost, write_cost)
from abus_jcr.rescore.encoder import build_encoder
from abus_jcr.rescore.setmodel import build_rescorer

from _phase4_common import (add_phase4_paths, assert_device, b1_hidden_for, cache_root,
                            load_deployed_report, load_variant_inputs, set_module_params)


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.10] persist the Phase-4 cost record")
    add_phase4_paths(ap)
    ap.add_argument("--seed", type=int, default=0, choices=list(C.RESC_SEEDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=50, help="timed forwards per measurement")
    ap.add_argument("--crop-k", type=int, default=5, help="repeats for the crop-extraction timing")
    args = ap.parse_args()
    assert_device(args.device)

    inputs = load_variant_inputs(args, args.seed, C.RESC_TOKEN_BLOCKS)
    d_in = inputs["d_in"]

    try:
        capacity = tuple(load_deployed_report(args, "B2", args.seed)["capacity"])
    except SystemExit:
        capacity = tuple(C.RESC_SET_CAPACITY_GRID[0])
        print(f"# [4.6] B2 report absent — costing the grid's first capacity {capacity[0]}")

    print(f"# [4.10] encoder cost ({C.RESC_ENCODER}) at (1,1,{C.RESC_CROP_OUT}^3)")
    enc = build_encoder(C.RESC_ENCODER, d_app=C.RESC_D_APP)
    enc_rec = encoder_cost(enc, device=args.device, k=args.k)
    print(f"  params {enc_rec['params_total']/1e6:.2f} M, GFLOPs {enc_rec['gflops']:.3f}, "
          f"latency {enc_rec['latency']['ms_mean']:.2f} +/- {enc_rec['latency']['ms_std']:.2f} ms")

    set_rec = {}
    for variant in ("B1", "B2", "FULL"):
        hidden = b1_hidden_for(args, d_in, capacity) if variant == "B1" else None
        model = build_rescorer(variant, d_in, capacity, hidden=hidden)
        use_geom = variant == "FULL"
        rec = setmodule_cost(model, d_in, device=args.device, k=args.k, use_geometry=use_geom)
        set_rec[variant] = rec
        print(f"# {variant}: {rec['params_total']/1e6:.3f} M params")
        for n, e in rec["per_set"].items():
            print(f"    set size {n:>4}: GFLOPs {e['gflops']:.4f}, "
                  f"latency {e['latency']['ms_mean']:.2f} +/- {e['latency']['ms_std']:.2f} ms")

    rec_va = inputs["rec_va"]
    pid = int(rec_va["public_id"].value_counts().idxmax())
    rows = rec_va[rec_va["public_id"] == pid]
    print(f"# crop extraction on volume {pid} ({len(rows)} candidates)")
    vol = np.asarray(open_vol(cache_root(args), pid))
    crop_rec = crop_extraction_cost(vol, rows, k=args.crop_k)
    print(f"  {crop_rec['seconds_per_volume_mean']:.3f} +/- {crop_rec['seconds_per_volume_std']:.3f} "
          f"s/vol ({crop_rec['ms_per_candidate']:.2f} ms/candidate)")

    payload = {"seed": args.seed, "capacity": list(capacity), "d_in": d_in,
               "set_sizes_costed": list(COSTED_SET_SIZES),
               "encoder": enc_rec, "set_modules": set_rec,
               "crop_extraction": crop_rec,
               "reference_set_params": set_module_params(d_in, capacity)}
    path = write_cost(payload, Path(args.out_root) / "cost")
    print(f"\n# wrote {path}")
    print("# Read against [P3U3.5]: detector 6.688 s/vol + linking 0.364 s/vol. The rescorer's "
          "per-volume cost is crop extraction + one encoder pass per candidate + one set forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
