"""[4.6] Train one ``(variant, seed)`` across its **exactly 4** hyperparameter trials.

Writes per-epoch checkpoints and the per-epoch val-CPM table for every trial (the selection
provenance, exit check 8), then picks the deployed trial + epoch on val CPM — never val loss.

The trial grid is fixed by ``variants.trials_for``: ``RESC_CE_SEARCH`` for the CE rungs
(B1/B2/A1), ``RESC_LAMBDA_SEARCH`` at B2's selected α/lr for the ranking rungs (A2/FULL).
The ranking rungs therefore need ``--b2-alpha``/``--b2-lr`` from [4.6]'s B2 run.

Usage:
    python scripts/phase4_train_variant.py --variant B2 --seed 0 --device cuda --out-root ...
    python scripts/phase4_train_variant.py --variant FULL --seed 0 --b2-alpha 0.25 --b2-lr 1e-3 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from abus_jcr import conventions as C
from abus_jcr.rescore.variants import VARIANTS, trials_for

from _phase4_common import (add_phase4_paths, assert_device, dump_json, load_variant_inputs,
                            run_variant_trial, variant_dir)


def resolve_capacity(args):
    """The [4.5]-frozen capacity, or the grid's first entry if [4.5] has not run yet."""
    p = Path(args.out_root) / "capacity" / "capacity_choice.json"
    if p.exists():
        import json
        return tuple(json.loads(p.read_text())["winner"]["capacity"])
    print(f"# WARNING: {p} not found — falling back to RESC_SET_CAPACITY_GRID[0]. "
          f"Run [4.5] first so every rung shares one frozen capacity.")
    return tuple(C.RESC_SET_CAPACITY_GRID[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.6] train one (variant, seed) over its 4 trials")
    add_phase4_paths(ap)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True, choices=list(C.RESC_SEEDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=C.RESC_SET_EPOCHS)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--b2-alpha", type=float, default=None, help="B2's selected alpha (A2/FULL)")
    ap.add_argument("--b2-lr", type=float, default=None, help="B2's selected lr (A2/FULL)")
    ap.add_argument("--blocks", nargs="+", default=list(C.RESC_TOKEN_BLOCKS),
                    help="[4.8] sub-ablations only: the enabled token blocks")
    ap.add_argument("--geom-mechanism", default=None,
                    choices=["additive", "multiplicative"],
                    help="the multiplicative FALLBACK is used ONLY if A1 == B2 within CI")
    ap.add_argument("--lambda-override", type=float, default=None,
                    help="single-lambda run (e.g. the lambda=0 diagnostic); bypasses the sweep")
    ap.add_argument("--tag", default="", help="suffix for the output dir (sub-ablations)")
    args = ap.parse_args()
    assert_device(args.device)

    b2 = None
    if args.b2_alpha is not None and args.b2_lr is not None:
        b2 = {"alpha": args.b2_alpha, "lr": args.b2_lr}
    trials = trials_for(args.variant, b2_choice=b2)
    if args.lambda_override is not None:
        trials = [{**trials[0], "lam": float(args.lambda_override)}]
        print(f"# DIAGNOSTIC run at lambda = {args.lambda_override} — reported, EXCLUDED from "
              f"the selection budget (spec §4.7)")

    capacity = resolve_capacity(args)
    inputs = load_variant_inputs(args, args.seed, tuple(args.blocks))
    print(f"# [4.6] {args.variant} seed {args.seed}: {len(trials)} trials, capacity {capacity[0]}, "
          f"blocks {args.blocks}")

    results = []
    for k, trial in enumerate(trials):
        out_dir = variant_dir(args, args.variant + (f"_{args.tag}" if args.tag else ""),
                              args.seed, k)
        payload = run_variant_trial(args, args.variant, args.seed, trial, capacity, out_dir,
                                    use_blocks=tuple(args.blocks),
                                    geom_mechanism=args.geom_mechanism, device=args.device,
                                    epochs=args.epochs, n_boot=args.n_boot, inputs=inputs)
        results.append({"trial": k, "hyperparameters": trial,
                        "val_cpm": payload["selected_val_cpm"],
                        "selected_epoch": payload["selected_epoch"],
                        "selected_ckpt": payload["selected_ckpt"],
                        "params": payload["params"], "dir": str(out_dir)})
        print(f"  trial {k} {trial}: val CPM {payload['selected_val_cpm']:.4f} "
              f"(epoch {payload['selected_epoch']})")

    best = max(results, key=lambda r: r["val_cpm"])
    print(f"\n# [4.6] {args.variant} seed {args.seed} DEPLOYED: trial {best['trial']} "
          f"{best['hyperparameters']}, epoch {best['selected_epoch']}, "
          f"val CPM {best['val_cpm']:.4f}")

    at_b2 = None
    if b2 is not None and args.variant == "A1":
        match = [r for r in results if r["hyperparameters"]["alpha"] == b2["alpha"]
                 and r["hyperparameters"]["lr"] == b2["lr"]]
        at_b2 = match[0] if match else None
        if at_b2:
            print(f"# A1 at B2's alpha/lr (the CLEAN geometry isolation, used for the PRIMARY "
                  f"A1-B2 delta): val CPM {at_b2['val_cpm']:.4f}")

    dump_json({"variant": args.variant, "seed": args.seed, "capacity": list(capacity),
               "blocks": list(args.blocks), "n_trials": len(trials), "trials": results,
               "deployed": best, "a1_at_b2_config": at_b2,
               "epochs": args.epochs, "geom_mechanism": args.geom_mechanism,
               "lambda_override": args.lambda_override},
              Path(args.out_root) / "variants"
              / f"{args.variant}{('_' + args.tag) if args.tag else ''}_seed{args.seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
