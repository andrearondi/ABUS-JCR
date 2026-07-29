"""[4.5] ONE-TIME set-module capacity selection, on **B2**, seed 0 — then FROZEN.

Sweeps ``RESC_SET_CAPACITY_GRID`` (2 configs) at B2's first CE trial and freezes the winner
on val CPM. Selecting on the *baseline* is deliberate: it can only make B2 stronger, never
handicap it — the do-not-drift rule is "never make the independent baseline weaker than the
joint variants". The winner is then used by **every** set rung, matched by ``B1Head``'s
parameter count, and reused unchanged in Phase 6.

Usage:
    python scripts/phase4_select_capacity.py --device cuda --out-root ...
"""

from __future__ import annotations

import argparse

from abus_jcr import conventions as C
from abus_jcr.rescore.variants import trials_for

from _phase4_common import (add_phase4_paths, assert_device, dump_json, load_variant_inputs,
                            run_variant_trial, set_module_params)


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.5] freeze the set-module capacity on B2 val CPM")
    add_phase4_paths(ap)
    ap.add_argument("--seed", type=int, default=0, help="one-time selection runs on seed 0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=C.RESC_SET_EPOCHS)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    assert_device(args.device)

    inputs = load_variant_inputs(args, args.seed, C.RESC_TOKEN_BLOCKS)
    trial = trials_for("B2")[0]
    print(f"# [4.5] capacity sweep on B2, seed {args.seed}, trial {trial}")

    results = []
    for capacity in C.RESC_SET_CAPACITY_GRID:
        tag = capacity[0]
        out_dir = (__import__("pathlib").Path(args.out_root) / "capacity" / f"B2_{tag}")
        payload = run_variant_trial(args, "B2", args.seed, trial, capacity, out_dir,
                                    device=args.device, epochs=args.epochs,
                                    n_boot=args.n_boot, inputs=inputs)
        results.append({"tag": tag, "capacity": list(capacity),
                        "val_cpm": payload["selected_val_cpm"],
                        "selected_epoch": payload["selected_epoch"],
                        "params": payload["params"]})
        print(f"  {tag}: val CPM {payload['selected_val_cpm']:.4f} "
              f"(epoch {payload['selected_epoch']}, {payload['params']} params)")

    winner = max(results, key=lambda r: r["val_cpm"])
    d_in = inputs["d_in"]
    ref_params = set_module_params(d_in, tuple(winner["capacity"]), use_geometry=False)
    from abus_jcr.rescore.variants import b1_param_count, match_b1_capacity
    hidden = match_b1_capacity(d_in, winner["capacity"][2], ref_params)
    b1_params = b1_param_count(d_in, winner["capacity"][2], hidden)

    print(f"\n# [4.5] FROZEN capacity = {winner['tag']} "
          f"(L={winner['capacity'][1]}, H={winner['capacity'][2]}, heads={winner['capacity'][3]})")
    print(f"# fairness contract: set module {ref_params} params -> B1 hidden {hidden} "
          f"({b1_params} params, {abs(b1_params-ref_params)/ref_params:.1%} off)")
    print("# This capacity is now CONSTANT for B2/A1/A2/FULL, matched by B1, and reused in Phase 6.")

    dump_json({"grid": results, "winner": winner, "d_in": d_in,
               "reference_params": ref_params, "b1_hidden": hidden, "b1_params": b1_params,
               "selected_on": f"B2 val CPM, seed {args.seed}", "trial": trial},
              __import__("pathlib").Path(args.out_root) / "capacity" / "capacity_choice.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
