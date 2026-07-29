"""[4.7] The whole ladder — per rung x seed CPM/ceiling/key_recall/CI, paired deltas, gates.

Reads the deployed checkpoint of every rung (from [4.3]/[4.6]), rescores each of the 3 val
seed pools **separately** (Inv. 14), and produces:

* B0 (the frozen ``score_max`` ranking) and the **B0-spread** control;
* per rung: CPM mean ± std over the 3 replicas, each with its own volume-level bootstrap CI,
  the seven ``key_recall`` points, the full ``fp``/``recall`` arrays, and the recall ceiling;
* the pre-registered comparisons with **paired** bootstrap intervals and the fraction of
  draws favouring each side;
* the machine-checked exit gates: **pool identity** (every rung's pred CSV differs from B0's
  only in ``probability``, and ``max_recall`` is identical within a seed), **no rung exceeds
  the ceiling**, **B1 > B0**, and the **fairness table**.

Train-pool CPM is reported next to val CPM per rung (exit check 13, the overfitting watch).

Usage:
    python scripts/phase4_eval_grid.py --device cuda --out-root ... --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.rescore.evaluate import (assert_pool_identity, b0_spread_probability,
                                       compare_variants, evaluate_variant, score_pool,
                                       seed_summary)
from abus_jcr.rescore.variants import COMPARISONS, LADDER, VARIANTS, assert_fairness, fairness_table

from _phase4_common import (add_phase4_paths, assert_device, boxes_of, dump_json, grid_dir,
                            load_deployed_model, load_deployed_report, load_variant_inputs,
                            set_index_lists)


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.7] evaluate the whole Phase-4 ladder")
    add_phase4_paths(ap)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="marginal CI draws per rung x seed (Inv. 12; Phase-3 precedent 1000)")
    ap.add_argument("--n-boot-compare", type=int, default=1000,
                    help="PAIRED draws per comparison x seed. Each draw costs 2 oracle calls, "
                         "so this dominates the wall clock — see RB_PHASE_4 [4.7]")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.RESC_SEEDS))
    ap.add_argument("--variants", nargs="+", default=[v for v in LADDER if v in VARIANTS])
    args = ap.parse_args()
    assert_device(args.device)

    grid = {"per_rung": {}, "per_seed": {}, "comparisons": {}, "gates": {}}
    preds_by_seed = {}          # seed -> {rung: pred DataFrame}
    inputs_by_seed = {}

    for seed in args.seeds:
        inputs = load_variant_inputs(args, seed, C.RESC_TOKEN_BLOCKS)
        inputs_by_seed[seed] = inputs
        rec_va, gt_va = inputs["rec_va"], inputs["gt_va"]
        preds_by_seed[seed] = {}

        # --- B0: the frozen Phase-3 floor, and the B0-spread control -------------
        b0 = evaluate_variant(rec_va, rec_va["score_max"].to_numpy(float), gt_va,
                              f"B0_seed{seed}", n_boot=args.n_boot)
        preds_by_seed[seed]["B0"] = b0["pred"]
        spread = evaluate_variant(rec_va, b0_spread_probability(rec_va["score_max"].to_numpy(float)),
                                  gt_va, f"B0spread_seed{seed}", n_boot=args.n_boot)
        preds_by_seed[seed]["B0-spread"] = spread["pred"]
        grid["per_seed"].setdefault(str(seed), {})["B0"] = {k: v for k, v in b0.items() if k != "pred"}
        grid["per_seed"][str(seed)]["B0-spread"] = {k: v for k, v in spread.items() if k != "pred"}
        print(f"\n# seed {seed}: B0 CPM {b0['cpm']:.4f} [{b0['ci']['lo']:.4f}, {b0['ci']['hi']:.4f}], "
              f"ceiling {b0['ceiling']:.4f}; B0-spread CPM {spread['cpm']:.4f} "
              f"(grid artefact {spread['cpm'] - b0['cpm']:+.4f})")

        # --- the trained rungs ---------------------------------------------------
        va_sets = set_index_lists(rec_va)
        va_coord, va_length = boxes_of(rec_va)
        Zva32 = np.ascontiguousarray(inputs["Zva"], dtype=np.float32)
        Ztr32 = np.ascontiguousarray(inputs["Ztr"], dtype=np.float32)
        tr_sets = set_index_lists(inputs["rec_tr"])
        tr_coord, tr_length = boxes_of(inputs["rec_tr"])

        for variant in args.variants:
            model, rep = load_deployed_model(args, variant, seed, inputs["d_in"], args.device)
            dep = rep["deployed"]
            trial_json = json.loads((Path(dep["dir"]) / "selection.json").read_text())
            prob = score_pool(model, Zva32, va_coord, va_length, va_sets,
                              n_rows=len(rec_va), device=args.device)
            res = evaluate_variant(rec_va, prob, gt_va, f"{variant}_seed{seed}", n_boot=args.n_boot)
            preds_by_seed[seed][variant] = res["pred"]

            prob_tr = score_pool(model, Ztr32, tr_coord, tr_length, tr_sets,
                                 n_rows=len(inputs["rec_tr"]), device=args.device)
            # train-pool CPM is the overfitting WATCH (exit check 13), not a reported
            # metric, so it carries no CI — Inv. 12 applies to the val/test numbers.
            res_tr = evaluate_variant(inputs["rec_tr"], prob_tr, inputs["gt_tr"],
                                      f"{variant}_train_seed{seed}", n_boot=0)
            grid["per_seed"][str(seed)][variant] = {
                **{k: v for k, v in res.items() if k != "pred"},
                "train_cpm": res_tr["cpm"], "train_ceiling": res_tr["ceiling"],
                "deployed": {k: v for k, v in dep.items() if k != "dir"},
                "hyperparameters": trial_json.get("hyperparameters"),
            }
            print(f"  seed {seed} {variant:<5} val CPM {res['cpm']:.4f} "
                  f"[{res['ci']['lo']:.4f}, {res['ci']['hi']:.4f}]  ceiling {res['ceiling']:.4f}  "
                  f"train CPM {res_tr['cpm']:.4f}")

    # --- per-rung mean +/- std over the 3 replicas (Inv. 14) ---------------------
    print(f"\n{'='*78}\n# [4.7] LADDER — CPM mean +/- std over {len(args.seeds)} replicas\n")
    print(f"  {'rung':<10} {'val CPM':>16} {'ceiling':>16} {'train CPM':>10}")
    for rung in ["B0", "B0-spread"] + list(args.variants):
        per = [grid["per_seed"][str(s)][rung] for s in args.seeds]
        summ = seed_summary(per)
        tr = [p.get("train_cpm") for p in per if p.get("train_cpm") is not None]
        grid["per_rung"][rung] = {**summ,
                                  "train_cpm_mean": float(np.mean(tr)) if tr else None,
                                  "key_recall": per[0]["key_recall"]}
        print(f"  {rung:<10} {summ['cpm_mean']:.4f} +/- {summ['cpm_std']:.4f}  "
              f"{summ['ceiling_mean']:.4f} +/- {summ['ceiling_std']:.4f}  "
              f"{(np.mean(tr) if tr else float('nan')):.4f}")

    # --- pre-registered comparisons, PAIRED intervals ----------------------------
    print(f"\n# [4.7] PRE-REGISTERED COMPARISONS (paired bootstrap, per seed)\n")
    for a, b in COMPARISONS:
        if a not in preds_by_seed[args.seeds[0]] or b not in preds_by_seed[args.seeds[0]]:
            continue
        rows = []
        for seed in args.seeds:
            cmpres = compare_variants(inputs_by_seed[seed]["gt_va"], preds_by_seed[seed][a],
                                      preds_by_seed[seed][b], a, b,
                                      n_boot=args.n_boot_compare, seed=0)
            rows.append(cmpres)
            print(f"  seed {seed}  {a} - {b}: {cmpres['delta']:+.4f} "
                  f"[{cmpres['lo']:+.4f}, {cmpres['hi']:+.4f}]  "
                  f"frac favouring {a} = {cmpres['frac_positive']:.3f}")
        grid["comparisons"][f"{a}-{b}"] = {
            "per_seed": rows,
            "delta_mean": float(np.mean([r["delta"] for r in rows])),
            "delta_std": float(np.std([r["delta"] for r in rows])),
        }
        print(f"    -> mean delta {grid['comparisons'][f'{a}-{b}']['delta_mean']:+.4f} "
              f"+/- {grid['comparisons'][f'{a}-{b}']['delta_std']:.4f}\n")

    # --- exit gates --------------------------------------------------------------
    print(f"{'='*78}\n# [4.7] EXIT GATES\n")
    gates = {}

    ident_ok, ceil_ok = True, True
    for seed in args.seeds:
        base = preds_by_seed[seed]["B0"]
        base_ceiling = grid["per_seed"][str(seed)]["B0"]["ceiling"]
        for rung, pred in preds_by_seed[seed].items():
            try:
                assert_pool_identity(base, pred)
            except AssertionError as e:
                ident_ok = False
                print(f"  FAIL pool identity, seed {seed}, {rung}: {e}")
            c = grid["per_seed"][str(seed)][rung]["ceiling"]
            if abs(c - base_ceiling) > 1e-9:
                ceil_ok = False
                print(f"  FAIL ceiling drift, seed {seed}, {rung}: {c} vs B0 {base_ceiling}")
            if grid["per_seed"][str(seed)][rung]["cpm"] > c + 1e-9:
                ceil_ok = False
                print(f"  FAIL CPM above ceiling, seed {seed}, {rung}")
    gates["pool_identity"] = ident_ok
    gates["ceiling_invariant_and_respected"] = ceil_ok
    print(f"  exit check 6 — pool identity + rung-invariant ceiling: "
          f"{'PASS' if ident_ok and ceil_ok else 'FAIL'}")

    if "B1" in grid["per_rung"]:
        b1 = grid["per_rung"]["B1"]["cpm_mean"]
        gates["b1_beats_b0"] = bool(b1 > C.RESC_B0_CPM)
        verdict = ("PASS" if gates["b1_beats_b0"] else
                   "FAIL — the encoder/token/crop pipeline is broken, not rescoring. "
                   "Stop and diagnose (spec Open escalation #2).")
        print(f"  exit check 4 — B1 {b1:.4f} > B0 {C.RESC_B0_CPM}: {verdict}")

    try:
        reports = {v: load_deployed_report(args, v, args.seeds[0]) for v in args.variants}
        tbl = fairness_table(
            params={v: reports[v]["trials"][0]["params"] for v in args.variants},
            epochs={v: reports[v]["epochs"] for v in args.variants},
            trials={v: reports[v]["n_trials"] for v in args.variants},
            reference="B2")
        assert_fairness(tbl)
        gates["fairness"] = True
        print(f"  exit check 5 — fairness contract: PASS "
              f"(B1 {tbl['b1_rel_error']:.1%} off the reference set module)")
    except (AssertionError, KeyError) as e:
        gates["fairness"] = False
        tbl = {"error": str(e)}
        print(f"  exit check 5 — fairness contract: FAIL ({e})")
    dump_json(tbl, grid_dir(args) / "fairness.json")

    spread_delta = [grid["per_seed"][str(s)]["B0-spread"]["cpm"] - grid["per_seed"][str(s)]["B0"]["cpm"]
                    for s in args.seeds]
    gates["b0_spread_delta_mean"] = float(np.mean(spread_delta))
    print(f"  exit check 11 — B0-spread grid artefact: {np.mean(spread_delta):+.4f} "
          f"+/- {np.std(spread_delta):.4f} (bounds the FROC quantisation effect)")

    over = {r: (grid["per_rung"][r]["cpm_mean"], grid["per_rung"][r].get("train_cpm_mean"))
            for r in args.variants}
    flagged = [r for r, (v, t) in over.items() if t is not None and v > t + 0.05]
    gates["overfit_flagged"] = flagged
    remedy = ("" if not flagged else
              " — report the PHASE_4 §5 remedies explicitly: shrink to L = 2, add crop jitter, "
              "re-confirm strict OOF")
    print(f"  exit check 13 — val far above train pool: "
          f"{'none' if not flagged else flagged}{remedy}")

    grid["gates"] = gates
    dump_json(grid, grid_dir(args) / "grid.json")

    md = grid_dir(args) / "grid_table.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| rung | val CPM (mean ± std) | ceiling | train CPM |", "|---|---|---|---|"]
    for rung in ["B0", "B0-spread"] + list(args.variants):
        r = grid["per_rung"][rung]
        t = r.get("train_cpm_mean")
        lines.append(f"| {rung} | {r['cpm_mean']:.4f} ± {r['cpm_std']:.4f} | "
                     f"{r['ceiling_mean']:.4f} | {'—' if t is None else f'{t:.4f}'} |")
    md.write_text("\n".join(lines) + "\n")
    print(f"# wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
