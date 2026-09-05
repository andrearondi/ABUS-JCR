"""[4.7] The whole ladder — per rung x seed CPM/ceiling/key_recall/CI, paired deltas, gates.

Reads the deployed checkpoint of every rung (from [4.3]/[4.6]), rescores each of the 3 val
seed pools **separately** (Inv. 14), and produces:

* B0 (the frozen ``score_max`` ranking) and two zero-parameter rungs built inline from the same
  pool: the **B0-spread** grid control, and **B0-rank** — within-set rank only, label-free and
  deployable, which [I3.11] measured at 0.7889 ± 0.0209 against B0' 0.7062 ± 0.0146. B0-rank is
  the floor that decides whether a trained rung was worth building; B0' alone is not;
* per rung: CPM mean ± std over the 3 replicas, each with its own volume-level bootstrap CI,
  the seven ``key_recall`` points, the full ``fp``/``recall`` arrays, and the recall ceiling;
* the pre-registered comparisons with **paired** bootstrap intervals and the fraction of
  draws favouring each side;
* the machine-checked exit gates: **pool identity** (every rung's pred CSV differs from B0's
  only in ``probability``, and ``max_recall`` is identical within a seed), **no rung exceeds
  the ceiling**, **B1 > the MEASURED B0**, and the **fairness table**.

Two gates are deliberately measured rather than compared against a stored number, because a
constant that names a substrate survives a promotion while its meaning does not:

* **exit check 4** gates B1 against B0 as computed **here, on the pool being held**. The
  version that read a hard-coded 0.5567 would have passed a B1 sitting 0.04 *below* the
  promoted B0 of 0.6327 — stale in a way that inverted the gate's effect.
* **exit check 13** (the overfitting watch) reports each rung's val-minus-train gap as an
  **excess over B0's own** val-minus-train gap. The train pool is structurally harder to rank
  than val on this substrate ([F.9] §2), so the raw gap is positive for reasons that have
  nothing to do with overfitting; B0 inherits all of them and none of the model's.

Usage:
    python scripts/phase4_eval_grid.py --device cuda --out-root ... --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.rescore.evaluate import (assert_pool_identity, b0_rank_probability,
                                       b0_spread_probability, compare_variants,
                                       evaluate_variant, score_pool, seed_summary)
from abus_jcr.rescore.variants import (COMPARISONS, COMPARISONS_FLOOR, COMPARISONS_POOLED,
                                       LADDER, LADDER_POOLED, VARIANTS, assert_fairness,
                                       fairness_table)

#: Default = the six pre-registered rungs PLUS the three pooled ones. Fixed 2026-09-03: the old
#: default read LADDER alone, so a flagless [4.7] run silently skipped every COMPARISONS_POOLED
#: entry — including FULL-P vs B2 — because absent rungs are "skipped rather than faked". A rung
#: with nothing trained is still skipped gracefully. Pinned by tests/test_froc_wiring.py.
DEFAULT_VARIANTS = tuple(v for v in tuple(LADDER) + tuple(LADDER_POOLED) if v in VARIANTS)

from _phase4_common import (add_phase4_paths, assert_device, boxes_of, dump_json, grid_dir,
                            load_deployed_model, load_deployed_report, load_variant_inputs,
                            reanchor, set_index_lists)


def dump_pred_frames(preds, seed: int, out_dir: Path):
    """Write one official-schema pred CSV per rung: ``pred_{rung}_seed{seed}.csv``.

    Phase 5's ``--dump-preds``: the stratification + curve work then runs torch-free off
    these files instead of re-loading models. Artefacts of the sanctioned evaluator
    contacts, not extra contacts (PHASE_5_SPEC §5.2).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rung, pred in preds.items():
        p = out_dir / f"pred_{rung}_seed{int(seed)}.csv"
        pred.to_csv(p, index=False)
        written.append(p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.7] evaluate the whole Phase-4 ladder")
    add_phase4_paths(ap)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"],
                    help="which frozen pool the ladder is scored on. 'test' is Phase 5's "
                         "one-touch evaluation and additionally requires --phase5-execute")
    ap.add_argument("--phase5-execute", action="store_true",
                    help="required with --eval-split test (Inv. 9 — the Phase-5 runbook is "
                         "the only sanctioned caller)")
    ap.add_argument("--dump-preds", action="store_true",
                    help="persist every rung's pred frame to grid/preds<grid-tag>/ "
                         "(Phase 5: enables torch-free stratification + curve work)")
    ap.add_argument("--grid-tag", default="",
                    help="suffix for every output file (grid<tag>.json etc). The seed-split "
                         "route runs 3 single-seed jobs concurrently ([MIG-6], 2026-09-04) — "
                         "without a tag they clobber one shared grid.json; "
                         "phase4_merge_grid.py reassembles the parts")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="marginal CI draws per rung x seed (Inv. 12; Phase-3 precedent 1000)")
    ap.add_argument("--n-boot-compare", type=int, default=1000,
                    help="PAIRED draws per comparison x seed. Each draw costs 2 oracle calls, "
                         "so this dominates the wall clock — see RB_PHASE_4 [4.7]")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.RESC_SEEDS))
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    args = ap.parse_args()
    # BEFORE anything loads: the Inv.-9 gate. assert_device and every loader come after,
    # so a mis-flagged invocation cannot touch a single byte of the test record.
    if args.eval_split == "test" and not args.phase5_execute:
        raise SystemExit("--eval-split test requires --phase5-execute (Inv. 9 — the "
                         "one-touch Phase-5 protocol); refusing before anything is loaded")
    assert_device(args.device)
    print(f"# eval split = {args.eval_split}")

    grid = {"per_rung": {}, "per_seed": {}, "comparisons": {}, "gates": {},
            "eval_split": args.eval_split}
    preds_by_seed = {}          # seed -> {rung: pred DataFrame}
    inputs_by_seed = {}

    for seed in args.seeds:
        inputs = load_variant_inputs(args, seed, C.RESC_TOKEN_BLOCKS,
                                     eval_split=args.eval_split)
        inputs_by_seed[seed] = inputs
        rec_va, gt_va = inputs["rec_va"], inputs["gt_va"]
        preds_by_seed[seed] = {}

        # --- B0: the frozen Phase-3 floor, plus the two zero-parameter rungs ------
        b0 = evaluate_variant(rec_va, rec_va["score_max"].to_numpy(float), gt_va,
                              f"B0_seed{seed}", n_boot=args.n_boot)
        preds_by_seed[seed]["B0"] = b0["pred"]
        spread = evaluate_variant(rec_va, b0_spread_probability(rec_va["score_max"].to_numpy(float)),
                                  gt_va, f"B0spread_seed{seed}", n_boot=args.n_boot)
        preds_by_seed[seed]["B0-spread"] = spread["pred"]
        # B0-rank — the REAL floor. Label-free, zero-parameter, deployable, and [I3.11]
        # measured it at 0.7889 +- 0.0209 against B0' 0.7062: a trained rung that clears B0
        # but not this has not earned its cost. Not in VARIANTS (nothing is trained) — it is
        # a scoring rule on the frozen pool, exactly like B0-spread.
        rank = evaluate_variant(rec_va, b0_rank_probability(rec_va["score_max"].to_numpy(float),
                                                            rec_va["public_id"].to_numpy()),
                                gt_va, f"B0rank_seed{seed}", n_boot=args.n_boot)
        preds_by_seed[seed]["B0-rank"] = rank["pred"]
        # B0 on the TRAIN pool too — the REFERENCE for exit check 13. The train pool is a
        # different, harder object than val (best-TP-not-rank-1 0.420 vs 0.286, [F.9] §2) and
        # is generated by weaker 80-volume fold detectors, so a positive val-minus-train gap
        # is STRUCTURAL, not overfitting. Only a rung's EXCESS over B0's own gap is evidence.
        b0_tr = evaluate_variant(inputs["rec_tr"], inputs["rec_tr"]["score_max"].to_numpy(float),
                                 inputs["gt_tr"], f"B0_train_seed{seed}", n_boot=0)
        b0["train_cpm"] = b0_tr["cpm"]
        b0["train_ceiling"] = b0_tr["ceiling"]
        grid["per_seed"].setdefault(str(seed), {})["B0"] = {k: v for k, v in b0.items() if k != "pred"}
        grid["per_seed"][str(seed)]["B0-spread"] = {k: v for k, v in spread.items() if k != "pred"}
        grid["per_seed"][str(seed)]["B0-rank"] = {k: v for k, v in rank.items() if k != "pred"}
        print(f"\n# seed {seed}: B0 CPM {b0['cpm']:.4f} [{b0['ci']['lo']:.4f}, {b0['ci']['hi']:.4f}], "
              f"ceiling {b0['ceiling']:.4f}; B0-spread CPM {spread['cpm']:.4f} "
              f"(grid artefact {spread['cpm'] - b0['cpm']:+.4f}); "
              f"B0-rank CPM {rank['cpm']:.4f} ({rank['cpm'] - b0['cpm']:+.4f} vs B0); "
              f"B0 train-pool CPM {b0_tr['cpm']:.4f} (val-train gap {b0['cpm'] - b0_tr['cpm']:+.4f})")

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
            # reanchor: dep["dir"] is the TRAINING machine's absolute path (PHASE_5_SPEC §5.1)
            trial_json = json.loads((reanchor(args, dep["dir"]) / "selection.json").read_text())
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
            print(f"  seed {seed} {variant:<5} {args.eval_split} CPM {res['cpm']:.4f} "
                  f"[{res['ci']['lo']:.4f}, {res['ci']['hi']:.4f}]  ceiling {res['ceiling']:.4f}  "
                  f"train CPM {res_tr['cpm']:.4f}")

        if args.dump_preds:
            written = dump_pred_frames(preds_by_seed[seed], seed,
                                       grid_dir(args) / f"preds{args.grid_tag}")
            print(f"  # dumped {len(written)} pred frames -> {written[0].parent}")

    # --- per-rung mean +/- std over the 3 replicas (Inv. 14) ---------------------
    lab = f"{args.eval_split} CPM"
    print(f"\n{'='*78}\n# [4.7] LADDER — CPM mean +/- std over {len(args.seeds)} replicas\n")
    print(f"  {'rung':<10} {lab:>16} {'ceiling':>16} {'train CPM':>10}")
    for rung in ["B0", "B0-spread", "B0-rank"] + list(args.variants):
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
    # COMPARISONS is the pre-registered list and stays first. COMPARISONS_POOLED and
    # COMPARISONS_FLOOR were added later and are labelled as such in the report; a rung that is
    # absent (nothing trained for it yet) is skipped rather than faked.
    print(f"\n# [4.7] COMPARISONS (paired bootstrap, per seed)\n")
    _label = ({c: "pre-registered" for c in COMPARISONS}
              | {c: "pooled-objective (added)" for c in COMPARISONS_POOLED}
              | {c: "vs the label-free floor (added)" for c in COMPARISONS_FLOOR})
    for a, b in tuple(COMPARISONS) + tuple(COMPARISONS_POOLED) + tuple(COMPARISONS_FLOOR):
        if a not in preds_by_seed[args.seeds[0]] or b not in preds_by_seed[args.seeds[0]]:
            continue
        print(f"  [{_label[(a, b)]}]")
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

    # exit check 4 — B1 must clear the MEASURED B0 of the pool actually held. Never a
    # constant (conventions.py 4 (H)): a hard-coded floor survives a substrate promotion
    # while its meaning does not, and the version that did read 0.5567 would have PASSED a
    # B1 sitting 0.04 below the promoted B0 of 0.6327.
    b0_mean = grid["per_rung"]["B0"]["cpm_mean"]
    if "B1" in grid["per_rung"]:
        b1 = grid["per_rung"]["B1"]["cpm_mean"]
        gates["b1_beats_b0"] = bool(b1 > b0_mean)
        gates["b0_cpm_mean_measured"] = float(b0_mean)
        print(f"  exit check 4 — B1 {b1:.4f} > B0 {b0_mean:.4f} (measured): "
              f"{'PASS' if gates['b1_beats_b0'] else 'BELOW B0'}")
        if not gates["b1_beats_b0"]:
            print("    TWO explanations, different remedies — do NOT jump to the fallback:")
            print("      (a) the encoder/token/crop pipeline is broken — check [4.3]'s reported "
                  "B1 balanced accuracy against this pool's single-feature ceiling (0.811 on the "
                  "promoted val pool, [F.9] §1). Well below it => spec Open escalation #2, "
                  "re-run [4.3] with --encoder small_cnn_3d.")
            print("      (b) B0 is simply a strong ranking on this substrate — the promoted "
                  "score_max separates TP/FP at delta 0.713 and carries 39.8% of the volume-trust "
                  "signal ([F.8]/[F.9]). That is a finding about B0, not a broken pipeline, and it "
                  "makes B2-B1 more interesting, not less. Report it; do not swap the encoder.")

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
    dump_json(tbl, grid_dir(args) / f"fairness{args.grid_tag}.json")

    spread_delta = [grid["per_seed"][str(s)]["B0-spread"]["cpm"] - grid["per_seed"][str(s)]["B0"]["cpm"]
                    for s in args.seeds]
    gates["b0_spread_delta_mean"] = float(np.mean(spread_delta))
    print(f"  exit check 11 — B0-spread grid artefact: {np.mean(spread_delta):+.4f} "
          f"+/- {np.std(spread_delta):.4f} (bounds the FROC quantisation effect)")

    # exit check 13 — the overfitting watch, measured RELATIVE TO B0's own val-train gap.
    # The raw gap is not evidence: the train pool is harder to rank than val (best-TP-not-
    # rank-1 0.420 vs 0.286, [F.9] §2), it comes from weaker 80-volume fold detectors
    # (Inv. 10), and its "CPM" aggregates 5 detectors' sets into one prediction frame. B0
    # inherits every one of those properties and none of the model's, so B0's gap IS the
    # structural baseline; only a rung's EXCESS over it can indicate overfitting.
    b0_gap = grid["per_rung"]["B0"]["cpm_mean"] - (grid["per_rung"]["B0"]["train_cpm_mean"] or 0.0)
    excess = {}
    for r in args.variants:
        v, t = grid["per_rung"][r]["cpm_mean"], grid["per_rung"][r].get("train_cpm_mean")
        if t is not None:
            excess[r] = float((v - t) - b0_gap)
    flagged = [r for r, e in excess.items() if e > 0.05]
    gates["overfit_excess_over_b0_gap"] = excess
    gates["overfit_b0_val_minus_train"] = float(b0_gap)
    gates["overfit_flagged"] = flagged
    remedy = ("" if not flagged else
              " — report the PHASE_4 §5 remedies explicitly: shrink to L = 2, add crop jitter, "
              "re-confirm strict OOF")
    print(f"  exit check 13 — B0's structural val-train gap = {b0_gap:+.4f}; per-rung EXCESS "
          f"over it: {', '.join(f'{r} {e:+.4f}' for r, e in excess.items()) or 'n/a'}")
    print(f"    flagged (excess > 0.05): {'none' if not flagged else flagged}{remedy}")

    grid["gates"] = gates
    dump_json(grid, grid_dir(args) / f"grid{args.grid_tag}.json")

    md = grid_dir(args) / f"grid_table{args.grid_tag}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"| rung | {args.eval_split} CPM (mean ± std) | ceiling | train CPM |",
             "|---|---|---|---|"]
    for rung in ["B0", "B0-spread", "B0-rank"] + list(args.variants):
        r = grid["per_rung"][rung]
        t = r.get("train_cpm_mean")
        lines.append(f"| {rung} | {r['cpm_mean']:.4f} ± {r['cpm_std']:.4f} | "
                     f"{r['ceiling_mean']:.4f} | {'—' if t is None else f'{t:.4f}'} |")
    md.write_text("\n".join(lines) + "\n")
    print(f"# wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
