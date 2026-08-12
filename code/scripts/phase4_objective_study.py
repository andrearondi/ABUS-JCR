"""[4.2b] Objective-alignment study — is the [4.3] shortfall the OBJECTIVE or the FEATURES?

`[4.2]` measured, on the seed-0 val pool: **B1 val CPM 0.5166** against a measured **B0 of
0.6673** (`[F.8]` denominator), with B1's per-candidate **balanced accuracy 0.9035** — above
the pool's 0.811 single-feature ceiling and far above `score_max`'s. B1's input vector
*contains* ``score_max`` (``SCORE_STAT_COLUMNS[0]``), so a monotone readout of one feature
would have reproduced B0 exactly. **The hypothesis class contains B0 and training moved
away from it**, which rules the features out and points at the objective.

This step decides that, cheaply, BEFORE `[4.3]` spends 3 seeds x 30 epochs of DenseNet:

* **Token-only by default** — the ``appearance`` block is dropped, so there is no encoder, no
  crop cache and no GPU requirement. That also makes this run the **appearance-free control**
  the encoder must later beat, which is the pre-registered way to decide whether a 3D
  appearance branch earns its cost (measured at ~0.57x a full-volume 3D pass per epoch, and
  ~2x a whole 30-epoch 3D detector training run over the 3 seeds).
* **A factorial over the four measured misalignments** (``rescore/objective.py``): ``gamma``
  (focal is not a proper scoring rule; `[F.8]` puts 78.7 % of the headroom in calibration),
  ``alpha`` (0.25 down-weights positives on a pool that is already 1:6.2), ``soft`` (Inv. 11's
  ignore band is a hole in the supervision that the oracle scores as an FP), ``per_lesion``
  (duplicates are free to the metric and cost ~15.6x in the loss). **The deployed cell is in
  the grid**, so every number is read against what actually ran.
* **Every variant is also scored after a rank-preserving spread** (``b0_spread_probability``,
  the exit-check-11 control). If a variant's CPM jumps under a transform that cannot change
  its ranking, the loss was fine and the oracle's fixed ``arange(0, 1, 0.005)`` sweep could
  not resolve a saturated score band — a scoring artefact, not a model failure.
* **Per variant it also reports** the three-way headroom decomposition (``volume_neutral`` /
  ``per_vol_oracle`` — calibration collapse vs within-set ranking), ``key_recall`` at all
  seven FP points (where in the curve the loss happens), the threshold occupancy, and the
  Inv.-11 ignore-band audit at the top of the ranking.

**Selection is post-hoc on val CPM per epoch, earliest within ``RESC_SELECT_CPM_TOL``** —
identical to `[4.6]`, via the same ``train_set_variant``. Nothing here is a reported result:
this is a **pre-`[4.3]` gate** whose output is a decision about the objective and about the
appearance branch. Whatever it selects must be declared as a dated, labelled forking path
(HISTORY.md §8) before `[4.3]` re-runs.

Usage (no GPU needed):
    python scripts/phase4_objective_study.py --seed 0 --device cpu \\
        --phase1-out ... --phase3-out ... --out-root ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.probe import calibration as CAL
from abus_jcr.probe.pool_diag import best_balacc
from abus_jcr.rescore.evaluate import b0_spread_probability, evaluate_variant, score_pool
from abus_jcr.rescore.objective import (ignore_band_audit, objective_grid, record_lesion_weights,
                                        record_targets, threshold_occupancy)
from abus_jcr.rescore.setmodel import B1Rescorer
from abus_jcr.rescore.train import train_set_variant

from _phase4_common import (add_phase4_paths, assert_device, boxes_of, build_features,
                            dump_json, emb_path, gt_for_pool, iso_shape_map, load_gt,
                            load_record, rest_blocks, set_batches, set_index_lists,
                            val_pool_for_seed)

_KEY_FP = C.KEY_FP


def _headroom(rec, gt, prob, tag: str) -> dict:
    """The [F.8] three-way decomposition, applied to THIS ranking instead of ``score_max``.

    ``probe.calibration`` keys its synthetic assignments off ``score_max``, so the variant's
    probability is substituted into a copy: ``volume_neutral`` then means "keep THIS model's
    within-set order, discard ALL its cross-volume confidence". A variant sitting at its own
    ``volume_neutral`` carries no usable cross-volume signal; one sitting *below* it is worse
    than discarding that signal outright.
    """
    sub = rec.copy()
    sub["score_max"] = np.asarray(prob, dtype=float)
    out = {}
    for name, p in CAL.assignments(sub).items():
        if name == "score_max":
            continue
        s = sub.copy()
        s["_p"] = np.clip(np.asarray(p, dtype=float), 0.0, 1.0 - C.RESC_PROB_EPS)
        r = evaluate_variant(s, s["_p"].to_numpy(), gt, seed_tag=f"{tag}_{name}", n_boot=0)
        out[name] = float(r["cpm"])
    return out


def _report(rec, gt, prob, tag: str, n_boot: int, with_headroom: bool = True) -> dict:
    """Score one probability assignment every way this step needs it."""
    res = evaluate_variant(rec, prob, gt, seed_tag=tag, n_boot=n_boot)
    lab = rec["label"].to_numpy()
    ba, thr = best_balacc(np.asarray(prob)[lab == "pos"], np.asarray(prob)[lab == "neg"])
    n_vol = int(rec["public_id"].nunique())
    out = {
        "tag": tag,
        "cpm": float(res["cpm"]),
        "ceiling": float(res["ceiling"]),
        "ci_lo": float(res["ci"]["lo"]), "ci_hi": float(res["ci"]["hi"]),
        "key_recall": res["key_recall"],
        "balacc": float(ba), "balacc_thresh": float(thr),
        # the FP/volume the balanced-accuracy threshold actually sits at: CPM's mass is at
        # 0.125-1 FP/vol, so a balacc optimum out at ~8 measures a regime CPM barely reads.
        "balacc_fp_per_vol": float((np.asarray(prob)[lab == "neg"] >= thr).sum()) / max(n_vol, 1),
        "threshold_occupancy": threshold_occupancy(prob),
        "prob_q": {q: float(np.quantile(prob, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
        "ignore_band": ignore_band_audit(rec, prob),
    }
    if with_headroom:
        out["headroom"] = _headroom(rec, gt, prob, tag)
    return out


def _print_row(r: dict, ref_cpm: float) -> None:
    ig = r["ignore_band"]
    print(f"  {r['tag']:<26} CPM {r['cpm']:.4f} ({r['cpm'] - ref_cpm:+.4f} vs B0)  "
          f"balacc {r['balacc']:.3f} @ {r['balacc_fp_per_vol']:.1f} FP/vol  "
          f"bins {r['threshold_occupancy']:>3}  "
          f"ign@topK {ig['ignore_in_top_k']:>3}/{ig['top_k']}  "
          f"1stTP@{ig['rank_of_first_tp']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.2b] objective-alignment study (token-only)")
    add_phase4_paths(ap)
    ap.add_argument("--seed", type=int, default=0, choices=list(C.RESC_SEEDS))
    ap.add_argument("--device", default="cpu",
                    help="cpu is enough: token-only B1 is a 32-dim MLP (default cpu)")
    ap.add_argument("--epochs", type=int, default=C.RESC_SET_EPOCHS)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=0,
                    help="draws for the SWEEP (0 = point estimates; the sweep selects, it "
                         "does not report)")
    ap.add_argument("--n-boot-final", type=int, default=200,
                    help="draws for B0 and the top-k variants — Inv. 12 for anything quoted")
    ap.add_argument("--ci-top-k", type=int, default=3)
    ap.add_argument("--variants", default="all",
                    help="'all' or a comma-separated list of variant names")
    ap.add_argument("--with-appearance", action="store_true",
                    help="add the appearance block back, reading the [4.4] embeddings at their "
                         "canonical path, so the same grid answers 'does the encoder earn its "
                         "cost'. Requires [4.4] to have run for this seed.")
    args = ap.parse_args()
    assert_device(args.device)

    rec_tr = load_record(args, "train")
    rec_va_all = load_record(args, "val")
    rec_va = val_pool_for_seed(rec_va_all, args.seed)
    gt_va = gt_for_pool(load_gt(args, "val"), rec_va)

    blocks = rest_blocks(C.RESC_TOKEN_BLOCKS)
    emb_tr = emb_va = None
    if args.with_appearance:
        # the [4.4] path, resolved by the shared helper so it cannot drift from what writes it
        p_tr, p_va = emb_path(args, "train", args.seed), emb_path(args, "val", args.seed)
        for p in (p_tr, p_va):
            if not p.exists():
                raise SystemExit(f"missing {p} — run [4.4] for seed {args.seed} first, or drop "
                                 f"--with-appearance to run the token-only reference arm")
        emb_tr = np.load(p_tr)
        emb_va_all = np.load(p_va)
        rows = rec_va_all.index[rec_va_all["detector_of_origin"]
                                == f"full_seed{args.seed}"].to_numpy()
        emb_va = emb_va_all[rows]
        blocks = tuple(C.RESC_TOKEN_BLOCKS)
        print(f"# APPEARANCE ARM: {p_tr.parent} ({emb_tr.shape[1]} dims)")

    Ztr, names, stats = build_features(rec_tr, emb_tr, blocks, iso_shape_map(args, rec_tr),
                                       stats=None)
    Zva, _, _ = build_features(rec_va, emb_va, blocks, iso_shape_map(args, rec_va), stats=stats)
    d_in = int(Ztr.shape[1])
    arm = "appearance ON" if emb_tr is not None else "token-only reference: appearance OFF"
    print(f"# blocks = {list(blocks)}  ->  d_in = {d_in}   ({arm})")
    print(f"# train {len(rec_tr)} rows / {len(set_index_lists(rec_tr))} sets;  "
          f"val seed{args.seed} {len(rec_va)} rows / {len(set_index_lists(rec_va))} sets;  "
          f"GT {len(gt_va)} lesions over {gt_va['public_id'].nunique()} volumes\n")

    # ---------------------------------------------------------------- references
    print("# ---- REFERENCES (same frozen pool, Inv. 8) ----")
    score_max = rec_va["score_max"].to_numpy(float)
    b0 = _report(rec_va, gt_va, score_max, "B0(score_max)", args.n_boot_final)
    b0s = _report(rec_va, gt_va, b0_spread_probability(score_max), "B0-spread",
                  args.n_boot_final, with_headroom=False)
    ref = b0["cpm"]
    for r in (b0, b0s):
        _print_row(r, ref)
    print(f"  B0 headroom: volume_neutral {b0['headroom']['volume_neutral']:.4f}  "
          f"per_vol_oracle {b0['headroom']['per_vol_oracle']:.4f}  "
          f"ceiling {b0['ceiling']:.4f}\n")

    # ---------------------------------------------------------------- the grid
    grid = objective_grid()
    if args.variants != "all":
        want = {s.strip() for s in args.variants.split(",")}
        grid = [c for c in grid if c["name"] in want]
    print(f"# ---- OBJECTIVE GRID ({len(grid)} cells x {args.epochs} epochs) ----")

    va_sets = set_index_lists(rec_va)
    va_coord, va_length = boxes_of(rec_va)
    Zva32 = np.ascontiguousarray(Zva, dtype=np.float32)
    out_root = Path(args.out_root) / "objective_study"
    results = []

    for cell in grid:
        targets = record_targets(rec_tr, cell["soft"])
        weights = record_lesion_weights(rec_tr) if cell["per_lesion"] else None
        model = B1Rescorer(d_in=d_in, d_model=128, hidden=256, depth=2)
        batches = set_batches(rec_tr, Ztr, seed=args.seed, labels=targets)

        def evaluate_epoch(epoch: int, _m=model) -> dict:
            prob = score_pool(_m, Zva32, va_coord, va_length, va_sets,
                              n_rows=len(rec_va), device=args.device)
            r = evaluate_variant(rec_va, prob, gt_va, seed_tag=cell["name"], n_boot=0)
            _m.train()
            return {"val_cpm": r["cpm"], "val_ceiling": r["ceiling"],
                    "val_ci_lo": float("nan"), "val_ci_hi": float("nan")}

        payload = train_set_variant(
            model, batches, evaluate_epoch, out_root / cell["name"], seed=args.seed,
            w_rank=0.0, lam=1.0, alpha=float(cell["alpha"]), lr=args.lr,
            epochs=args.epochs, device=args.device, gamma=float(cell["gamma"]),
            soft_targets=bool(cell["soft"]), row_weights=weights)

        prob = score_pool(model, Zva32, va_coord, va_length, va_sets,
                          n_rows=len(rec_va), device=args.device)
        raw = _report(rec_va, gt_va, prob, cell["name"], args.n_boot)
        spr = _report(rec_va, gt_va, b0_spread_probability(prob), cell["name"] + "+spread",
                      args.n_boot, with_headroom=False)
        rec = {**cell, "selected_epoch": payload["selected_epoch"],
               "epochs_table": payload["epochs"], "raw": raw, "spread": spr}
        results.append(rec)
        mark = "  <-- DEPLOYED CELL" if cell["is_deployed"] else ""
        print(f"\n[{cell['name']}] epoch {payload['selected_epoch']}/{args.epochs}{mark}")
        _print_row(raw, ref)
        _print_row(spr, ref)
        print(f"  headroom: volume_neutral {raw['headroom']['volume_neutral']:.4f}  "
              f"per_vol_oracle {raw['headroom']['per_vol_oracle']:.4f}")
        print("  key_recall " + "  ".join(
            f"{k}:{raw['key_recall'][str(k)]:.3f}" for k in _KEY_FP))

    # ---------------------------------------------------------------- verdict
    best = sorted(results, key=lambda r: -max(r["raw"]["cpm"], r["spread"]["cpm"]))
    dep = next((r for r in results if r["is_deployed"]), None)
    print(f"\n{'='*78}\n# [4.2b] RESULT — token-only B1 vs B0 = {ref:.4f} "
          f"(seed {args.seed}, {len(gt_va)}-lesion GT)\n")
    print(f"  {'variant':<26} {'raw':>8} {'spread':>8} {'best-B0':>9}")
    for r in best:
        b = max(r["raw"]["cpm"], r["spread"]["cpm"])
        print(f"  {r['name']:<26} {r['raw']['cpm']:>8.4f} {r['spread']['cpm']:>8.4f} "
              f"{b - ref:>+9.4f}{'   <-- DEPLOYED' if r['is_deployed'] else ''}")
    top = best[0]
    print(f"\n# BEST: {top['name']} at {max(top['raw']['cpm'], top['spread']['cpm']):.4f}")
    if dep is not None:
        d = max(top['raw']['cpm'], top['spread']['cpm']) - max(dep['raw']['cpm'],
                                                               dep['spread']['cpm'])
        print(f"# vs the DEPLOYED cell ({dep['name']}): {d:+.4f}")
    print("#\n# READ THIS BEFORE ACTING — three separable questions, three separate answers:\n"
          "#  1. spread >> raw anywhere  => the oracle's fixed 0.005 sweep could not resolve a\n"
          "#     saturated score band. That is a SCORING artefact, not a model failure, and it\n"
          "#     applies to every rung: report B0-spread beside it (exit check 11).\n"
          "#  2. best CPM still <= B0    => a per-candidate rescorer cannot beat this baseline\n"
          "#     on a metric that is 78.7 % cross-volume ([F.8]). That is a finding about the\n"
          "#     TASK, and it makes exit check 4's premise ('any rescoring => B1 > B0')\n"
          "#     questionable: cross-volume calibration needs the SET, i.e. B2, not B1.\n"
          "#  3. token-only ~= the appearance arm => the encoder is not earning ~2x a full 3D\n"
          "#     detector training run. Decide the appearance branch on THAT number.\n"
          "# None of this is a reported result. Whatever is promoted is a DATED, LABELLED\n"
          "# forking path chosen after seeing [4.2] — say so in the report (HISTORY.md §8).")

    dump_json({"seed": args.seed, "blocks": list(blocks), "d_in": d_in,
               "n_gt_lesions": int(len(gt_va)),
               "b0": b0, "b0_spread": b0s, "epochs": args.epochs,
               "feature_names": names, "results": results},
              out_root / f"objective_study_seed{args.seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
