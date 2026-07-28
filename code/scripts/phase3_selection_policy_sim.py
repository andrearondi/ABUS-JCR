"""[P3U2.SIM] Checkpoint-selection POLICY SIMULATION — READ-ONLY, changes nothing.

Replays candidate selection policies over the per-epoch tables ALREADY recorded in
``outputs/phase3/selection/select_<run>.json`` (written by ``phase2_select_checkpoint.py``). No GPU, no
re-detection, no retraining; it writes no conventions and deploys no checkpoint. It only prints what each
policy WOULD have picked, so the Inv.-2 / A1 question ("CPM-first vs recall-first") is decided on evidence.

Why this is answerable offline: each selection JSON already stores, for EVERY epoch, the linked val
``cpm``, the recall ``ceiling``, and ``cands_per_vol`` — computed through the real detect->link->oracle
path under the deployed aggregation. The policy is just an argmax over that table.

Policies simulated
  current            the production rule (Inv. 2 / A1 revised): max CPM, ties within `cpm_tol` broken on
                     highest ceiling then earliest epoch. Uses the SAME function production uses.
  cpm_argmax         bare max CPM (no ceiling tie-break) — the pre-2026-07-19 behaviour, for contrast.
  recall_floor@F     among epochs with ceiling >= F AND pool <= budget: max CPM. Fallback if none clears:
                     max ceiling. (Mirrors the A2 operating-point rule's shape: floor on recall, then rank.)
  a2_shaped          among epochs with ceiling >= 0.98 * (that run's max ceiling) and pool <= budget: max CPM.
  recall_primary     max ceiling, CPM tie-break (the naive "recall-first") — included to EXPOSE its
                     noise-maximisation risk, not as a recommendation.

Every policy respects the run's own ``min_epoch`` floor.

Usage (server or laptop, wherever the selection JSONs are):
    python scripts/phase3_selection_policy_sim.py \
        --selection-dir /home/maia-user/Andre2/outputs/phase3/selection --pool-budget 230
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.detect.select import select_epoch

FOLD_PREFIX, SEED_PREFIX = "retinanet_fold", "retinanet_full_seed"


# ----------------------------------------------------------------------------- load
def _load_runs(sel_dir: Path) -> dict:
    """{run: {"min_epoch": int, "n_val": int, "sel": int, "ci": dict, "ep": {epoch: row}}}."""
    runs = {}
    for p in sorted(sel_dir.glob("select_*.json")):
        d = json.loads(p.read_text())
        ep = {int(k): v for k, v in d.get("per_epoch", {}).items()}
        if not ep:
            print(f"  (skip {p.name}: no per_epoch table)")
            continue
        runs[d.get("run", p.stem.replace("select_", ""))] = {
            "min_epoch": int(d.get("min_epoch", C.DET_SELECT_MIN_EPOCH)),
            "n_val": int(d.get("n_val", 0)), "sel": d.get("selected_epoch"),
            "ci": d.get("ci", {}), "ep": ep,
        }
    return runs


def _regime(run: str) -> str:
    return "FOLD" if run.startswith(FOLD_PREFIX) else ("SEED" if run.startswith(SEED_PREFIX) else "OTHER")


def _f(v, default=float("nan")):
    return default if v is None else float(v)


# ----------------------------------------------------------------------------- policies
def _eligible(info: dict, pool_budget=None):
    """Epochs >= min_epoch with a finite CPM (and pool <= budget when a budget is given)."""
    out = {}
    for e, r in info["ep"].items():
        if e < info["min_epoch"]:
            continue
        cpm = _f(r.get("cpm"))
        if np.isnan(cpm):
            continue
        if pool_budget is not None and _f(r.get("cands_per_vol"), 0.0) > pool_budget:
            continue
        out[e] = r
    return out


def pick_current(info: dict, **_) -> int:
    cands = _eligible(info)                       # production rule ignores the pool budget
    return select_epoch({e: _f(r["cpm"]) for e, r in cands.items()}, info["min_epoch"],
                        {e: _f(r.get("ceiling")) for e, r in cands.items()}, C.DET_SELECT_CPM_TOL)


def pick_cpm_argmax(info: dict, **_) -> int:
    cands = _eligible(info)
    return int(max(cands, key=lambda e: (_f(cands[e]["cpm"]), -e)))


def pick_recall_floor(info: dict, floor: float, pool_budget=None, **_) -> int:
    cands = _eligible(info, pool_budget)
    ok = {e: r for e, r in cands.items() if _f(r.get("ceiling"), -1) >= floor}
    if ok:                                        # floor cleared -> best ranking among them
        return int(max(ok, key=lambda e: (_f(ok[e]["cpm"]), -e)))
    if not cands:                                 # budget removed everything -> ignore budget
        cands = _eligible(info)
    return int(max(cands, key=lambda e: (_f(cands[e].get("ceiling"), -1), _f(cands[e]["cpm"]), -e)))


def pick_a2_shaped(info: dict, pool_budget=None, frac: float = 0.98, **_) -> int:
    cands = _eligible(info, pool_budget) or _eligible(info)
    max_ceil = max((_f(r.get("ceiling"), -1) for r in cands.values()), default=-1)
    return pick_recall_floor(info, frac * max_ceil, pool_budget)


def pick_recall_primary(info: dict, pool_budget=None, **_) -> int:
    cands = _eligible(info, pool_budget) or _eligible(info)
    return int(max(cands, key=lambda e: (_f(cands[e].get("ceiling"), -1), _f(cands[e]["cpm"]), -e)))


# ----------------------------------------------------------------------------- report
def _row(info: dict, e: int) -> dict:
    r = info["ep"][e]
    return {"epoch": e, "cpm": _f(r.get("cpm")), "ceiling": _f(r.get("ceiling")),
            "pool": _f(r.get("cands_per_vol"))}


def _tied_block(info: dict, tol: float):
    """(#epochs statistically tied with the CPM max, min/max ceiling among them) — how arbitrary the pick is."""
    cands = _eligible(info)
    if not cands:
        return 0, float("nan"), float("nan")
    mx = max(_f(r["cpm"]) for r in cands.values())
    band = [e for e, r in cands.items() if _f(r["cpm"]) >= mx - tol]
    ceils = [_f(cands[e].get("ceiling"), np.nan) for e in band]
    return len(band), float(np.nanmin(ceils)), float(np.nanmax(ceils))


def main() -> int:
    ap = argparse.ArgumentParser(description="[P3U2.SIM] selection-policy simulation (read-only)")
    ap.add_argument("--selection-dir", default="/home/maia-user/Andre2/outputs/phase3/selection")
    ap.add_argument("--pool-budget", type=float, default=float(C.RESCORER_POOL_BUDGET),
                    help=f"max cands/vol a policy may accept (default RESCORER_POOL_BUDGET={C.RESCORER_POOL_BUDGET})")
    ap.add_argument("--floors", default="0.80,0.85,0.90", help="comma-separated recall floors to simulate")
    ap.add_argument("--out-json", default=None, help="optional path to write the simulation summary")
    args = ap.parse_args()

    sel_dir = Path(args.selection_dir)
    runs = _load_runs(sel_dir)
    if not runs:
        raise SystemExit(f"no select_*.json with a per_epoch table under {sel_dir}")
    floors = [float(x) for x in args.floors.split(",") if x.strip()]

    print("=" * 100)
    print("# [P3U2.SIM] CHECKPOINT-SELECTION POLICY SIMULATION — READ-ONLY (deploys nothing, writes no conventions)")
    print(f"  runs={len(runs)}  cpm_tol={C.DET_SELECT_CPM_TOL}  pool_budget={args.pool_budget:.0f}  src={sel_dir}\n")

    # ---- A. how arbitrary is the current pick? -------------------------------------------------
    print("# A. IS THE CURRENT PICK RESOLVED BY THE DATA?  (epochs statistically tied with the CPM max)\n")
    print(f"  {'run':>22} {'sel':>4} {'CPM':>7} {'ceil':>6} {'#tied':>6} {'ceiling range (tied)':>22} {'CI width':>9}")
    for run, info in runs.items():
        e = pick_current(info)
        r = _row(info, e)
        n_tied, lo, hi = _tied_block(info, C.DET_SELECT_CPM_TOL)
        ci = info.get("ci") or {}
        ciw = _f(ci.get("hi")) - _f(ci.get("lo")) if ci else float("nan")
        print(f"  {run:>22} {e:>4} {r['cpm']:>7.3f} {r['ceiling']:>6.3f} {n_tied:>6} "
              f"{lo:>10.3f} - {hi:<9.3f} {ciw:>9.3f}")
    print("  ^ a WIDE ceiling range among CPM-tied epochs means the current rule is deciding recall by luck;")
    print("    a CI width >> the CPM gaps means the CPM ordering itself is not resolved by the val set.\n")

    # ---- B. per-run policy comparison ------------------------------------------------------------
    policies = [("current", pick_current, {}), ("cpm_argmax", pick_cpm_argmax, {})]
    policies += [(f"recall_floor@{f:g}", pick_recall_floor, {"floor": f, "pool_budget": args.pool_budget})
                 for f in floors]
    policies += [("a2_shaped", pick_a2_shaped, {"pool_budget": args.pool_budget}),
                 ("recall_primary", pick_recall_primary, {"pool_budget": args.pool_budget})]

    picks = {name: {run: _row(info, fn(info, **kw)) for run, info in runs.items()}
             for name, fn, kw in policies}

    print("# B. WHAT EACH POLICY WOULD PICK   (epoch / CPM / ceiling / cands-per-vol)\n")
    hdr = f"  {'run':>22} " + "".join(f"{n:>26}" for n, _, _ in policies)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for run in runs:
        line = f"  {run:>22} "
        for name, _, _ in policies:
            p = picks[name][run]
            line += f"{p['epoch']:>4}/{p['cpm']:.2f}/{p['ceiling']:.2f}/{p['pool']:>6.0f}".rjust(26)
        print(line)

    # ---- C. aggregate by regime ------------------------------------------------------------------
    print("\n# C. AGGREGATE BY REGIME — the two decision numbers\n")
    print(f"  {'policy':>20} {'FOLD cpm':>9} {'FOLD ceil*':>11} {'SEED cpm':>9} {'SEED ceil':>10} "
          f"{'moved F/S':>10} {'max pool':>9}")
    summary = {}
    base = picks["current"]
    for name, _, _ in policies:
        pk = picks[name]
        fold = [pk[r] for r in runs if _regime(r) == "FOLD"]
        seed = [pk[r] for r in runs if _regime(r) == "SEED"]
        mf = lambda rows, k: float(np.nanmean([x[k] for x in rows])) if rows else float("nan")
        moved_f = sum(1 for r in runs if _regime(r) == "FOLD" and pk[r]["epoch"] != base[r]["epoch"])
        moved_s = sum(1 for r in runs if _regime(r) == "SEED" and pk[r]["epoch"] != base[r]["epoch"])
        mx_pool = float(np.nanmax([pk[r]["pool"] for r in runs]))
        summary[name] = {"fold_cpm": mf(fold, "cpm"), "fold_ceiling": mf(fold, "ceiling"),
                         "seed_cpm": mf(seed, "cpm"), "seed_ceiling": mf(seed, "ceiling"),
                         "moved_fold": moved_f, "moved_seed": moved_s, "max_pool": mx_pool,
                         "per_run": pk}
        s = summary[name]
        print(f"  {name:>20} {s['fold_cpm']:>9.3f} {s['fold_ceiling']:>11.3f} {s['seed_cpm']:>9.3f} "
              f"{s['seed_ceiling']:>10.3f} {str(moved_f)+'/'+str(moved_s):>10} {mx_pool:>9.0f}")
    print("  * FOLD ceiling == the fraction of TRAIN sets that will contain a positive → the rescorer's usable")
    print("    training signal (Phase-4 §2: a set with no positive contributes nothing to the ranking loss).")
    print("    SEED ceiling == the hard cap on every reported val/test number (Inv. 8).\n")

    # ---- D. verdicts ------------------------------------------------------------------------------
    print("# D. READ\n")
    b = summary["current"]
    for name, _, _ in policies:
        if name == "current":
            continue
        s = summary[name]
        d_fc, d_fcpm = s["fold_ceiling"] - b["fold_ceiling"], s["fold_cpm"] - b["fold_cpm"]
        d_sc, d_scpm = s["seed_ceiling"] - b["seed_ceiling"], s["seed_cpm"] - b["seed_cpm"]
        evalnote = ("evaluation UNTOUCHED (no seed moved)" if s["moved_seed"] == 0
                    else f"EVALUATION CHANGES ({s['moved_seed']} seed(s) moved): seed ceiling {d_sc:+.3f}, seed CPM {d_scpm:+.3f}")
        print(f"  {name:>20}: fold ceiling {d_fc:+.3f} (≈{d_fc*100:+.0f} TP-bearing train sets per 100) "
              f"at fold CPM {d_fcpm:+.3f}; {evalnote}"
              f"{'  [POOL OVER BUDGET]' if s['max_pool'] > args.pool_budget else ''}")
    print("\n  Decision frame: FOLD changes only affect the rescorer's TRAINING pool (calibration [P3U2.8],")
    print("  B0' [P3U2.12] and the FP probe read SEED/val data only) — low risk, re-run [P3U2.7]+train generation.")
    print("  SEED changes move the headline numbers — that is the real bet. A policy that improves fold ceiling")
    print("  with moved_seed=0 captures the upside with no change to any reported result.")
    print("\n  NOTE: this simulation changes NOTHING. Adopting any policy requires an Inv. 2 / A1 amendment.")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items()} for k, v in summary.items()}, indent=2, default=float))
        print(f"\njson = {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
