"""Paired intervals for the one-switch pairs of the 2x3 factor grid (thesis ch. 7).

The reportable pass ([5.5]) stored paired intervals for 13 comparisons; the grid of
``abus_jcr.rescore.factor_grid`` needs pairs that were never among them (Joint/pooled vs
Independent/pooled, Joint+Geo/pooled vs Joint/pooled, pooled vs CE at each architecture, the
complete module vs the detector). Nothing is trained, no model is loaded and the test set is
not touched again: the inputs are the per-candidate predictions [5.5] dumped
(``preds_TEST_seed{S}/pred_<rung>_seed{S}.csv``) and the official ground truth.

How: ``bootstrap_cpm_ci`` and ``paired_bootstrap_delta`` draw their resamples from the same
seeded sequence, and the paired statistic is ``cpm(a) - cpm(b)`` on one resample. So ONE
per-condition bootstrap vector per (seed, rung) serves every pair: ``M[a] - M[b]`` is the
paired bootstrap. Each draw also records the tie bounds of its CPM
(``phase4_tie_bound.cpm_bounds``), so the regression check against the stored intervals holds
across platforms whose non-stable sort orders ties differently.

    python scripts/phase5_factor_comparisons.py --phase5-execute --boot-workers 15 \\
        --grid $G/grid_TEST.json --preds-dirs $G/preds_TEST_seed0 $G/preds_TEST_seed1 \\
        $G/preds_TEST_seed2 --gt-csv $WORK/data/Test/bbx_labels.csv \\
        --boot-dir $G/boot_TEST --out $G/factor_comparisons_TEST.json

Resumable: every (seed, rung) unit is persisted in chunks; re-running the same command
continues where it stopped. ``--report-only`` assembles the JSON from the units on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("TQDM_DISABLE", "1")      # the oracle draws one bar per call

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from abus_jcr import conventions as C  # noqa: E402
from abus_jcr.eval import froc as F  # noqa: E402
from abus_jcr.gt_labels import load_gt_documented, to_official_gt  # noqa: E402
from abus_jcr.rescore.evaluate import assert_pool_identity  # noqa: E402
from abus_jcr.rescore.factor_grid import (BOOT_CONDITIONS, DISPLAY, FACTOR_PAIRS, OVERALL,  # noqa: E402
                                          pair_key, pair_stats)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_tie_bound import cpm_bounds  # noqa: E402

CHUNK = 50          # draws persisted at a time: a kill loses minutes, not the unit


# ------------------------------------------------------------------------------- the draws
def make_draws(n_volumes: int, n_boot: int, seed: int = 0) -> List[np.ndarray]:
    """The exact draw sequence of ``froc.bootstrap_cpm_ci`` / ``paired_bootstrap_delta``."""
    rng = np.random.default_rng(seed)
    return [rng.choice(n_volumes, size=n_volumes, replace=True) for _ in range(int(n_boot))]


def _draw_with_bounds(draw) -> tuple:
    """One resample -> (official CPM, smallest, largest CPM over tie orderings)."""
    res = F.evaluate_froc(F._relabelled_gt(draw), F._relabelled_pred(draw, "pred"))
    det = res["detection"]
    b = cpm_bounds(det["fp"], det["recall"])
    return F.cpm(res), b["cpm_lo"], b["cpm_hi"]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot_unit(gt: pd.DataFrame, pred: pd.DataFrame, out_npy: Path, meta: Dict,
              n_boot: int, workers: int, limit: int | None = None) -> np.ndarray:
    """(n_boot, 3) array of per-draw (cpm, cpm_lo, cpm_hi); chunk-persisted and resumable.

    A finished unit is ``out_npy`` + a ``.json`` sidecar whose ``meta`` must match, else it is
    recomputed. ``limit`` stops after that many draws (timing slices) and never finalises.
    """
    out_npy = Path(out_npy)
    side = out_npy.with_suffix(".json")
    part = out_npy.with_suffix(".partial.npy")
    if out_npy.exists() and side.exists() and json.loads(side.read_text()) == meta:
        return np.load(out_npy)

    arr = np.full((int(n_boot), 3), np.nan)
    part_side = part.with_suffix(".json")
    if part.exists() and part_side.exists() and json.loads(part_side.read_text()) == meta:
        old = np.load(part)
        if old.shape == arr.shape:
            arr = old
    pids = list(pd.unique(gt["public_id"]))
    draws = make_draws(len(pids), n_boot, seed=int(meta["boot_seed"]))
    state = {"pids": pids, "gt": {p: gt[gt["public_id"] == p] for p in pids},
             "pred": {p: pred[pred["public_id"] == p] for p in pids}}
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    part_side.write_text(json.dumps(meta))
    todo = [i for i in range(int(n_boot)) if np.isnan(arr[i, 0])]
    if limit is not None:
        todo = todo[:int(limit)]
    for k in range(0, len(todo), CHUNK):
        idx = todo[k:k + CHUNK]
        arr[idx] = np.asarray(F._run_draws(_draw_with_bounds, [draws[i] for i in idx],
                                           state, workers), dtype=float)
        tmp = part.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, part)
    if np.isnan(arr).any():
        return arr
    tmp = out_npy.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, out_npy)
    side.write_text(json.dumps(meta))
    part.unlink(missing_ok=True)
    part_side.unlink(missing_ok=True)
    return arr


# ------------------------------------------------------------------------------ the report
def _quantiles(v: np.ndarray, ci: float = 0.95) -> tuple:
    a = (1.0 - ci) / 2.0
    return float(np.quantile(v, a)), float(np.quantile(v, 1.0 - a))


def _inside(x: float, lo: float, hi: float, tol: float = 1e-9) -> bool:
    return lo - tol <= x <= hi + tol


def regression_report(grid: Dict, M: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """Stored [5.5] intervals vs this run's: exact deviation, and inside-the-tie-envelope."""
    rows, ok = [], True
    for s, by_rung in M.items():
        for rung, arr in by_rung.items():
            st = grid["per_seed"][s][rung]["ci"]
            lo, hi = _quantiles(arr[:, 0])
            env_lo = (_quantiles(arr[:, 1])[0], _quantiles(arr[:, 2])[0])
            env_hi = (_quantiles(arr[:, 1])[1], _quantiles(arr[:, 2])[1])
            good = _inside(st["lo"], *env_lo) and _inside(st["hi"], *env_hi)
            ok &= good
            rows.append({"kind": "marginal", "seed": s, "what": rung, "inside_envelope": good,
                         "dev_lo": lo - st["lo"], "dev_hi": hi - st["hi"]})
        for key, comp in grid["comparisons"].items():
            a, b = next(((x, y) for x in by_rung for y in by_rung if pair_key(x, y) == key),
                        (None, None))
            if a is None:
                continue
            st = next(r for r, ss in zip(comp["per_seed"], sorted(grid["per_seed"])) if ss == s)
            ps = pair_stats(by_rung[a][:, 0], by_rung[b][:, 0],
                            lo_a=by_rung[a][:, 1], hi_a=by_rung[a][:, 2],
                            lo_b=by_rung[b][:, 1], hi_b=by_rung[b][:, 2])
            e = ps["envelope"]
            good = (_inside(st["lo"], *e["lo"]) and _inside(st["hi"], *e["hi"])
                    and _inside(st["frac_positive"], *e["frac_positive"]))
            ok &= good
            rows.append({"kind": "paired", "seed": s, "what": key, "inside_envelope": good,
                         "dev_lo": ps["lo"] - st["lo"], "dev_hi": ps["hi"] - st["hi"],
                         "dev_frac": ps["frac_positive"] - st["frac_positive"]})
    dev = max([abs(r[k]) for r in rows for k in ("dev_lo", "dev_hi")] or [0.0])
    dev_frac = max([abs(r["dev_frac"]) for r in rows if "dev_frac" in r] or [0.0])
    return {"all_inside_envelope": bool(ok), "max_abs_dev_interval": dev,
            "max_abs_dev_frac": dev_frac, "n_checked": len(rows), "rows": rows}


def build_report(grid: Dict, M: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    seeds = sorted(M)
    spread = {s: {r: cpm_bounds(grid["per_seed"][s][r]["fp"], grid["per_seed"][s][r]["recall"])
                  ["spread"] for r in M[s]} for s in seeds}
    pairs = []
    for p in tuple(FACTOR_PAIRS) + (OVERALL,):
        a, b = p["a"], p["b"]
        per_seed = []
        for s in seeds:
            if a not in M[s] or b not in M[s]:
                continue
            ps = pair_stats(M[s][a][:, 0], M[s][b][:, 0], lo_a=M[s][a][:, 1], hi_a=M[s][a][:, 2],
                            lo_b=M[s][b][:, 1], hi_b=M[s][b][:, 2])
            # the point difference is the reportable run's, never a re-evaluation
            delta = float(grid["per_seed"][s][a]["cpm"] - grid["per_seed"][s][b]["cpm"])
            tie = float(spread[s][a] + spread[s][b])
            per_seed.append({"seed": int(s), "delta": delta, **ps, "joint_tie_bound": tie,
                             "directional": bool(abs(delta) > tie),
                             "separated": bool(ps["lo"] > 0 or ps["hi"] < 0)})
        if not per_seed:
            continue
        d = np.array([r["delta"] for r in per_seed])
        rates = {}
        for k in C.KEY_FP:
            v = [grid["per_seed"][s][a]["key_recall"][_rate_key(grid, s, a, k)]
                 - grid["per_seed"][s][b]["key_recall"][_rate_key(grid, s, b, k)] for s in seeds]
            rates[str(k)] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
        pairs.append({**p, "key": pair_key(a, b), "display": f"{DISPLAY[a]} - {DISPLAY[b]}",
                      "delta_mean": float(d.mean()), "delta_std": float(d.std()),
                      "per_seed": per_seed, "per_rate": rates,
                      "stored_in_grid": pair_key(a, b) in grid["comparisons"]})
    return {"eval_split": grid.get("eval_split"), "pairs": pairs,
            "regression": regression_report(grid, M),
            "platform": {"machine": platform.machine(), "processor": platform.processor(),
                         "system": platform.platform(), "numpy": np.__version__,
                         "pandas": pd.__version__}}


def _rate_key(grid: Dict, s: str, rung: str, k) -> str:
    keys = grid["per_seed"][s][rung]["key_recall"]
    return next(x for x in keys if float(x) == float(k))


def _find_pred(preds_dirs: Sequence[str], rung: str, seed: int) -> Path:
    name = f"pred_{rung}_seed{int(seed)}.csv"
    for d in preds_dirs:
        if (Path(d) / name).exists():
            return Path(d) / name
    raise SystemExit(f"{name} not found in any of {list(preds_dirs)}")


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="paired intervals for the factor-grid pairs")
    ap.add_argument("--grid", required=True, help="merged grid_TEST.json of [5.6]")
    ap.add_argument("--preds-dirs", nargs="+", required=True)
    ap.add_argument("--gt-csv", required=True, help="data/Test/bbx_labels.csv")
    ap.add_argument("--boot-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.RESC_SEEDS))
    ap.add_argument("--conditions", nargs="+", default=list(BOOT_CONDITIONS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-workers", type=int, default=1)
    ap.add_argument("--draw-slice", type=int, default=None,
                    help="timing: stop each unit after this many draws (never finalised)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--phase5-execute", action="store_true")
    args = ap.parse_args()

    grid = json.loads(Path(args.grid).read_text())
    if grid.get("eval_split") == "test" and not args.phase5_execute:
        raise SystemExit("a test grid requires --phase5-execute (Inv. 9): this reads the "
                         "predictions of the sanctioned test contact, it makes no new one")
    gt = to_official_gt(load_gt_documented(Path(args.gt_csv)))
    boot_dir = Path(args.boot_dir)

    M: Dict[str, Dict[str, np.ndarray]] = {}
    for s in args.seeds:
        ref = pd.read_csv(_find_pred(args.preds_dirs, "B0", s), float_precision="round_trip")
        for rung in args.conditions:
            path = _find_pred(args.preds_dirs, rung, s)
            out_npy = boot_dir / f"seed{s}" / f"{rung}.npy"
            meta = {"rung": rung, "seed": int(s), "n_boot": int(args.n_boot), "boot_seed": 0,
                    "pred_sha256": _sha(path), "gt_sha256": _sha(Path(args.gt_csv))}
            if args.report_only:
                if out_npy.exists():
                    M.setdefault(str(s), {})[rung] = np.load(out_npy)
                continue
            pred = pd.read_csv(path, float_precision="round_trip")
            assert_pool_identity(ref, pred)
            t0 = time.time()
            arr = boot_unit(gt, pred, out_npy, meta, args.n_boot, args.boot_workers,
                            limit=args.draw_slice)
            done = int((~np.isnan(arr[:, 0])).sum())
            print(f"# seed {s} {rung:7s} {done}/{args.n_boot} draws  ({time.time() - t0:.0f} s)",
                  flush=True)
            if done == args.n_boot:
                M.setdefault(str(s), {})[rung] = arr

    if not M:
        print("# no complete unit yet — nothing to report")
        return 0
    rep = build_report(grid, M)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    reg = rep["regression"]
    print(f"# regression: {reg['n_checked']} stored intervals checked, inside tie envelope = "
          f"{reg['all_inside_envelope']}, max |dev| interval {reg['max_abs_dev_interval']:.2e}, "
          f"frac {reg['max_abs_dev_frac']:.3f}")
    for p in rep["pairs"]:
        cells = "  ".join(f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
                          f"{100 * r['frac_positive']:.0f}%" for r in p["per_seed"])
        print(f"{p['factor']:10s} {p['display']:42s} {p['delta_mean']:+.3f}  {cells}")
    print(f"# wrote {args.out}")
    return 0 if reg["all_inside_envelope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
