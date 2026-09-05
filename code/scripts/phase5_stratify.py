"""[5.7] Per-candidate-count stratification + per-rate deltas — torch-free, post-reportable.

**Why this exists (PHASE_5.md / PHASE_5_SPEC §5.3).** The brief's honesty checks: relational
gains should concentrate where volumes have MANY artifact candidates and vanish on clean
single-candidate volumes, and any pooled-surrogate gain should concentrate at the low-FP
operating points. A single CPM mean can show neither.

**The leakage rule.** Bin edges are the per-seed TERCILES of the VAL pool's per-volume set
size (val is design-sanctioned data), applied FIXED to the test volumes — no test-derived
edge exists anywhere. Per-bin numbers are ordinary ``evaluate_variant`` calls on the volume
subset (same vendored oracle, same bootstrap seed), off the ``--dump-preds`` CSVs the eval
grid wrote — no model is loaded here.

Usage (MAIA — LOGIN, minutes; after [5.6]):
    python scripts/phase5_stratify.py --eval-split test --phase5-execute \\
        --grid $WORK/outputs_iso/phase5/grid/grid_TEST.json \\
        --preds-dirs $WORK/outputs_iso/phase5/grid/preds_TEST_seed0 \\
                     $WORK/outputs_iso/phase5/grid/preds_TEST_seed1 \\
                     $WORK/outputs_iso/phase5/grid/preds_TEST_seed2 \\
        --out-tag _TEST \\
        --phase1-out $WORK/outputs_iso/phase1 --phase3-out $WORK/outputs_iso/phase3 \\
        --out-root $WORK/outputs_iso/phase5 --variants-root $WORK/outputs_iso/phase4 \\
        --data-root $WORK/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.candidates.record import CANDIDATE_COLUMNS
from abus_jcr.rescore.evaluate import evaluate_variant
from abus_jcr.rescore.variants import COMPARISONS, COMPARISONS_FLOOR, COMPARISONS_POOLED

from _phase4_common import (add_phase4_paths, dump_json, grid_dir, load_gt, load_record,
                            val_pool_for_seed)

BIN_ORDER = ("low", "mid", "high")


def tercile_edges(sizes) -> Tuple[float, float]:
    """The 33.33 / 66.67 percentiles of a per-volume set-size distribution."""
    a = np.asarray(sizes, dtype=float)
    return (float(np.percentile(a, 100.0 / 3.0)), float(np.percentile(a, 200.0 / 3.0)))


def assign_bins(sizes: Dict[int, int], e1: float, e2: float) -> Dict[int, str]:
    """``low`` (size <= e1) / ``mid`` (<= e2) / ``high`` — boundary goes to the LOWER bin."""
    out = {}
    for pid, n in sizes.items():
        out[int(pid)] = "low" if n <= e1 else ("mid" if n <= e2 else "high")
    return out


def as_record_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Pad a dumped pred frame up to the candidate-record schema, presence-only.

    ``evaluate_variant`` routes through ``to_official_pred_csv``, which validates the FULL
    frozen record schema even though it reads only ``public_id`` + the six box columns + the
    probability. The dumped CSVs carry exactly those, in pool row order, so padding the
    unread columns with zeros changes nothing the oracle sees — and keeps this script on the
    ONE reported evaluation path instead of re-assembling its payload.
    """
    rec = pred_df.copy()
    for c in CANDIDATE_COLUMNS:
        if c not in rec.columns:
            rec[c] = 0
    return rec


def stratified_eval(pred_df: pd.DataFrame, gt_df: pd.DataFrame, bins: Dict[int, str],
                    n_boot: int, tag: str) -> Dict[str, dict]:
    """Per-bin ``evaluate_variant`` on the volume subset; empty bins are absent, never faked.

    The pred frame IS the record for the oracle's purposes (official schema + probability),
    so the exact reported machinery re-runs on each subset — pinned by
    ``tests/test_phase5_stratify.py`` against a direct subset evaluation.
    """
    out: Dict[str, dict] = {}
    for b in BIN_ORDER:
        vols = [pid for pid, bb in bins.items() if bb == b]
        if not vols:
            continue
        sub_p = pred_df[pred_df["public_id"].isin(vols)].reset_index(drop=True)
        sub_g = gt_df[gt_df["public_id"].isin(vols)].reset_index(drop=True)
        res = evaluate_variant(as_record_frame(sub_p), sub_p["probability"].to_numpy(float),
                               sub_g, f"{tag}_{b}", n_boot=n_boot)
        out[b] = {k: v for k, v in res.items() if k != "pred"}
    return out


def per_rate_deltas(grid: dict, pairs: Iterable[Tuple[str, str]],
                    seeds: Sequence[str]) -> dict:
    """Seven per-operating-point ``key_recall`` deltas per comparison, mean ± std over seeds."""
    out: dict = {}
    for a, b in pairs:
        per_seed = grid["per_seed"]
        if any(a not in per_seed[s] or b not in per_seed[s] for s in seeds):
            continue
        rates = sorted(per_seed[seeds[0]][a]["key_recall"], key=float)
        table = {}
        for r in rates:
            d = [float(per_seed[s][a]["key_recall"][r]) - float(per_seed[s][b]["key_recall"][r])
                 for s in seeds]
            table[r] = {"mean": float(np.mean(d)), "std": float(np.std(d)),
                        "per_seed": [float(x) for x in d]}
        out[f"{a}-{b}"] = table
    return out


def _set_sizes(record: pd.DataFrame) -> Dict[int, int]:
    return {int(p): int(n) for p, n in record.groupby("public_id").size().items()}


def _find_pred(preds_dirs, rung: str, seed: int) -> Path:
    name = f"pred_{rung}_seed{int(seed)}.csv"
    for d in preds_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    raise SystemExit(f"{name} not found in any of {list(preds_dirs)} — was the eval grid "
                     f"run with --dump-preds?")


def main() -> int:
    ap = argparse.ArgumentParser(description="[5.7] per-count stratification + per-rate deltas")
    add_phase4_paths(ap)
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--phase5-execute", action="store_true")
    ap.add_argument("--grid", required=True, help="the MERGED reportable grid json")
    ap.add_argument("--preds-dirs", nargs="+", required=True,
                    help="the preds<grid-tag> dirs written by --dump-preds (one per seed job)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.RESC_SEEDS))
    ap.add_argument("--out-tag", default="", help="suffix for stratified<tag>.json")
    args = ap.parse_args()
    if args.eval_split == "test" and not args.phase5_execute:
        raise SystemExit("--eval-split test requires --phase5-execute (Inv. 9)")

    grid = json.loads(Path(args.grid).read_text())
    rungs = sorted(grid["per_seed"][str(args.seeds[0])])
    rec_val_all = load_record(args, "val")           # bin edges: val, design-sanctioned
    rec_ev_all = load_record(args, args.eval_split)
    gt_ev = load_gt(args, args.eval_split)

    payload: dict = {"eval_split": args.eval_split, "n_boot": int(args.n_boot),
                     "bin_rule": "per-seed VAL set-size terciles, boundary to the lower bin",
                     "per_seed": {}}
    for seed in args.seeds:
        e1, e2 = tercile_edges(list(_set_sizes(val_pool_for_seed(rec_val_all, seed)).values()))
        ev_pool = val_pool_for_seed(rec_ev_all, seed)
        bins = assign_bins(_set_sizes(ev_pool), e1, e2)
        counts = {b: sum(1 for v in bins.values() if v == b) for b in BIN_ORDER}
        print(f"\n# seed {seed}: val tercile edges = ({e1:.1f}, {e2:.1f}); "
              f"{args.eval_split} volumes per bin = {counts}")
        seed_out = {"edges": [e1, e2], "bin_of_volume": {str(k): v for k, v in bins.items()},
                    "rungs": {}}
        for rung in rungs:
            pred = pd.read_csv(_find_pred(args.preds_dirs, rung, seed))
            res = stratified_eval(pred, gt_ev, bins, n_boot=args.n_boot, tag=f"{rung}_s{seed}")
            seed_out["rungs"][rung] = res
            row = "  ".join(f"{b}: {res[b]['cpm']:.4f} [{res[b]['ci']['lo']:.4f}, "
                            f"{res[b]['ci']['hi']:.4f}] (n={res[b]['n_volumes']})"
                            for b in BIN_ORDER if b in res)
            print(f"  {rung:<10} {row}")
        payload["per_seed"][str(seed)] = seed_out

    pairs = tuple(COMPARISONS) + tuple(COMPARISONS_POOLED) + tuple(COMPARISONS_FLOOR)
    payload["per_rate_deltas"] = per_rate_deltas(grid, pairs,
                                                 seeds=[str(s) for s in args.seeds])
    print(f"\n# per-rate deltas (mean over seeds; + favours the first name)")
    for key, table in payload["per_rate_deltas"].items():
        cells = "  ".join(f"{r}: {v['mean']:+.4f}" for r, v in table.items())
        print(f"  {key:<16} {cells}")

    dump_json(payload, grid_dir(args) / f"stratified{args.out_tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
