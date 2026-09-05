"""[MIG-6] Merge the seed-split ``grid<tag>.json`` parts into the canonical ``grid.json``.

**Why this exists.** The [MIG-5] smoke measured 11 h 50 m at n_boot=100 on the MIG slice, so the
n=1000 reportable (~110–120 h) cannot fit the 72 h walltime as one job. The route decided
2026-09-04: three concurrent single-seed jobs (``--seeds N --grid-tag _seedN``), merged here.

**What merging must preserve.** Each part is a complete, gate-checked evaluation of ITS seed —
nothing statistical happens across parts except the same mean±std the single-job path computes
(``rescore.evaluate.seed_summary``, reused verbatim) and the same comparison-row aggregation
(mean/std over per-seed deltas). Inv. 14 is untouched: seeds are summarised, never pooled.
``tests/test_merge_grid.py`` pins merged == what a single 3-seed run would have assembled.

Usage (LOGIN, seconds):
    python scripts/phase4_merge_grid.py --out-root $WORK/outputs_iso/phase4 \\
        --tags _seed0 _seed1 _seed2
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from abus_jcr.rescore.evaluate import seed_summary

from _phase4_common import add_phase4_paths, dump_json, grid_dir

BASE_RUNGS = ("B0", "B0-spread", "B0-rank")


def merge_grids(parts: list) -> dict:
    """Assemble one grid from single-seed (or disjoint multi-seed) parts."""
    merged = {"per_rung": {}, "per_seed": {}, "comparisons": {}, "gates": {}}

    # Phase 5 (2026-09-05): parts carry the split they scored; a val part merged into a
    # test grid (or vice versa) would silently average two different questions. Parts
    # written before the key existed are val grids by construction.
    splits = {p.get("eval_split", "val") for p in parts}
    if len(splits) > 1:
        raise SystemExit(f"parts disagree on eval_split: {sorted(splits)} — a val part "
                         f"cannot be merged with a test part; check the --tags list")
    merged["eval_split"] = splits.pop()

    # --- per_seed: a disjoint union, nothing recomputed ---------------------------
    for p in parts:
        overlap = set(p["per_seed"]) & set(merged["per_seed"])
        if overlap:
            raise SystemExit(f"seed(s) {sorted(overlap)} appear in more than one part — "
                             f"a duplicated seed would be double-counted in every mean; "
                             f"check the --tags list")
        merged["per_seed"].update(p["per_seed"])
    seeds = sorted(merged["per_seed"])
    rung_sets = [set(merged["per_seed"][s]) for s in seeds]
    if any(r != rung_sets[0] for r in rung_sets):
        raise SystemExit(f"parts disagree on the rung set: "
                         f"{[sorted(r) for r in rung_sets]} — evaluate the missing rungs "
                         f"before merging")

    # --- per_rung: the same summary the single-job path computes ------------------
    rung_order = [r for r in BASE_RUNGS if r in rung_sets[0]]
    rung_order += [r for p in parts for r in p["per_rung"]
                   if r not in rung_order and r in rung_sets[0]]
    for rung in rung_order:
        per = [merged["per_seed"][s][rung] for s in seeds]
        summ = seed_summary(per)
        tr = [p.get("train_cpm") for p in per if p.get("train_cpm") is not None]
        merged["per_rung"][rung] = {**summ,
                                    "train_cpm_mean": float(np.mean(tr)) if tr else None,
                                    "key_recall": per[0]["key_recall"]}

    # --- comparisons: concatenate per-seed rows, recompute the two aggregates ----
    for p in parts:
        for key, comp in p.get("comparisons", {}).items():
            merged["comparisons"].setdefault(key, {"per_seed": []})["per_seed"].extend(
                comp["per_seed"])
    for key, comp in merged["comparisons"].items():
        deltas = [r["delta"] for r in comp["per_seed"]]
        comp["delta_mean"] = float(np.mean(deltas))
        comp["delta_std"] = float(np.std(deltas))

    # --- gates: parts' verdicts recorded; the two mean-based ones recomputed ------
    merged["gates"]["parts"] = [p.get("gates", {}) for p in parts]
    for flag in ("pool_identity", "ceiling_invariant_and_respected"):
        vals = [p.get("gates", {}).get(flag) for p in parts]
        merged["gates"][flag] = bool(all(v for v in vals if v is not None)) if any(
            v is not None for v in vals) else None
    if "B1" in merged["per_rung"] and "B0" in merged["per_rung"]:
        b0m = merged["per_rung"]["B0"]["cpm_mean"]
        merged["gates"]["b1_beats_b0"] = bool(merged["per_rung"]["B1"]["cpm_mean"] > b0m)
        merged["gates"]["b0_cpm_mean_measured"] = float(b0m)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="[MIG-6] merge seed-split grid parts")
    add_phase4_paths(ap)
    ap.add_argument("--tags", nargs="+", required=True,
                    help="the --grid-tag values of the parts, e.g. _seed0 _seed1 _seed2")
    ap.add_argument("--out-tag", default="",
                    help="tag for the MERGED outputs (default: none -> the canonical grid.json)")
    args = ap.parse_args()

    parts = []
    for tag in args.tags:
        path = grid_dir(args) / f"grid{tag}.json"
        parts.append(json.load(open(path)))
        print(f"# part {path}: seeds {sorted(parts[-1]['per_seed'])}, "
              f"{len(parts[-1]['per_rung'])} rungs, {len(parts[-1].get('comparisons', {}))} "
              f"comparisons")
    merged = merge_grids(parts)
    seeds = sorted(merged["per_seed"])

    lab = f"{merged['eval_split']} CPM"
    print(f"\n{'='*78}\n# [4.7] LADDER (merged over {len(seeds)} seed-split parts) — "
          f"CPM mean +/- std\n")
    print(f"  {'rung':<10} {lab:>16} {'ceiling':>16} {'train CPM':>10}")
    for rung, r in merged["per_rung"].items():
        t = r.get("train_cpm_mean")
        print(f"  {rung:<10} {r['cpm_mean']:.4f} +/- {r['cpm_std']:.4f}  "
              f"{r['ceiling_mean']:.4f} +/- {r['ceiling_std']:.4f}  "
              f"{'nan' if t is None else f'{t:.4f}'}")

    print(f"\n# [4.7] COMPARISONS (merged; paired bootstrap rows per seed)\n")
    for key, comp in merged["comparisons"].items():
        for row in comp["per_seed"]:
            print(f"  {key}: {row['delta']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]  "
                  f"frac {row['frac_positive']:.3f}")
        print(f"    -> mean delta {comp['delta_mean']:+.4f} +/- {comp['delta_std']:.4f}")

    g = merged["gates"]
    print(f"\n# [4.7] GATES (merged): pool_identity+ceiling "
          f"{'PASS' if g.get('pool_identity') and g.get('ceiling_invariant_and_respected') else 'CHECK PARTS'}"
          f"; exit check 4 — B1 "
          f"{merged['per_rung'].get('B1', {}).get('cpm_mean', float('nan')):.4f} > "
          f"B0 {g.get('b0_cpm_mean_measured', float('nan')):.4f}: "
          f"{'PASS' if g.get('b1_beats_b0') else 'BELOW B0'}")

    dump_json(merged, grid_dir(args) / f"grid{args.out_tag}.json")
    md = grid_dir(args) / f"grid_table{args.out_tag}.md"
    lines = [f"| rung | {merged['eval_split']} CPM (mean ± std) | ceiling | train CPM |",
             "|---|---|---|---|"]
    for rung, r in merged["per_rung"].items():
        t = r.get("train_cpm_mean")
        lines.append(f"| {rung} | {r['cpm_mean']:.4f} ± {r['cpm_std']:.4f} | "
                     f"{r['ceiling_mean']:.4f} | {'—' if t is None else f'{t:.4f}'} |")
    md.write_text("\n".join(lines) + "\n")
    print(f"# wrote {grid_dir(args) / f'grid{args.out_tag}.json'}\n# wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
