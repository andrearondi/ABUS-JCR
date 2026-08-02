"""Recompute candidate elongation about each axis in TRUE mm. READ-ONLY, no volumes read.

Answers one question the recorded pool diagnostics could not: **does elongation about the
BEAM axis separate TP from FP?** The shipped ``anisotropy`` feature is
``ext_d0 / mean(ext_d1, ext_d2)`` on iso-cache voxels, which measures elongation about
``d0`` — and ``d0`` is lateral, not depth (``results/AXIS_CHECK.md``). So the near-null
effect sizes on record are about a different quantity; the depth ratio has never been tested.

This runs on the frozen candidate record alone — no NRRD, no cache, no checkpoint, no
retraining. Every detector in the record is reported in one pass, because replication across
detectors is the only thing that separates a finding from a coincidence here.

**It is a diagnostic, not a feature proposal.** ``abs_geom`` already carries
``log1p(ext_d0..d2)``, so any ratio of extents is a difference of logs and is linearly
recoverable from what the set model already receives. A corrected scalar would add no
information. What a large delta WOULD mean is that a ray-shaped population exists in the
pool — which bears on the shadow question and on whether the distorted cache costs the 3-D
encoder anything.

Usage:
    python scripts/phase3_anisotropy_recheck.py \
        --out-root /home/maia-user/Andre2/outputs/phase3 --split both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.candidates.record import read_candidate_record
from abus_jcr.probe import anisotropy as A
from abus_jcr.probe.intensity_geom import cliffs_delta
from _phase3_common import add_phase3_paths

# Measured on 130 volumes (30 Val + 100 Train, disjoint): four independent lines agree.
# See results/AXIS_CHECK.md and RESULTS_INTENSITY_PROBE.md [I.3]. Passed as a DEFAULT, not
# written to conventions.py — nothing frozen is touched by this script.
MEASURED_SPACING_MM = (0.200, 0.073, 0.475674)
ROLE = {0: "lateral", 1: "depth/beam", 2: "sweep/elev"}


def analyse(pool: pd.DataFrame, true_spacing, iso_mm, split: str) -> list:
    rows = []
    for det, g_all in pool.groupby("detector_of_origin", sort=True):
        # Inv. 11 ignore band dropped, not folded into the FPs (they are partial hits).
        g = g_all[g_all["label"].isin(("pos", "neg"))]
        if g.empty:
            continue
        is_tp = (g["label"].to_numpy() == "pos")
        vid = g["public_id"].to_numpy()
        e = A.extents_mm(g, true_spacing, iso_mm)
        feats = {"deployed": A.deployed_anisotropy(g)}
        feats.update(A.elongation_ratios(e["native_mm"]))

        # consistency of the two independent mm routes
        rel = np.abs(e["native_mm"] - e["iso_mm"]) / np.maximum(np.abs(e["native_mm"]), 1e-9)
        rays = A.ray_fractions(feats["elong_d1"], is_tp)
        for name, v in feats.items():
            pooled = cliffs_delta(v[is_tp], v[~is_tp])
            pv, sign, nvol = A.per_volume_delta(v, is_tp, vid)
            rows.append({
                "split": split, "detector": det, "ratio": name,
                "tp_med": float(np.nanmedian(v[is_tp])) if is_tp.any() else np.nan,
                "fp_med": float(np.nanmedian(v[~is_tp])) if (~is_tp).any() else np.nan,
                "pooled_d": pooled, "perVol_d": pv, "sign": sign,
                "n_tp": int(is_tp.sum()), "n_fp": int((~is_tp).sum()), "n_vol": nvol,
                "mm_rel_p50": float(np.nanpercentile(rel, 50)),
                "mm_rel_p90": float(np.nanpercentile(rel, 90)),
                "mm_rel_max": float(np.nanmax(rel)),
                **rays,
            })
    return rows


def _figure(splits, args, true_spacing, iso_mm, out_dir: Path):
    """One panel per elongation ratio: TP vs FP distributions with the CUBIC line marked.

    The point of drawing it is §D: whether anything sits to the right of 1.0. A median
    comparison cannot show an absent sub-population; a histogram can.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                  # pragma: no cover
        print(f"(figure skipped: {type(e).__name__}: {e})")
        return
    frames = []
    for sp in splits:
        cpath = Path(args.candidates) if args.candidates else \
            Path(args.out_root) / "candidates" / f"candidates_{sp}"
        try:
            pool = read_candidate_record(cpath.with_suffix(""))
        except Exception:
            continue
        pool = pool[(pool["split"] == sp) & (pool["label"].isin(("pos", "neg")))]
        if len(pool):
            frames.append((sp, pool))
    if not frames:
        return
    fig, axes = plt.subplots(len(frames), 3, figsize=(13, 3.4 * len(frames)), squeeze=False)
    for i, (sp, pool) in enumerate(frames):
        e = A.extents_mm(pool, true_spacing, iso_mm)["native_mm"]
        ratios = A.elongation_ratios(e)
        is_tp = (pool["label"].to_numpy() == "pos")
        for j, name in enumerate(A.RATIO_NAMES):
            ax = axes[i][j]
            v = ratios[name]
            v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
            bins = np.logspace(np.log10(np.nanpercentile(v, 0.5)),
                               np.log10(np.nanpercentile(v, 99.5)), 50)
            for tag, m, col in (("FP", ~is_tp, "tab:red"), ("TP", is_tp, "tab:blue")):
                ax.hist(v[m], bins=bins, density=True, alpha=0.5, color=col,
                        label=f"{tag} (n={int(np.isfinite(v[m]).sum())})")
            ax.axvline(1.0, color="k", ls="--", lw=1.2)
            ax.set_xscale("log")
            ax.set_title(f"{sp} — {name}  ({ROLE[j]})\ndashed = physically cubic", fontsize=9)
            ax.set_xlabel("extent along axis / mean(other two), mm")
            if j == 0:
                ax.set_ylabel("density")
            ax.legend(fontsize=8)
    fig.suptitle("Anisotropy recheck — elongation in TRUE mm. A posterior-shadow ray would "
                 "sit to the RIGHT of the dashed line on the depth panel.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "anisotropy_recheck.png"
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print(f"\n  figure -> {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Corrected-anisotropy recheck (read-only)")
    add_phase3_paths(ap)
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--candidates", default=None, help="explicit record path (else <out-root>/candidates/)")
    ap.add_argument("--true-spacing", default=",".join(str(s) for s in MEASURED_SPACING_MM),
                    help="measured native spacing, storage order, mm")
    args = ap.parse_args()

    true_spacing = tuple(float(x) for x in args.true_spacing.split(","))
    iso_mm = A.iso_voxel_mm(true_spacing)
    ref = A.cubic_reference(iso_mm)

    print("=" * 84)
    print("# ANISOTROPY RECHECK — elongation about each axis, in TRUE millimetres\n")
    print(f"  declared spacing (conventions) = {tuple(C.SPACING_STORAGE_MM)}")
    print(f"  measured spacing (AXIS_CHECK)  = {true_spacing}")
    print(f"  => one CACHE voxel really spans {iso_mm[0]:.4f} x {iso_mm[1]:.4f} x {iso_mm[2]:.4f} mm")
    print(f"     (it would be {C.ISO_SPACING_MM} on every axis if the declared spacing were right)\n")
    print(f"  A physically CUBIC candidate reads:")
    print(f"     deployed anisotropy = {ref:.3f}   <-- NOT 1.0; this is the origin the recorded")
    print(f"                                          medians (~0.39-0.41) must be read against")
    print(f"     elong_d0 / elong_d1 / elong_d2 = 1.000 each\n")
    print(f"  elong_dA = extent along dA / mean(the other two), in mm.")
    for a in range(3):
        print(f"     {A.RATIO_NAMES[a]}: elongation about d{a} = {ROLE[a]}")
    print("  A posterior-shadow 'ray' is elongated about the BEAM axis -> large elong_d1.\n")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    rows = []
    for sp in splits:
        cpath = Path(args.candidates) if args.candidates else \
            Path(args.out_root) / "candidates" / f"candidates_{sp}"
        pool = read_candidate_record(cpath.with_suffix(""))
        pool = pool[pool["split"] == sp]
        if pool.empty:
            print(f"  (no rows for split '{sp}' in {cpath}) — skipped")
            continue
        rows += analyse(pool, true_spacing, iso_mm, sp)
    if not rows:
        raise SystemExit("no candidate rows found for any requested split")
    df = pd.DataFrame(rows)

    print("=" * 84)
    print("# A. REPLICATION — per-volume Cliff's delta (TP vs FP), sign consistency in ()")
    print("     Positive = TPs more elongated about that axis than FPs.")
    print("     |delta| >= 0.15 is the same 'worth a feature block' bar the Axis-A test uses;")
    print("     a delta only matters if it holds across detectors AND sign is near 1.0.\n")
    order = ["deployed"] + list(A.RATIO_NAMES)
    print(f"  {'split':>6} {'detector':>12} " + " ".join(f"{n:>16}" for n in order))
    for (sp, det), g in df.groupby(["split", "detector"], sort=True):
        cells = []
        for n in order:
            r = g[g["ratio"] == n]
            cells.append(f"{r['perVol_d'].iloc[0]:+.3f} ({r['sign'].iloc[0]:.2f})" if len(r) else "-")
        print(f"  {sp:>6} {det:>12} " + " ".join(f"{c:>16}" for c in cells))

    print("\n# B. RANGE ACROSS DETECTORS (the replication test)\n")
    print(f"  {'ratio':>16} {'perVol d: min':>14} {'max':>8} {'median':>8} {'sign min':>9} "
          f"{'all same sign?':>15}")
    for n in order:
        g = df[df["ratio"] == n]["perVol_d"].dropna()
        s = df[df["ratio"] == n]["sign"].dropna()
        same = "YES" if (g > 0).all() or (g < 0).all() else "NO — flips"
        print(f"  {n:>16} {g.min():>14.3f} {g.max():>8.3f} {g.median():>8.3f} "
              f"{(s.min() if len(s) else float('nan')):>9.2f} {same:>15}")

    print("\n# C. PHYSICAL VALUES — median elongation in mm terms (1.000 = cubic)\n")
    print(f"  {'split':>6} {'detector':>12} " + " ".join(f"{n + ' TP/FP':>22}" for n in A.RATIO_NAMES))
    for (sp, det), g in df.groupby(["split", "detector"], sort=True):
        cells = []
        for n in A.RATIO_NAMES:
            r = g[g["ratio"] == n]
            cells.append(f"{r['tp_med'].iloc[0]:.2f} / {r['fp_med'].iloc[0]:.2f}" if len(r) else "-")
        print(f"  {sp:>6} {det:>12} " + " ".join(f"{c:>22}" for c in cells))

    print("\n# D. IS THERE A RAY-SHAPED POPULATION AT ALL?  (elong_d1 = depth / mean(lat, sweep))")
    print("     A posterior-shadow ray must be LONGER along the beam than across it, i.e. > 1.")
    print("     Medians cannot answer this; a sub-population can hide under them.\n")
    print(f"  {'split':>6} {'detector':>12} " + " ".join(
        f"{'TP>' + t:>9} {'FP>' + t:>9}" for t in ("1", "1.5", "2")))
    for (sp, det), g in df.groupby(["split", "detector"], sort=True):
        r = g.iloc[0]
        cells = []
        for t in ("1", "1.5", "2"):
            cells += [f"{r['frac_tp_gt' + t]:.3f}", f"{r['frac_fp_gt' + t]:.3f}"]
        print(f"  {sp:>6} {det:>12} " + " ".join(f"{c:>9}" for c in cells))

    print(f"\n  consistency check — the two independent routes to millimetres (official native")
    print(f"  lengths x true spacing  vs  iso extents x true cache-voxel size). Relative")
    print(f"  disagreement: p50={df['mm_rel_p50'].max():.4f}  p90={df['mm_rel_p90'].max():.4f}  "
          f"max={df['mm_rel_max'].max():.4f}")
    print(f"  (worst-case over detectors. The max is dominated by the SMALLEST boxes, where a")
    print(f"  half-voxel of tube-reconstruction rounding is a large fraction; p50 is the number")
    print(f"  to read. A large p50 would mean a coordinate bug, not a finding.)")

    out_dir = Path(args.out_root) / "intensity_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    _figure(splits, args, true_spacing, iso_mm, out_dir)
    jp = out_dir / "anisotropy_recheck.json"
    jp.write_text(json.dumps({"true_spacing_mm": list(true_spacing),
                              "iso_voxel_mm": list(iso_mm),
                              "deployed_cubic_reference": ref,
                              "rows": df.to_dict("records")}, indent=2, default=str))
    print(f"\njson = {jp}")
    print("\nNOTE: abs_geom already ships log1p(ext_d0..d2), so any ratio of extents is a")
    print("difference of logs and is ALREADY recoverable by the set model. A large delta here")
    print("is evidence about the POOL (is there a ray-shaped FP population?), not a licence to")
    print("add a feature.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
