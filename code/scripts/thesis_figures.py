#!/usr/bin/env python3
"""Thesis figures for ABUS-JCR, regenerated from the mirrored artefacts on the laptop.

Torch-free. Reads the Phase 1-5 JSON/CSV artefacts under an Evo mirror root and writes vector
PDFs into a figures directory. Every number drawn here is read from an artefact; the only
hard-coded values are (a) the FINAL deployed epoch map of iso/RESULTS_ISO_REBUILD.md [I3.5],
which the laptop copies of select_*.json predate, and (b) the val pairwise-geometry Cliff's
deltas of results/RESULTS_PHASE_4_ISO.md [I3.12] val/ALL, whose pair-level record is not
mirrored. Both are marked HARDCODED below with their source.

Usage:
    python scripts/thesis_figures.py --evo /Volumes/Evo --out ../figures --split val
    python scripts/thesis_figures.py --evo /Volumes/Evo --out ../figures --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

# ----------------------------------------------------------------------------- style
OI = {  # Okabe-Ito, colour-blind safe
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F",
}
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abus_jcr.rescore.factor_grid import ARCH, DISPLAY, FACTOR_PAIRS, GRID, OBJ, OVERALL  # noqa: E402

KEY_FP = [0.125, 0.25, 0.5, 1, 2, 4, 8]
FULL_W, HALF_W = 6.2, 3.1

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# HARDCODED (source: iso/RESULTS_ISO_REBUILD.md [I3.5], md5-verified final deploy map)
FINAL_EPOCH = {"fold0": 4, "fold1": 7, "fold2": 5, "fold3": 7, "fold4": 3,
               "full_seed0": 5, "full_seed1": 7, "full_seed2": 5}
# HARDCODED (source: results/RESULTS_PHASE_4_ISO.md [I3.12] val/ALL, section 3)
PAIRWISE_VAL = {
    "log|dx|/w": (0.095, 0.970), "log|dy|/h": (0.024, 0.510), "log|dz|/d": (0.142, 0.812),
    "log w_n/w_m": (0.009, 0.000), "log h_n/h_m": (0.004, 0.000), "log d_n/d_m": (0.005, 0.000),
}


PNG_DIR: Path | None = None


def save(fig, out: Path, name: str):
    p = out / f"{name}.pdf"
    fig.savefig(p)
    if PNG_DIR is not None:
        fig.savefig(PNG_DIR / f"{name}.png", dpi=170)
    plt.close(fig)
    print(f"wrote {p}")


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- Ch2 schematic
def fig_froc_schematic(out: Path):
    fp = np.logspace(-2, 1.2, 400)
    rec = 0.95 * (1 - np.exp(-1.6 * fp ** 0.55))
    fig, ax = plt.subplots(figsize=(HALF_W * 1.4, 2.4))
    ax.plot(fp, rec, color=OI["blue"], lw=1.5)
    ys = np.interp(KEY_FP, fp, rec)
    for x, y in zip(KEY_FP, ys):
        ax.plot([x, x], [0, y], color=OI["grey"], lw=0.6, ls=":")
        ax.plot(x, y, "o", color=OI["vermilion"], ms=4)
    ax.axhline(0.95, color=OI["grey"], lw=0.8, ls="--")
    ax.text(0.011, 0.955, "recall ceiling of the pool", va="bottom", fontsize=8, color=OI["grey"])
    ax.axhline(ys.mean(), color=OI["vermilion"], lw=0.8, ls="--")
    ax.text(0.011, ys.mean() - 0.02, "mean of the seven sensitivities", va="top", fontsize=8,
            color=OI["vermilion"])
    ax.set_xscale("log")
    ax.set_xticks(KEY_FP)
    ax.set_xticklabels([str(k) for k in KEY_FP])
    ax.set_xlim(0.01, 16)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positives per volume (log scale)")
    ax.set_ylabel("sensitivity")
    save(fig, out, "froc_schematic")


# ----------------------------------------------------------------------------- Ch4
def fig_fidelity(evo: Path, out: Path):
    tr = pd.read_csv(evo / "outputs_iso/phase1/resample_fidelity_Train.csv")["iou"]
    va = pd.read_csv(evo / "outputs_iso/phase1/resample_fidelity_Validation.csv")["iou"]
    fig, ax = plt.subplots(figsize=(HALF_W * 1.4, 2.3))
    bins = np.arange(0.70, 1.001, 0.02)
    ax.hist(tr, bins=bins, color=OI["blue"], alpha=0.8, label=f"training (n={len(tr)})")
    ax.hist(va, bins=bins, color=OI["orange"], alpha=0.8, label=f"validation (n={len(va)})")
    for x, lab in [(0.5, "gate 0.5"), (0.3, "hit 0.3")]:
        pass  # both lie left of the plotted range; stated in the caption instead
    ax.set_xlabel("round-trip overlap of the released box (isotropic → native)")
    ax.set_ylabel("cases")
    ax.legend(loc="upper left")
    save(fig, out, "fidelity")


def fig_selection_curves(evo: Path, out: Path):
    runs = ["fold0", "fold1", "fold2", "fold3", "fold4", "full_seed0", "full_seed1", "full_seed2"]
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.6), sharex=True, sharey=True)
    for ax, run in zip(axes.ravel(), runs):
        s = load_json(evo / f"outputs_iso/phase3/selection/select_retinanet_{run}.json")
        pe = s["per_epoch"]
        ep = np.array(sorted(int(k) for k in pe))
        cpm = np.array([pe[str(e)]["cpm"] for e in ep])
        ceil = np.array([pe[str(e)]["ceiling"] for e in ep])
        elig = np.array([pe[str(e)]["eligible"] for e in ep])
        mx = cpm[elig].max()
        ax.axhspan(mx - s["cpm_tol"], mx, color=OI["yellow"], alpha=0.35, lw=0)
        ax.axvspan(-0.5, s["min_epoch"] - 0.5, color=OI["grey"], alpha=0.15, lw=0)
        ax.plot(ep, cpm, color=OI["blue"], lw=1.2, label="CPM")
        ax.plot(ep, ceil, color=OI["green"], lw=1.2, ls="--", label="recall ceiling")
        fe = FINAL_EPOCH[run]
        ax.plot(fe, cpm[ep == fe][0], "o", color=OI["vermilion"], ms=5, zorder=5)
        ax.set_title(run.replace("full_seed", "seed ").replace("fold", "fold "), fontsize=8)
        ax.set_xlim(-0.5, 29.5)
        ax.set_ylim(0.3, 1.0)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "selection_curves")


def _two_axis(ax, x, recall, pool, xlabel, frozen, xlog=False, xticklabels=None):
    ax.plot(x, recall, "o-", color=OI["blue"], ms=3, lw=1.2)
    ax.set_ylabel("linked recall", color=OI["blue"])
    ax.tick_params(axis="y", colors=OI["blue"])
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, pool, "s-", color=OI["vermilion"], ms=3, lw=1.2)
    ax2.set_ylabel("candidates / volume", color=OI["vermilion"])
    ax2.tick_params(axis="y", colors=OI["vermilion"])
    ax2.axhline(200, color=OI["grey"], lw=0.8, ls=":")
    if frozen is not None:
        ax.axvline(frozen, color=OI["grey"], lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    if xlog:
        ax.set_xscale("log")
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
    return ax2


def fig_linker_sweeps(evo: Path, out: Path):
    f = load_json(evo / "outputs_iso/phase3/linking/freeze_linking.json")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3))
    sw = f["sweeps"]["min_tube_len"]
    _two_axis(axes[0], [d["value"] for d in sw], [d["recall"] for d in sw],
              [d["cands_per_vol_mean"] for d in sw], "minimum tube length (slices)", 10)
    sw = f["score_floor_sweep"]
    _two_axis(axes[1], [d["score_floor"] for d in sw], [d["recall"] for d in sw],
              [d["cands_per_vol_mean"] for d in sw], "score floor before linking", 0.08)
    sw = f["nms_3d_sweep"]
    x = [9.0 if d["nms_iou"] == "off" else d["nms_iou"] for d in sw]
    order = np.argsort(x)[::-1]
    xs = [x[i] for i in order]
    _two_axis(axes[2], range(len(xs)), [sw[i]["recall"] for i in order],
              [sw[i]["cands_per_vol_mean"] for i in order], "3D NMS overlap threshold",
              None, xticklabels=["off" if v == 9.0 else str(v) for v in xs])
    for ax in axes:
        ax.set_ylim(0.70, 0.95)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "linker_sweeps")


def fig_operating_point(evo: Path, out: Path):
    o = load_json(evo / "outputs_iso/phase3/calibration/operating_point.json")
    sw = sorted(o["sweep"], key=lambda d: d["thresh"])
    th = np.array([d["thresh"] for d in sw])
    fig, ax = plt.subplots(figsize=(HALF_W * 1.6, 2.6))
    ax.errorbar(th, [d["recall_mean"] for d in sw], yerr=[d["recall_std"] for d in sw],
                fmt="o-", color=OI["blue"], ms=3, lw=1.2, capsize=2, label="linked recall")
    ax.plot(th, [d["cpm_mean"] for d in sw], "^-", color=OI["green"], ms=3, lw=1.2, label="CPM")
    ax.set_ylabel("recall / CPM")
    ax.set_ylim(0, 1.0)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.errorbar(th, [d["cands_per_vol_mean"] for d in sw], yerr=[d["cands_per_vol_std"] for d in sw],
                 fmt="s-", color=OI["vermilion"], ms=3, lw=1.2, capsize=2, label="candidates / volume")
    ax2.axhline(200, color=OI["grey"], lw=0.8, ls=":")
    ax2.set_ylabel("candidates / volume", color=OI["vermilion"])
    ax2.tick_params(axis="y", colors=OI["vermilion"])
    ax.axvline(0.05, color=OI["grey"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("detector score threshold (log scale)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    save(fig, out, "operating_point")


def _grid(evo: Path, split: str):
    if split == "val":
        p = evo / "maia_47_stage/phase4/grid/grid.json"
    else:
        p = evo / "maia_47_stage/phase5/grid/grid_TEST.json"
    if not p.exists():
        print(f"[skip] {p} not present")
        return None
    return load_json(p)


def _froc_axes(ax):
    ax.set_xscale("log")
    ax.set_xticks(KEY_FP)
    ax.set_xticklabels([str(k) for k in KEY_FP])
    ax.set_xlim(0.08, 12)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("false positives per volume")
    ax.set_ylabel("sensitivity")
    for k in KEY_FP:
        ax.axvline(k, color=OI["grey"], lw=0.4, ls=":")


def fig_baseline_froc(evo: Path, out: Path):
    g = _grid(evo, "val")
    fig, ax = plt.subplots(figsize=(HALF_W * 1.6, 2.6))
    cols = [OI["blue"], OI["green"], OI["orange"]]
    for seed, c in zip(sorted(g["per_seed"]), cols):
        r = g["per_seed"][seed]["B0"]
        fp, rc = np.array(r["fp"]), np.array(r["recall"])
        o = np.argsort(fp)
        ax.step(fp[o], rc[o], where="post", color=c, lw=1.2,
                label=f"seed {seed}: CPM {r['cpm']:.3f}, ceiling {r['ceiling']:.3f}")
        ax.axhline(r["ceiling"], color=c, lw=0.6, ls="--")
    _froc_axes(ax)
    ax.legend(loc="lower right")
    save(fig, out, "baseline_froc")


def fig_headroom(evo: Path, out: Path):
    q = load_json(evo / "maia_47_stage/phase3/baseline/calibration_quantisation_val.json")
    rows = [("detector\nscore", "score_max"), ("bound of any\nmonotone\nrescaling", "global_monotone"),
            ("rank rule\n(label-free)", "volume_neutral_anchored"), ("per-volume\noracle", "per_vol_oracle")]
    vals = {k: [s[k]["cpm"] for s in q["per_seed"]] for _, k in rows}
    ceil = [s["score_max"]["ceiling"] for s in q["per_seed"]]
    fig, ax = plt.subplots(figsize=(HALF_W * 1.6, 2.6))
    xs = np.arange(len(rows) + 1)
    means = [np.mean(vals[k]) for _, k in rows] + [np.mean(ceil)]
    cols = [OI["black"], OI["grey"], OI["blue"], OI["green"], OI["vermilion"]]
    ax.bar(xs, means, color=cols, width=0.6, alpha=0.85)
    for i, (_, k) in enumerate(rows):
        ax.plot([i] * 3, vals[k], "o", color="white", mec="black", ms=4, zorder=5)
    ax.plot([len(rows)] * 3, ceil, "o", color="white", mec="black", ms=4, zorder=5)
    for i, m in enumerate(means):
        ax.text(i + 0.33, m, f"{m:.3f}", ha="left", va="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows] + ["recall\nceiling"], fontsize=7.5)
    ax.set_xlim(-0.6, len(rows) + 0.9)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("validation CPM")
    save(fig, out, "headroom")


def _cands_val(evo: Path):
    c = pd.read_csv(evo / "maia_47_stage/phase3/candidates/candidates_val.csv")
    c = c[c["label"].isin(["pos", "neg"])].copy()
    c["elong_depth"] = c["ext_d1"] / ((c["ext_d0"] + c["ext_d2"]) / 2)
    c["elong_lateral"] = c["ext_d0"] / ((c["ext_d1"] + c["ext_d2"]) / 2)
    c["box_diag"] = np.sqrt(c["x_length"] ** 2 + c["y_length"] ** 2 + c["z_length"] ** 2)
    return c


def fig_audit_elongation(evo: Path, out: Path):
    c = _cands_val(evo)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for ax, col, xl, lines in [
        (axes[0], "elong_depth", "depth elongation", (1.0, 1.5, 2.0)),
        (axes[1], "elong_lateral", "lateral elongation", (1.0,)),
    ]:
        bins = np.linspace(0, 5 if col == "elong_lateral" else 2.5, 51)
        for lab, colr, name in [("neg", OI["vermilion"], "false positives"), ("pos", OI["blue"], "true positives")]:
            v = c.loc[c["label"] == lab, col].clip(upper=bins[-1])
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.4, color=colr,
                    label=f"{name} (n={len(v)})")
        for x in lines:
            ax.axvline(x, color=OI["grey"], lw=0.8, ls="--")
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel("density")
    axes[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "audit_elongation")
    # numbers behind the caption, printed for cross-checking against the results log
    for lab in ["pos", "neg"]:
        v = c.loc[c["label"] == lab, "elong_depth"]
        print(f"  elong_depth {lab}: median {v.median():.3f}  >1 {(v > 1).mean():.3f}  "
              f">1.5 {(v > 1.5).mean():.3f}  >2 {(v > 2).mean():.3f}")


def _cliffs(a, b):
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def fig_pool_priors(evo: Path, out: Path):
    c = _cands_val(evo)
    feats = ["score_max", "box_diag", "score_std", "slice_count", "z_span", "rank_norm", "score_mean",
             "area_cv", "centroid_jitter", "elong_depth", "fill_ratio", "score_min"]
    labels = {"elong_depth": "depth elongation"}
    tp, fp_ = c[c["label"] == "pos"], c[c["label"] == "neg"]
    d = {f: _cliffs(tp[f].values, fp_[f].values) for f in feats}
    order = sorted(feats, key=lambda f: -abs(d[f]))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ys = np.arange(len(order))[::-1]
    ax.barh(ys, [d[f] for f in order], color=[OI["blue"] if d[f] > 0 else OI["vermilion"] for f in order])
    ax.set_yticks(ys)
    ax.set_yticklabels([labels.get(f, f).replace("_", " ") for f in order])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Cliff's δ, true vs false positives (+ = higher on true positives)", fontsize=8)
    ax.set_xlim(-0.8, 0.8)
    ax = axes[1]
    comps = list(PAIRWISE_VAL)
    x = np.arange(len(comps))
    ax.bar(x - 0.2, [PAIRWISE_VAL[k][0] for k in comps], width=0.4, color=OI["orange"],
           label="hit–miss vs miss–miss pairs (suppression)")
    ax.bar(x + 0.2, [PAIRWISE_VAL[k][1] for k in comps], width=0.4, color=OI["blue"],
           label="hit–hit vs hit–miss pairs (co-location)")
    ax.axhline(0.15, color=OI["grey"], lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("log", "log ") for k in comps], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("|Cliff's δ|")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5, ncol=1)
    fig.tight_layout(w_pad=2)
    save(fig, out, "pool_priors")
    print("  computed val Cliff's deltas:", {k: round(v, 3) for k, v in d.items()})


def fig_objective_study(evo: Path, out: Path):
    o = load_json(evo / "maia_47_stage/phase4/objective_study/objective_study_seed0.json")
    fig, ax = plt.subplots(figsize=(HALF_W * 1.5, 2.6))
    fam_col = {"lesion": OI["blue"], "cand": OI["orange"]}
    for r in o["results"]:
        fam = "lesion" if r["per_lesion"] else "cand"
        mk = "o" if r["gamma"] == 0 else "^"
        ax.plot(r["raw"]["balacc"], r["raw"]["cpm"], mk, color=fam_col[fam], ms=5, mfc="none" if r["soft"] else fam_col[fam])
    b0 = o["b0"]
    ax.plot(b0["balacc"], b0["cpm"], "*", color=OI["black"], ms=11, zorder=5)
    ax.annotate("detector score", (b0["balacc"], b0["cpm"]), xytext=(8, -2), textcoords="offset points",
                ha="left", va="top", fontsize=8)
    ax.set_ylim(0.53, 0.74)
    ax.set_xlim(0.77, 0.90)
    ax.axhline(b0["cpm"], color=OI["grey"], lw=0.6, ls=":")
    ax.axvline(b0["balacc"], color=OI["grey"], lw=0.6, ls=":")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=OI["blue"], ls="", label="per-lesion weights"),
               Line2D([], [], marker="o", color=OI["orange"], ls="", label="per-candidate weights"),
               Line2D([], [], marker="o", color="black", ls="", label="γ = 0"),
               Line2D([], [], marker="^", color="black", ls="", label="γ = 2"),
               Line2D([], [], marker="o", color="black", mfc="none", ls="", label="soft labels")]
    ax.legend(handles=handles, loc="lower left", fontsize=7)
    ax.set_xlabel("per-candidate balanced accuracy")
    ax.set_ylabel("validation CPM")
    save(fig, out, "objective_study")


# ----------------------------------------------------------------------------- Ch7: the factor grid
# One encoding in every figure: ARCHITECTURE is the colour, OBJECTIVE is the marker and the line
# weight; the two reference rows are black dotted (detector score) and grey dashed (rank rule).
ARCH_COLOUR = {"Independent": OI["blue"], "Joint": OI["orange"], "Joint+Geo": OI["green"]}
OBJ_MARK = {"CE": dict(marker="o", mfc="white", lw=1.0), "pooled": dict(marker="D", lw=1.9)}
REF_STYLE = {"B0": dict(color=OI["black"], ls=":", lw=1.1), "B0-rank": dict(color=OI["grey"], ls="--", lw=1.1)}
FACTOR_COLOUR = {"jointness": OI["blue"], "geometry": OI["green"], "objective": OI["vermilion"],
                 "overall": OI["black"]}
CEIL_STYLE = dict(color=OI["black"], lw=0.6, ls="-.")


def _style(cid):
    (arch, obj), = [k for k, v in GRID.items() if v == cid]
    return dict(color=ARCH_COLOUR[arch], ms=4.5, **OBJ_MARK[obj])


def _factor_json(evo: Path):
    d = evo / "maia_47_stage/phase5/grid"
    for name in ("factor_comparisons_TEST.json", "factor_comparisons_TEST_laptop.json"):
        if (d / name).exists():
            return load_json(d / name)
    print(f"[skip] no factor_comparisons_TEST*.json in {d}")
    return None


def _seed_mean_recall(g, rung):
    seeds = sorted(g["per_seed"])
    return [float(np.mean([g["per_seed"][s][rung]["key_recall"][str(k)] for s in seeds])) for k in KEY_FP]


def fig_grid(evo: Path, out: Path, split: str):
    """The 2x3 grid: objective on the x axis, one line per architecture, seeds as markers."""
    g = _grid(evo, split)
    if g is None:
        return
    fig, ax = plt.subplots(figsize=(HALF_W * 1.75, 2.9))
    off = {"Independent": -0.07, "Joint": 0.0, "Joint+Geo": 0.07}
    for arch in ARCH:
        xs = [i + off[arch] for i in range(len(OBJ))]
        means = [g["per_rung"][GRID[(arch, o)]]["cpm_mean"] for o in OBJ]
        ax.plot(xs, means, color=ARCH_COLOUR[arch], lw=1.4, zorder=2, label=arch)
        for x, o in zip(xs, OBJ):
            seeds = g["per_rung"][GRID[(arch, o)]]["per_seed_cpm"]
            ax.plot([x] * len(seeds), seeds, ls="", marker=OBJ_MARK[o]["marker"], ms=3.5,
                    color=ARCH_COLOUR[arch], mfc="white", alpha=0.75, zorder=3)
            ax.plot(x, np.mean(seeds), ls="", marker=OBJ_MARK[o]["marker"], ms=6.5,
                    color=ARCH_COLOUR[arch], zorder=4)
    for key, lab in (("B0", "detector score"), ("B0-rank", "rank rule")):
        m = g["per_rung"][key]["cpm_mean"]
        ax.axhline(m, **REF_STYLE[key], zorder=1)
        ax.text(len(OBJ) - 0.62, m + 0.004, lab, fontsize=7.5, color=REF_STYLE[key]["color"], ha="left", va="bottom")
    ceil = g["per_rung"]["B0"]["ceiling_mean"]
    ax.axhline(ceil, **CEIL_STYLE)
    ax.text(len(OBJ) - 0.62, ceil + 0.004, "recall ceiling", fontsize=7.5, ha="left", va="bottom")
    ax.set_xticks(range(len(OBJ)))
    ax.set_xticklabels(["CE", "pooled"])
    ax.set_xlim(-0.4, len(OBJ) - 0.05)
    ax.set_xlabel("training objective")
    ax.set_ylabel(f"{'validation' if split == 'val' else 'test'} CPM")
    lo = min(g["per_rung"][k]["cpm_mean"] for k in ("B0",)) - 0.05
    ax.set_ylim(round(lo, 2), ceil + 0.04)
    ax.legend(loc="center left", bbox_to_anchor=(0.66, 0.62), fontsize=7.5, title="architecture",
              title_fontsize=7.5)
    save(fig, out, f"grid_{split}")


def fig_froc(evo: Path, out: Path, split: str):
    g = _grid(evo, split)
    if g is None:
        return
    seed = sorted(g["per_seed"])[0]
    fig, ax = plt.subplots(figsize=(HALF_W * 1.7, 2.8))
    for r in ["B0", "B0-rank", "A1", "FULL-P"]:
        d = g["per_seed"][seed][r]
        fp, rc = np.array(d["fp"]), np.array(d["recall"])
        o = np.argsort(fp)
        if r in REF_STYLE:
            kw = dict(REF_STYLE[r])
        else:
            st = _style(r)
            kw = dict(color=st["color"], lw=st["lw"], ls="-")
        ax.step(fp[o], rc[o], where="post", label=DISPLAY[r].replace("Detector", "detector score").replace("Rank rule", "rank rule"), **kw)
    ax.axhline(g["per_seed"][seed]["B0"]["ceiling"], **CEIL_STYLE)
    _froc_axes(ax)
    ax.set_ylim(0.3 if split == "test" else 0.4, 1.0)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, out, f"froc_{split}")


def fig_per_rate(evo: Path, out: Path, split: str):
    """Left: sensitivity per rate (seed mean) of the references and the complete module.
    Right: what the pooled objective adds per rate, at each architecture, per seed."""
    g = _grid(evo, split)
    if g is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    ax = axes[0]
    for r, lab in (("B0", "detector score"), ("B0-rank", "rank rule")):
        ax.plot(KEY_FP, _seed_mean_recall(g, r), marker="o", ms=2.5, label=lab, **REF_STYLE[r])
    for r in ("A1", "FULL-P"):
        ax.plot(KEY_FP, _seed_mean_recall(g, r), label=DISPLAY[r], ls="-", **_style(r))
    ax.set_xscale("log"); ax.set_xticks(KEY_FP); ax.set_xticklabels([str(k) for k in KEY_FP])
    ax.set_xlabel("false positives per volume")
    ax.set_ylabel("sensitivity, mean over seeds")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    seeds = sorted(g["per_seed"])
    for p in [q for q in FACTOR_PAIRS if q["factor"] == "objective"]:
        c = ARCH_COLOUR[p["held"]]
        per_seed = np.array([[g["per_seed"][s][p["a"]]["key_recall"][str(k)]
                              - g["per_seed"][s][p["b"]]["key_recall"][str(k)] for k in KEY_FP] for s in seeds])
        for row in per_seed:
            ax.plot(KEY_FP, row, color=c, lw=0.6, alpha=0.45)
        ax.plot(KEY_FP, per_seed.mean(0), color=c, lw=1.8, marker="D", ms=3.5, label=p["held"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log"); ax.set_xticks(KEY_FP); ax.set_xticklabels([str(k) for k in KEY_FP])
    ax.set_xlabel("false positives per volume")
    ax.set_ylabel("sensitivity gain, pooled $-$ CE")
    ax.legend(fontsize=7, loc="upper right", title="architecture", title_fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, f"per_rate_{split}")


def fig_factor_pairs(evo: Path, out: Path):
    """The seven one-switch pairs and the overall pair: one interval per seed replica."""
    rep = _factor_json(evo)
    if rep is None:
        return
    blocks = [("jointness", "H1  joint processing: Joint $-$ Independent"),
              ("geometry", "H2  relative geometry: Joint+Geo $-$ Joint"),
              ("objective", "H3  objective: pooled $-$ CE"),
              ("overall", "overall: Joint+Geo/pooled $-$ detector score")]
    held_label = {"CE": "under CE", "pooled": "under pooled", "-": "", "Independent": "Independent",
                  "Joint": "Joint", "Joint+Geo": "Joint+Geo"}
    fig, ax = plt.subplots(figsize=(FULL_W, 3.7))
    y, ticks, labels = 0.0, [], []
    for factor, title in blocks:
        rows = [p for p in rep["pairs"] if p["factor"] == factor]
        if not rows:
            continue
        ax.text(-0.205, y - 0.15, title, fontsize=8, fontweight="bold", ha="left", va="center",
                color=FACTOR_COLOUR[factor])
        y += 0.75
        for p in rows:
            for j, r in enumerate(sorted(p["per_seed"], key=lambda r: r["seed"])):
                yy = y + (j - 1) * 0.23
                ax.plot([r["lo"], r["hi"]], [yy, yy], color=FACTOR_COLOUR[factor], lw=1.3,
                        alpha=0.55 + 0.2 * j)
                ax.plot(r["delta"], yy, "o", ms=4, color=FACTOR_COLOUR[factor],
                        mfc=FACTOR_COLOUR[factor] if r["directional"] else "white")
            ticks.append(y)
            labels.append(held_label[p["held"]] if factor != "overall" else "")
            y += 1.0
        y += 0.35
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.6, -0.7)
    ax.set_xlim(-0.21, 0.21)
    ax.set_xlabel("paired difference in test CPM (95 % bootstrap over volumes; seeds 0, 1, 2 from top to bottom)")
    save(fig, out, "factor_pairs_test")


def fig_lambda0(evo: Path, out: Path):
    d = load_json(evo / "maia_47_stage/phase4/grid/sub_ablations_lam0.json")["lambda0_diagnostics"]
    g = _grid(evo, "val")
    fig, ax = plt.subplots(figsize=(HALF_W * 1.5, 2.6))
    variants = ["A2", "FULL", "FULL-P"]
    cols = [OI["orange"], OI["green"], OI["green"]]
    x = np.arange(len(variants))
    for i, (v, c) in enumerate(zip(variants, cols)):
        lam0 = [r["val_cpm"] for r in d if r["variant"] == v]
        full = g["per_rung"][v]["per_seed_cpm"]
        ax.bar(i - 0.2, np.mean(full), width=0.38, color=c, alpha=0.9, label="λ swept" if i == 0 else None)
        ax.bar(i + 0.2, np.mean(lam0), width=0.38, color=c, alpha=0.35, hatch="//", label="λ = 0" if i == 0 else None)
        ax.plot([i - 0.2] * 3, full, "o", color="white", mec="black", ms=3.5, zorder=5)
        ax.plot([i + 0.2] * 3, lam0, "o", color="white", mec="black", ms=3.5, zorder=5)
    ax.axhline(g["per_rung"]["B2"]["cpm_mean"], color=OI["grey"], lw=0.8, ls="--")
    ax.text(2.45, g["per_rung"]["B2"]["cpm_mean"] + 0.01, DISPLAY["B2"], fontsize=7.5, color=OI["grey"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[v].replace("/", "/\n") for v in variants], fontsize=7.5)
    ax.set_ylabel("validation CPM")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper center", fontsize=7, ncol=2, bbox_to_anchor=(0.5, 1.12))
    save(fig, out, "lambda0")


def fig_strata(evo: Path, out: Path):
    """Set-size bins for {Independent, Joint} x {CE, pooled}: where would attention help?"""
    s = load_json(evo / "maia_47_stage/phase5/grid/stratified_TEST.json")
    show = ["B1", "B2", "B1-P", "A2-P"]
    bins = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), sharey=True)
    for ax, seed in zip(axes, sorted(s["per_seed"])):
        ps = s["per_seed"][seed]
        n = {b: sum(1 for v in ps["bin_of_volume"].values() if v == b) for b in bins}
        x = np.arange(3)
        for r in show:
            ax.plot(x, [ps["rungs"][r][b]["cpm"] for b in bins], ls="-", label=DISPLAY[r], **_style(r))
        ax.plot(x, [ps["rungs"]["B0"][b]["ceiling"] for b in bins], label="recall ceiling", **CEIL_STYLE)
        e = ps["edges"]
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\leq${e[0]:.0f}\n(n={n['low']})", f"{e[0]:.0f}–{e[1]:.0f}\n(n={n['mid']})", f">{e[1]:.0f}\n(n={n['high']})"], fontsize=7.5)
        ax.set_title(f"seed {seed}", fontsize=8)
        ax.set_xlabel("candidates in the set", fontsize=8)
    axes[0].set_ylabel("test CPM in bin")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.05, 1, 1))
    save(fig, out, "strata_test")


def fig_val_vs_test(evo: Path, out: Path):
    """Every reported pair, validation against test. Differences are exact CPM differences
    (mean and spread over the three seed replicas), so validation needs no stored comparison."""
    gv = _grid(evo, "val"); gt = _grid(evo, "test")
    if gv is None or gt is None:
        return
    held_short = {"CE": "CE", "pooled": "pooled", "Independent": "Ind.", "Joint": "Joint", "Joint+Geo": "J+Geo", "-": ""}
    fig, ax = plt.subplots(figsize=(HALF_W * 1.7, 3.3))
    for p in tuple(FACTOR_PAIRS) + (OVERALL,):
        d = {}
        for name, g in (("val", gv), ("test", gt)):
            v = np.array(g["per_rung"][p["a"]]["per_seed_cpm"]) - np.array(g["per_rung"][p["b"]]["per_seed_cpm"])
            d[name] = (v.mean(), v.std())
        c = FACTOR_COLOUR[p["factor"]]
        ax.errorbar(d["val"][0], d["test"][0], xerr=d["val"][1], yerr=d["test"][1], fmt="o", ms=4, color=c,
                    elinewidth=0.8, capsize=0)
        if p["factor"] != "overall":
            ax.annotate(held_short[p["held"]], (d["val"][0], d["test"][0]), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.5, color=c)
    lim = (-0.08, 0.15)
    ax.plot(lim, lim, color=OI["grey"], lw=0.6, ls=":")
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("validation: difference in CPM (mean $\\pm$ std over seeds)")
    ax.set_ylabel("test: difference in CPM")
    from matplotlib.lines import Line2D
    names = {"jointness": "H1 jointness", "geometry": "H2 geometry", "objective": "H3 objective", "overall": "overall"}
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FACTOR_COLOUR[k], label=v) for k, v in names.items()],
              loc="upper left", fontsize=7)
    save(fig, out, "val_vs_test")


# ----------------------------------------------------------------------------- artifact audit (2026-09-15)
AUDIT_CACHE = "outputs_iso/phase1/cache/3f94f17f04055ffa8f5a6c7b33495d64d630112b98bdc31e32f1b5b258e88830"
AUDIT_DIR = "outputs_iso/phase3/artifact_audit"
# The four walkthrough candidates of the audit section (validation pool, picked as the top-scoring
# member of their class; see results/RESULTS_ARTIFACT_AUDIT.md). Row index into signatures_val.csv.
AUDIT_EXEMPLARS = [
    ("full_seed0", 103, "neg", "narrow_column", "column-shaped FP"),
    ("full_seed2", 118, "neg", "bounded_blob", "look-alike FP"),
    ("full_seed0", 103, "pos", "bounded_blob", "lesion, bounded"),
    ("full_seed0", 111, "pos", "shadowing_blob", "lesion, shadowing"),
]
MECH_GROUPS = [
    ("column shadow", ["narrow_column", "broad_column"]),
    ("position", ["skin", "marginal", "deep", "reverberation"]),
    ("shadowing blob", ["shadowing_blob"]),
    ("bounded blob", ["bounded_blob"]),
    ("other", ["other"]),
]


def fig_audit_mechanisms(evo: Path, out: Path):
    """Four candidates and their beam-line profiles: what the mechanism labels mean."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from abus_jcr.probe import artifact_audit as AA
    sig = pd.read_csv(evo / AUDIT_DIR / "signatures_val.csv")
    knobs = AA.Knobs()
    iso = knobs.iso_mm
    fig, axes = plt.subplots(2, len(AUDIT_EXEMPLARS), figsize=(FULL_W, 3.9),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for j, (det, vid, lab, mech, title) in enumerate(AUDIT_EXEMPLARS):
        rows = sig[(sig.detector_of_origin == det) & (sig.public_id == vid) & (sig.label == lab) & (sig.mechanism == mech)]
        r = rows.sort_values("score_max", ascending=False).iloc[0]
        vol = np.load(evo / AUDIT_CACHE / "vol" / f"VOL_{vid}.npy", mmap_mode="r")
        D0, D1, D2 = vol.shape
        c = [r.cen_d0, r.cen_d1, r.cen_d2]; e = [r.ext_d0, r.ext_d1, r.ext_d2]
        bp = AA.beam_profile(vol, c, e, knobs)
        z = int(round(c[2]))
        frame = np.asarray(vol[:, :, z]).T                      # rows = depth, cols = lateral
        half = 90
        x0, x1 = int(max(0, c[0] - half)), int(min(D0, c[0] + half))
        ax = axes[0, j]
        ax.imshow(frame[:, x0:x1], cmap="gray", vmin=0, vmax=0.8, aspect="equal",
                  extent=[0, (x1 - x0) * iso, D1 * iso, 0])
        col = OI["blue"] if lab == "pos" else OI["vermilion"]
        ax.add_patch(plt.Rectangle(((c[0] - e[0] / 2 - x0) * iso, (c[1] - e[1] / 2) * iso), e[0] * iso, e[1] * iso,
                                   fill=False, ec=col, lw=1.2))
        w = bp["l1"] - bp["l0"]; off = int(round(e[0] / 2 + w / 2 + round(knobs.flank_gap_mm / iso)))
        for a in (bp["l0"] - off, bp["l0"] + off):
            ax.add_patch(plt.Rectangle(((a - x0) * iso, 0), w * iso, D1 * iso, fill=False, ec=OI["yellow"], lw=0.7, ls="--"))
        ax.add_patch(plt.Rectangle(((bp["l0"] - x0) * iso, 0), w * iso, D1 * iso, fill=False, ec="white", lw=0.7, ls="--"))
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_ylabel("depth (mm)" if j == 0 else ""); ax.tick_params(labelsize=7)
        if j > 0:
            ax.set_yticks([])
        ax = axes[1, j]
        d = np.arange(D1) * iso
        ax.plot(d, bp["col"], color=OI["black"], lw=1.0, label="column")
        ax.plot(d, bp["flank"], color=OI["orange"], lw=1.0, label="flanks")
        ax.plot(d, bp["contrast"], color=OI["green"], lw=1.0, label="contrast")
        ax.axvspan(bp["top"] * iso, bp["bot"] * iso, color=col, alpha=0.15)
        ax.axhline(-knobs.tau, color=OI["grey"], lw=0.7, ls="--")
        ax.set_xlabel("depth (mm)")
        ax.set_ylim(-0.35, 0.75); ax.set_xlim(0, 48)
        ax.text(0.97, 0.95, f"below {r.shadow_run_mm:.0f} mm\nabove {r.run_above_mm:.1f} mm\nstripe {r.stripe_width_mm:.1f} mm",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        if j == 0:
            ax.set_ylabel("intensity")
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6, handlelength=1.2)
        else:
            ax.set_yticks([])
        print(f"  exemplar {title}: vol {vid} {det} score {r.score_max:.2f} diag {r.diag_mm:.1f} mm")
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    save(fig, out, "audit_mechanisms")


def fig_audit_partition(evo: Path, out: Path):
    """Left: mechanism partition (grouped) FP vs TP on both pools. Right: metric change when each class is demoted."""
    part = {sp: pd.DataFrame(load_json(evo / AUDIT_DIR / f"artifact_audit_{sp}.json")["partition"]) for sp in ("val", "train")}
    rules = {sp: pd.DataFrame(load_json(evo / AUDIT_DIR / f"artifact_audit_rules_{sp}.json")["interventions"]) for sp in ("val", "train")}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    bars = [("val", "neg"), ("val", "pos"), ("train", "neg"), ("train", "pos")]
    names = ["val.\nFP", "val.\nTP", "train\nFP", "train\nTP"]
    gcol = [OI["vermilion"], OI["orange"], OI["purple"], OI["sky"], OI["grey"]]
    for i, (sp, lab) in enumerate(bars):
        p = part[sp][part[sp]["label"] == lab]
        per_det = p.pivot_table(index="detector", columns="mechanism", values="frac_pooled", fill_value=0.0)
        base = 0.0
        for gi, (gname, members) in enumerate(MECH_GROUPS):
            v = float(per_det[[m for m in members if m in per_det]].sum(axis=1).mean())
            ax.bar(i, v, bottom=base, color=gcol[gi], width=0.7, label=gname if i == 0 else None)
            if v > 0.06:
                ax.text(i, base + v / 2, f"{100 * v:.0f}", ha="center", va="center", fontsize=7, color="white" if gi != 4 else "black")
            base += v
        print(f"  partition {sp} {lab}: " + ", ".join(f"{g}={100 * float(per_det[[m for m in mm if m in per_det]].sum(axis=1).mean()):.1f}%" for g, mm in MECH_GROUPS))
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("fraction of candidates"); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), fontsize=6.5, handlelength=1.0)
    ax = axes[1]
    classes = [("column shadow", "demote narrow+broad column"), ("skin", "demote skin"), ("marginal", "demote marginal"),
               ("reverberation", "demote reverberation"), ("shadowing blob", "demote shadowing_blob"),
               ("bounded blob", "demote bounded_blob"), ("other", "demote other")]
    x = np.arange(len(classes))
    tr = rules["train"]
    ax.bar(x, [float(tr[tr["rule"] == r]["delta"].iloc[0]) if (tr["rule"] == r).any() else 0.0 for _, r in classes],
           color=OI["grey"], width=0.6, label="training pool")
    va = rules["val"]
    for k, (_, r) in enumerate(classes):
        d = va[va["rule"] == r]["delta"].to_numpy(float)
        ax.plot(np.full(len(d), k) + np.linspace(-0.12, 0.12, len(d)), d, "o", color=OI["blue"], ms=3.5,
                label="validation, three seeds" if k == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([c for c, _ in classes], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("metric change, class demoted")
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "audit_partition")
    for sp in ("val", "train"):
        print(f"  interventions {sp}:")
        print(rules[sp][["detector", "rule", "n_fp", "n_tp", "delta"]].to_string(index=False))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo", type=Path, default=Path("/Volumes/Evo"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "figures")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--only", nargs="*", default=None, help="figure names to regenerate")
    ap.add_argument("--png-dir", type=Path, default=None, help="also write PNG previews here")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    global PNG_DIR
    if a.png_dir is not None:
        a.png_dir.mkdir(parents=True, exist_ok=True)
        PNG_DIR = a.png_dir
    if a.split == "val":
        jobs = {
            "froc_schematic": lambda: fig_froc_schematic(a.out),
            "fidelity": lambda: fig_fidelity(a.evo, a.out),
            "selection_curves": lambda: fig_selection_curves(a.evo, a.out),
            "linker_sweeps": lambda: fig_linker_sweeps(a.evo, a.out),
            "operating_point": lambda: fig_operating_point(a.evo, a.out),
            "baseline_froc": lambda: fig_baseline_froc(a.evo, a.out),
            "headroom": lambda: fig_headroom(a.evo, a.out),
            "audit_elongation": lambda: fig_audit_elongation(a.evo, a.out),
            "audit_mechanisms": lambda: fig_audit_mechanisms(a.evo, a.out),
            "audit_partition": lambda: fig_audit_partition(a.evo, a.out),
            "pool_priors": lambda: fig_pool_priors(a.evo, a.out),
            "objective_study": lambda: fig_objective_study(a.evo, a.out),
            "grid_val": lambda: fig_grid(a.evo, a.out, "val"),
            "lambda0": lambda: fig_lambda0(a.evo, a.out),
        }
    else:
        jobs = {
            "grid_test": lambda: fig_grid(a.evo, a.out, "test"),
            "froc_test": lambda: fig_froc(a.evo, a.out, "test"),
            "per_rate_test": lambda: fig_per_rate(a.evo, a.out, "test"),
            "factor_pairs_test": lambda: fig_factor_pairs(a.evo, a.out),
            "strata_test": lambda: fig_strata(a.evo, a.out),
            "val_vs_test": lambda: fig_val_vs_test(a.evo, a.out),
        }
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
