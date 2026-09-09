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
RUNG_ORDER = ["B0", "B0-spread", "B0-rank", "B1", "B2", "A1", "A2", "FULL", "B1-P", "A2-P", "FULL-P"]
RUNG_COLOUR = {
    "B0": OI["black"], "B0-spread": OI["grey"], "B0-rank": OI["grey"],
    "B1": OI["sky"], "B2": OI["blue"], "A1": OI["purple"], "A2": OI["green"], "FULL": OI["orange"],
    "B1-P": OI["sky"], "A2-P": OI["green"], "FULL-P": OI["vermilion"],
}
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
    rows = [("B0\n(detector\nscore)", "score_max"), ("bound of any\nmonotone\nrescaling", "global_monotone"),
            ("B0-rank\n(label-free)", "volume_neutral_anchored"), ("per-volume\noracle", "per_vol_oracle")]
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
    ax.annotate("B0 (detector score)", (b0["balacc"], b0["cpm"]), xytext=(8, -2), textcoords="offset points",
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


# ----------------------------------------------------------------------------- Ch5
def fig_ladder(evo: Path, out: Path, split: str):
    g = _grid(evo, split)
    if g is None:
        return
    rungs = [r for r in RUNG_ORDER if r in g["per_rung"]]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.8))
    x = np.arange(len(rungs))
    for i, r in enumerate(rungs):
        pr = g["per_rung"][r]
        seeds = pr["per_seed_cpm"]
        ax.plot([i] * len(seeds), seeds, "o", color=RUNG_COLOUR[r], ms=4, alpha=0.6)
        ax.plot([i - 0.25, i + 0.25], [pr["cpm_mean"]] * 2, color=RUNG_COLOUR[r], lw=2.5)
    for key, lab, c, ls in [("B0", "B0", OI["black"], ":"), ("B0-rank", "B0-rank", OI["grey"], "--")]:
        if key in g["per_rung"]:
            ax.axhline(g["per_rung"][key]["cpm_mean"], color=c, lw=0.8, ls=ls)
    ax.axhline(g["per_rung"]["B0"]["ceiling_mean"], color=OI["green"], lw=0.8, ls="--")
    ax.text(len(rungs) - 0.6, g["per_rung"]["B0"]["ceiling_mean"] + 0.005, "recall ceiling", ha="right",
            va="bottom", fontsize=8, color=OI["green"])
    for xx in [2.5, 7.5]:
        ax.axvline(xx, color=OI["grey"], lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rungs, rotation=25, ha="right")
    ax.set_ylabel(f"{split} CPM")
    ax.set_ylim(0.6, 1.0)
    save(fig, out, "ladder" if split == "val" else "ladder_test")


def fig_froc_ladder(evo: Path, out: Path, split: str):
    g = _grid(evo, split)
    if g is None:
        return
    show = ["B0", "B0-rank", "B1", "B2", "A2", "FULL", "B1-P", "FULL-P"]
    seed = sorted(g["per_seed"])[0]
    fig, ax = plt.subplots(figsize=(HALF_W * 1.7, 2.8))
    for r in show:
        if r not in g["per_seed"][seed]:
            continue
        d = g["per_seed"][seed][r]
        fp, rc = np.array(d["fp"]), np.array(d["recall"])
        o = np.argsort(fp)
        ls = "--" if r in ("B0", "B0-rank") else "-"
        lw = 1.8 if r.endswith("-P") else 1.1
        ax.step(fp[o], rc[o], where="post", color=RUNG_COLOUR[r], lw=lw, ls=ls, label=r)
    ax.axhline(g["per_seed"][seed]["B0"]["ceiling"], color=OI["green"], lw=0.6, ls="--")
    _froc_axes(ax)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    save(fig, out, "froc_ladder" if split == "val" else "froc_ladder_test")


def fig_per_rate(evo: Path, out: Path, split: str):
    g = _grid(evo, split)
    if g is None:
        return
    show = ["B0", "B0-rank", "B1", "B2", "A2", "FULL", "B1-P", "A2-P", "FULL-P"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    ax = axes[0]
    for r in show:
        if r not in g["per_rung"]:
            continue
        kr = g["per_rung"][r]["key_recall"]
        ax.plot(KEY_FP, [kr[str(k)] for k in KEY_FP], marker="o", ms=3, lw=1.6 if r.endswith("-P") else 1.0,
                ls="--" if r in ("B0", "B0-rank") else "-", color=RUNG_COLOUR[r], label=r)
    ax.set_xscale("log")
    ax.set_xticks(KEY_FP)
    ax.set_xticklabels([str(k) for k in KEY_FP])
    ax.set_xlabel("false positives per volume")
    ax.set_ylabel("mean sensitivity over seeds")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    ax = axes[1]
    pairs = [("FULL-P", "FULL", OI["vermilion"]), ("B1-P", "B1", OI["sky"]), ("A2-P", "A2", OI["green"])]
    for a, b, c in pairs:
        for seed in sorted(g["per_seed"]):
            if a not in g["per_seed"][seed] or b not in g["per_seed"][seed]:
                continue
            ka, kb = g["per_seed"][seed][a]["key_recall"], g["per_seed"][seed][b]["key_recall"]
            ax.plot(KEY_FP, [ka[str(k)] - kb[str(k)] for k in KEY_FP], "o-", ms=2.5, lw=0.9, color=c, alpha=0.8,
                    label=f"{a} − {b}" if seed == sorted(g["per_seed"])[0] else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks(KEY_FP)
    ax.set_xticklabels([str(k) for k in KEY_FP])
    ax.set_xlabel("false positives per volume")
    ax.set_ylabel("gain from the pooled objective")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(w_pad=1.5, rect=(0, 0.04, 1, 1))
    save(fig, out, "per_rate" if split == "val" else "per_rate_test")


PRE_REG = ["FULL-B2", "A1-B2", "A2-B2", "B2-B1", "B1-B0", "FULL-A1"]
ADDED = ["FULL-P-B2", "FULL-P-FULL", "A2-P-A2", "FULL-P-B1-P", "FULL-B0-rank", "FULL-P-B0-rank", "B2-B0-rank"]


def _pretty(c):
    for a in ["FULL-P", "B0-rank", "B1-P", "A2-P"]:
        c = c.replace(a, a.replace("-", "§"))
    a, b = c.split("-", 1)
    return f"{a} − {b}".replace("§", "-")


def fig_comparisons(evo: Path, out: Path, split: str):
    g = _grid(evo, split)
    if g is None:
        return
    comps = [c for c in PRE_REG if c in g["comparisons"]] + [c for c in ADDED if c in g["comparisons"]]
    fig, ax = plt.subplots(figsize=(FULL_W, 3.6))
    y = 0
    ticks, labels = [], []
    seed_cols = [OI["blue"], OI["green"], OI["orange"]]
    for i, c in enumerate(comps):
        if c == ADDED[0] and any(k in g["comparisons"] for k in PRE_REG):
            ax.axhline(y - 0.5, color=OI["grey"], lw=0.6)
            ax.text(-0.295, y - 0.42, "added after the substrate measurement", fontsize=7,
                    color=OI["grey"], ha="left", va="top")
            ax.text(0.195, -0.42, "pre-registered", fontsize=7, color=OI["grey"], ha="right", va="top")
        ps = g["comparisons"][c]["per_seed"]
        for j, p in enumerate(ps):
            yy = y + (j - 1) * 0.22
            ax.plot([p["lo"], p["hi"]], [yy, yy], color=seed_cols[j], lw=1.2)
            ax.plot(p["delta"], yy, "o", color=seed_cols[j], ms=3.5)
        ticks.append(y)
        labels.append(_pretty(c))
        y += 1
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(comps) - 0.5, -0.9)
    ax.set_xlabel(f"paired difference in {split} CPM (95 % volume bootstrap, per seed)")
    ax.set_xlim(-0.3, 0.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=3.5, label=f"seed {i}") for i, c in enumerate(seed_cols)],
              loc="lower left", fontsize=7)
    save(fig, out, "comparisons" if split == "val" else "comparisons_test")


def fig_lambda0(evo: Path, out: Path):
    d = load_json(evo / "maia_47_stage/phase4/grid/sub_ablations_lam0.json")["lambda0_diagnostics"]
    g = _grid(evo, "val")
    fig, ax = plt.subplots(figsize=(HALF_W * 1.4, 2.5))
    variants = ["A2", "FULL", "FULL-P"]
    x = np.arange(len(variants))
    for i, v in enumerate(variants):
        lam0 = [r["val_cpm"] for r in d if r["variant"] == v]
        full = g["per_rung"][v]["per_seed_cpm"]
        ax.bar(i - 0.2, np.mean(full), width=0.38, color=RUNG_COLOUR[v], alpha=0.9, label="λ swept" if i == 0 else None)
        ax.bar(i + 0.2, np.mean(lam0), width=0.38, color=RUNG_COLOUR[v], alpha=0.35, hatch="//", label="λ = 0" if i == 0 else None)
        ax.plot([i - 0.2] * 3, full, "o", color="white", mec="black", ms=3.5, zorder=5)
        ax.plot([i + 0.2] * 3, lam0, "o", color="white", mec="black", ms=3.5, zorder=5)
    ax.axhline(g["per_rung"]["B2"]["cpm_mean"], color=OI["grey"], lw=0.8, ls="--")
    ax.text(2.45, g["per_rung"]["B2"]["cpm_mean"] + 0.01, "B2", fontsize=8, color=OI["grey"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("validation CPM")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper center", fontsize=7, ncol=2, bbox_to_anchor=(0.5, 1.12))
    save(fig, out, "lambda0")


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
            "pool_priors": lambda: fig_pool_priors(a.evo, a.out),
            "objective_study": lambda: fig_objective_study(a.evo, a.out),
            "ladder": lambda: fig_ladder(a.evo, a.out, "val"),
            "froc_ladder": lambda: fig_froc_ladder(a.evo, a.out, "val"),
            "per_rate": lambda: fig_per_rate(a.evo, a.out, "val"),
            "comparisons": lambda: fig_comparisons(a.evo, a.out, "val"),
            "lambda0": lambda: fig_lambda0(a.evo, a.out),
        }
    else:
        jobs = {
            "ladder_test": lambda: fig_ladder(a.evo, a.out, "test"),
            "froc_ladder_test": lambda: fig_froc_ladder(a.evo, a.out, "test"),
            "per_rate_test": lambda: fig_per_rate(a.evo, a.out, "test"),
            "comparisons_test": lambda: fig_comparisons(a.evo, a.out, "test"),
        }
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
