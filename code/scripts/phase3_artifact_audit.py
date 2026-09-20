"""Artifact-signature audit of a frozen candidate pool. READ-ONLY: writes CSV/JSON/figures only.

Three blocks (module docstring of ``abus_jcr.probe.artifact_audit`` says why):

  Q1  per-candidate beam-line signatures + mechanism partition, FP vs TP, size first
  Q2  per-volume arrangement statistics of the FP field against permutation nulls
  Q3  grouped-by-volume out-of-fold reader of the features, scored by the OFFICIAL evaluator

There is no verdict block and no pass/fail gate. Numbers are printed, figures are drawn, and the
reading is written by hand afterwards.

Usage (laptop, isotropic substrate):
    export ABUS_AXIS_PROFILE=measured
    python scripts/phase3_artifact_audit.py --split val \
        --phase1-out /Volumes/Evo/outputs_iso/phase1 --out-root /Volumes/Evo/outputs_iso/phase3 \
        --data-root /Volumes/Evo/data
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.candidates.record import read_candidate_record, to_official_pred_csv
from abus_jcr.eval.froc import evaluate_froc, cpm as froc_cpm, key_recall
from abus_jcr.probe import artifact_audit as AA
from _phase3_common import add_phase3_paths, cache_root, load_official_gt, profile_banner


def _fmt(x, w=8, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else f"{'nan':>{w}}"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    return o


# ------------------------------------------------------------------ printing blocks

def print_contrast(rep: pd.DataFrame, title: str):
    print(f"\n# {title}")
    print("  (delta = Cliff's, FP vs TP, POSITIVE = larger on FPs; perVol = median over (detector, volume) sets;")
    print("   sign = fraction of sets with delta > 0; size-matched = delta within diagonal quartiles, weighted)")
    print(f"  {'feature':>18} {'TP med':>8} {'FP med':>8} {'pooled':>8} {'perVol':>8} {'sign':>6} {'n_vol':>6} {'sizeM':>8}  per-quartile delta")
    for _, r in rep.iterrows():
        bins = " ".join(_fmt(d, 6, 2) for d in r["delta_by_size_bin"])
        print(f"  {r['feature']:>18} {_fmt(r['tp_median'])} {_fmt(r['fp_median'])} {_fmt(r['delta_pooled'])} "
              f"{_fmt(r['delta_pervol_median'])} {_fmt(r['sign_frac'], 6, 2)} {int(r['n_vol']):>6} "
              f"{_fmt(r['delta_size_matched'])}  {bins}")


def print_partition(part: pd.DataFrame, title: str):
    print(f"\n# {title}")
    dets = sorted(part["detector"].unique())
    print(f"  {'detector':>12} {'label':>5} {'n':>6} | " + " ".join(f"{m[:9]:>9}" for m in AA.MECHANISMS))
    for det in dets:
        for lab in ("neg", "pos"):
            g = part[(part["detector"] == det) & (part["label"] == lab)].set_index("mechanism")
            if g.empty:
                continue
            n = int(g["n"].sum())
            print(f"  {det:>12} {('FP' if lab == 'neg' else 'TP'):>5} {n:>6} | "
                  + " ".join(_fmt(g.loc[m, 'frac_pooled'], 9, 3) for m in AA.MECHANISMS))
    for lab in ("neg", "pos"):
        g = part[part["label"] == lab].groupby("mechanism")["frac_pooled"].mean()
        print(f"  {'MEAN over det':>12} {('FP' if lab == 'neg' else 'TP'):>5} {'':>6} | "
              + " ".join(_fmt(g.get(m, 0.0), 9, 3) for m in AA.MECHANISMS))


# ------------------------------------------------------------------ Q2 driver

def arrangement_block(sig: pd.DataFrame, knobs: AA.Knobs, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    iso = knobs.iso_mm
    per_vol = []
    hits = []
    for (det, vid), g in sig.groupby(["detector_of_origin", "public_id"], sort=True):
        fp = g[g["label"] == "neg"]; tp = g[g["label"] == "pos"]
        p_fp = fp[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float) * iso
        rec = {"detector": det, "public_id": int(vid), "n_fp": int(len(fp)), "n_tp": int(len(tp))}
        if len(fp) >= 5:
            rec.update(AA.arrangement_null(p_fp, knobs, rng, n_perm=n_perm))
        if len(tp) >= 1 and len(fp) >= 1:
            hit = tp.sort_values("score_max", ascending=False).iloc[0]
            h = np.array([hit.cen_d0, hit.cen_d1, hit.cen_d2]) * iso
            hv = AA.hit_vs_field(p_fp, h, knobs); hv.update({"detector": det, "public_id": int(vid)})
            hits.append(hv)
        per_vol.append(rec)
    # depth-band ratio: within-volume SD of FP depth / pooled SD of FP depth (per detector)
    ratios = []
    for det, g in sig[sig["label"] == "neg"].groupby("detector_of_origin"):
        pooled_sd = float(g["depth_mm"].std(ddof=0))
        for vid, gv in g.groupby("public_id"):
            if len(gv) >= 5 and pooled_sd > 0:
                ratios.append({"detector": det, "public_id": int(vid),
                               "depth_band_ratio": float(gv["depth_mm"].std(ddof=0) / pooled_sd)})
    return {"per_volume": per_vol, "hit_vs_field": hits, "depth_band_ratio": ratios}


def print_arrangement(arr: dict):
    pv = pd.DataFrame([r for r in arr["per_volume"] if "same_column" in r])
    print("\n# Q2 — ARRANGEMENT of the FP field, per (detector, volume) with >= 5 FPs, against permutation nulls")
    print("  same_column: pairs within 3 mm laterally and in sweep, > 5 mm apart in depth (null: lateral permuted)")
    print("  sweep_chain: pairs within 3 mm laterally and in depth, > 5 mm apart in sweep (null: lateral permuted)")
    print("  band_regularity: autocorrelation peak of the lateral-offset histogram inside a depth band (null: lateral uniform)")
    print(f"  {'detector':>12} {'n_vol':>5} | {'stat':>16} {'obs med':>8} {'null med':>8} {'z med':>7} {'frac>p95':>9}")
    if pv.empty:
        print("  (no volume with >= 5 FPs)"); return
    for det, g in pv.groupby("detector"):
        for stat in ("same_column", "sweep_chain", "band_regularity"):
            obs = np.array([r["obs"] for r in g[stat]], float)
            nul = np.array([r["null_mean"] for r in g[stat]], float)
            z = np.array([r["z"] for r in g[stat]], float)
            ex = np.array([bool(r["exceeds"]) for r in g[stat]])
            ok = np.isfinite(obs) & np.isfinite(z)
            print(f"  {det:>12} {len(g):>5} | {stat:>16} {_fmt(np.nanmedian(obs), 8, 2)} {_fmt(np.nanmedian(nul), 8, 2)} "
                  f"{_fmt(np.median(z[ok]) if ok.any() else np.nan, 7, 2)} {_fmt(ex[ok].mean() if ok.any() else np.nan, 9, 2)}")
    dr = pd.DataFrame(arr["depth_band_ratio"])
    if not dr.empty:
        print("\n  depth_band_ratio = within-volume SD of FP depth / pooled SD across the detector's volumes (<< 1 = a volume-specific band)")
        for det, g in dr.groupby("detector"):
            print(f"  {det:>12} median {_fmt(g['depth_band_ratio'].median(), 6, 2)}  IQR {_fmt(g['depth_band_ratio'].quantile(.25), 5, 2)}-{_fmt(g['depth_band_ratio'].quantile(.75), 5, 2)}")
    hv = pd.DataFrame(arr["hit_vs_field"])
    if not hv.empty:
        print("\n  hit vs field: the best hit's counts against the FPs minus the median FP's counts against the other FPs (per set)")
        for det, g in hv.groupby("detector"):
            for nm in ("band", "column", "chain"):
                d = g[f"hit_{nm}"] - g[f"fp_{nm}_med"]
                d = d[np.isfinite(d)]
                print(f"  {det:>12} {nm:>7}: hit med {_fmt(g[f'hit_{nm}'].median(), 5, 1)} FP med {_fmt(g[f'fp_{nm}_med'].median(), 5, 1)} "
                      f"diff med {_fmt(d.median(), 5, 1)} sign(hit>FP) {_fmt((d > 0).mean(), 5, 2)} n {len(d)}")


# ------------------------------------------------------------------ Q3 driver

def feature_matrix(sig: pd.DataFrame, blocks: list) -> np.ndarray:
    cols = []
    for b in blocks:
        if b == "base":
            cols += ["score_max", "score_mean", "score_std", "log_ext_d0", "log_ext_d1", "log_ext_d2", "rank_norm"]
        elif b == "signature":
            cols += list(AA.SIGNATURE_FEATURES)
        elif b == "context":
            cols += ["log_n_band", "log_n_column", "log_n_chain", "log_set_size"]
    return sig[cols].to_numpy(float)


def ranking_block(sig: pd.DataFrame, gt_df: pd.DataFrame, per_detector: bool, k_folds: int, seed: int) -> list:
    sig = sig.copy()
    for a in range(3):
        sig[f"log_ext_d{a}"] = np.log1p(sig[f"ext_d{a}"].astype(float))
    for c in AA.CONTEXT_FEATURES:
        sig[f"log_{c}"] = np.log1p(sig[c].astype(float))
    groups_of = sig.groupby("detector_of_origin") if per_detector else [("ALL", sig)]
    rows = []
    for det, g in groups_of:
        g = g.reset_index(drop=True)
        y = (g["label"] == "pos").to_numpy()
        fit = g["label"].isin(["pos", "neg"]).to_numpy()
        vols = g["public_id"].to_numpy()
        gt = gt_df[gt_df["public_id"].isin(np.unique(vols))]
        configs = [("B0 score_max", None), ("base", ["base"]), ("+signature", ["base", "signature"]),
                   ("+context", ["base", "context"]), ("+both", ["base", "signature", "context"])]
        for name, blocks in configs:
            if blocks is None:
                prob = g["score_max"].to_numpy(float)
            else:
                X = feature_matrix(g, blocks)
                prob = AA.grouped_oof_scores(X, y.astype(float), vols, fit, k=k_folds, seed=seed)
            gg = g.copy(); gg["_p"] = np.clip(prob, 0.0, 1.0 - 1e-6)
            pred = to_official_pred_csv(gg, "_p")
            with contextlib.redirect_stderr(io.StringIO()):      # the vendored oracle prints tqdm bars
                res = evaluate_froc(gt, pred)
            kr = key_recall(res)
            rows.append({"detector": det, "config": name, "cpm": froc_cpm(res),
                         "auc": AA.auc(gg.loc[fit, "_p"], y[fit]),
                         "r_at": {str(k): float(v) for k, v in kr.items()}})
    return rows


def print_ranking(rows: list):
    print("\n# Q3 — RANKING INFORMATION: grouped-by-volume out-of-fold L2-logistic reader, scored by the OFFICIAL evaluator")
    print("  base = {score_max, score_mean, score_std, log extents x3, rank_norm} (what the token already carries)")
    print("  +signature = Q1 beam-line/position features; +context = label-free counts of candidates in the same band/column/chain")
    print(f"  {'detector':>12} {'config':>14} {'CPM':>7} {'AUC':>6} | " + " ".join(f"R@{k:<5}" for k in C.KEY_FP))
    for r in rows:
        print(f"  {r['detector']:>12} {r['config']:>14} {_fmt(r['cpm'], 7, 4)} {_fmt(r['auc'], 6, 3)} | "
              + " ".join(_fmt(r['r_at'][str(k)], 7, 3) for k in C.KEY_FP))


# ------------------------------------------------------------------ figures

def make_figures(sig: pd.DataFrame, part: pd.DataFrame, arr: dict, rank_rows: list, load_vol, out_dir: Path, tag: str, knobs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = sig[sig["label"].isin(["pos", "neg"])].copy()
    g["qbin"] = pd.qcut(g["diag_mm"], 4, labels=False, duplicates="drop")
    feats = ["diag_mm", "inside_contrast", "prox_contrast", "distal_contrast", "cap_bright", "shadow_run_mm",
             "run_above_mm", "stripe_width_mm", "periodicity", "depth_mm", "contact_edge_mm", "abs_intensity"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, f in zip(axes.ravel(), feats):
        data, pos, colors = [], [], []
        for q in sorted(g["qbin"].dropna().unique()):
            for lab, col, off in (("neg", "tab:red", -0.18), ("pos", "tab:blue", 0.18)):
                v = g[(g["qbin"] == q) & (g["label"] == lab)][f].dropna().to_numpy()
                if v.size:
                    data.append(v); pos.append(q + off); colors.append(col)
        if data:
            bp = ax.boxplot(data, positions=pos, widths=0.3, showfliers=False, patch_artist=True)
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col); patch.set_alpha(0.5)
        ax.set_title(f, fontsize=9); ax.set_xticks(range(4)); ax.set_xticklabels([f"Q{i+1}" for i in range(4)], fontsize=8)
    fig.suptitle(f"{tag}: FP (red) vs TP (blue) by diagonal quartile — size is controlled by reading within a quartile")
    fig.tight_layout(); fig.savefig(out_dir / f"fig_signatures_{tag}.png", dpi=110); plt.close(fig)

    dets = sorted(part["detector"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(dets) + 4, 4.5))
    xs, bottoms, labels = [], [], []
    cmap = plt.get_cmap("tab10")
    for i, det in enumerate(dets):
        for j, lab in enumerate(("neg", "pos")):
            x = i * 2.5 + j
            base = 0.0
            for m_i, m in enumerate(AA.MECHANISMS):
                v = part[(part["detector"] == det) & (part["label"] == lab) & (part["mechanism"] == m)]["frac_pooled"]
                v = float(v.iloc[0]) if len(v) else 0.0
                ax.bar(x, v, bottom=base, color=cmap(m_i), label=m if (i == 0 and j == 0) else None)
                base += v
            xs.append(x); labels.append(f"{det}\n{'FP' if lab == 'neg' else 'TP'}")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("fraction of candidates")
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.set_title(f"{tag}: mechanism partition (tau={knobs.tau})")
    fig.tight_layout(); fig.savefig(out_dir / f"fig_partition_{tag}.png", dpi=110); plt.close(fig)

    # exemplar gallery: top-scoring FP per mechanism (+ a TP for the two blob classes)
    picks = []
    for m in AA.MECHANISMS:
        fp = g[(g["label"] == "neg") & (g["mechanism"] == m)].sort_values("score_max", ascending=False)
        if len(fp):
            picks.append(("FP", m, fp.iloc[0]))
    for m in ("shadowing_blob", "bounded_blob", "narrow_column", "broad_column"):
        tp = g[(g["label"] == "pos") & (g["mechanism"] == m)].sort_values("score_max", ascending=False)
        if len(tp):
            picks.append(("TP", m, tp.iloc[0]))
    if picks:
        fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 9), squeeze=False)
        for j, (lab, m, r) in enumerate(picks):
            vol = load_vol(int(r.public_id)); D0, D1, D2 = vol.shape
            c = [r.cen_d0, r.cen_d1, r.cen_d2]; e = [r.ext_d0, r.ext_d1, r.ext_d2]
            z = int(round(c[2])); frame = np.asarray(vol[:, :, z]).T
            ax = axes[0, j]; ax.imshow(frame, cmap="gray", vmin=0, vmax=0.8)
            ax.add_patch(plt.Rectangle((c[0] - e[0] / 2, c[1] - e[1] / 2), e[0], e[1], fill=False, ec="lime" if lab == "TP" else "red"))
            ax.set_xlim(max(0, c[0] - 100), min(D0, c[0] + 100)); ax.set_ylim(D1, 0)
            ax.set_title(f"{lab} {m}\nvol {int(r.public_id)} s={r.score_max:.2f}", fontsize=8)
            x = int(round(c[0])); sag = np.asarray(vol[x, :, :])
            ax = axes[1, j]; ax.imshow(sag, cmap="gray", vmin=0, vmax=0.8)
            ax.add_patch(plt.Rectangle((c[2] - e[2] / 2, c[1] - e[1] / 2), e[2], e[1], fill=False, ec="lime" if lab == "TP" else "red"))
            ax.set_xlim(max(0, c[2] - 100), min(D2, c[2] + 100)); ax.set_ylim(D1, 0); ax.set_title("sagittal (sweep x depth)", fontsize=8)
            bp = AA.beam_profile(vol, c, e, knobs)
            ax = axes[2, j]; d = np.arange(D1) * knobs.iso_mm
            ax.plot(d, bp["col"], label="column"); ax.plot(d, bp["flank"], label="flanks"); ax.plot(d, bp["contrast"], label="contrast")
            ax.axvspan(bp["top"] * knobs.iso_mm, bp["bot"] * knobs.iso_mm, color="y", alpha=0.2); ax.axhline(-knobs.tau, ls="--", c="k", lw=0.6)
            ax.set_xlabel("depth (mm)"); ax.set_title(f"run {r.shadow_run_mm:.1f} mm, stripe {r.stripe_width_mm:.1f} mm", fontsize=8)
            if j == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"{tag}: exemplars per mechanism (top: axial frame, lateral x depth)")
        fig.tight_layout(); fig.savefig(out_dir / f"fig_exemplars_{tag}.png", dpi=100); plt.close(fig)

    pv = pd.DataFrame([r for r in arr["per_volume"] if "same_column" in r])
    if not pv.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, stat in zip(axes, ("same_column", "sweep_chain", "band_regularity")):
            obs = np.array([r["obs"] for r in pv[stat]], float); p95 = np.array([r["null_p95"] for r in pv[stat]], float)
            mean = np.array([r["null_mean"] for r in pv[stat]], float)
            idx = np.arange(len(obs))
            ax.scatter(idx, obs, s=14, c="tab:red", label="observed"); ax.plot(idx, mean, c="k", lw=0.8, label="null mean")
            ax.plot(idx, p95, c="k", ls="--", lw=0.8, label="null p95"); ax.set_title(stat); ax.set_xlabel("(detector, volume) sets")
            if stat == "same_column":
                ax.legend(fontsize=8)
        fig.suptitle(f"{tag}: FP arrangement statistics against permutation nulls")
        fig.tight_layout(); fig.savefig(out_dir / f"fig_arrangement_{tag}.png", dpi=110); plt.close(fig)

    if rank_rows:
        rr = pd.DataFrame(rank_rows)
        dets = list(rr["detector"].unique()); cfgs = list(rr["config"].unique())
        fig, ax = plt.subplots(figsize=(2 + 1.4 * len(dets), 4))
        for j, cfg in enumerate(cfgs):
            vals = [float(rr[(rr["detector"] == d) & (rr["config"] == cfg)]["cpm"].iloc[0]) for d in dets]
            ax.bar(np.arange(len(dets)) + (j - len(cfgs) / 2) * 0.15, vals, width=0.15, label=cfg)
        ax.set_xticks(range(len(dets))); ax.set_xticklabels(dets); ax.set_ylabel("CPM (official evaluator, OOF)")
        lo = max(0.0, rr["cpm"].min() - 0.05); ax.set_ylim(lo, min(1.0, rr["cpm"].max() + 0.05)); ax.legend(fontsize=8)
        ax.set_title(f"{tag}: what a per-candidate reader of each feature block reaches")
        fig.tight_layout(); fig.savefig(out_dir / f"fig_ranking_{tag}.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Artifact-signature audit (read-only)")
    add_phase3_paths(ap)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--detectors", default=None, help="comma list; default = every detector in the record")
    ap.add_argument("--tau-sweep", default="0.01,0.02,0.04")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--k-folds", type=int, default=0, help="grouped folds by volume; 0 = leave-one-volume-out (default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip-q3", action="store_true")
    args = ap.parse_args()

    profile_banner()
    croot = cache_root(args)
    t0 = time.time()
    pool = read_candidate_record(Path(args.out_root) / "candidates" / f"candidates_{args.split}")
    pool = pool[pool["split"] == args.split].copy()
    if args.detectors:
        keep = [d.strip() for d in args.detectors.split(",")]
        pool = pool[pool["detector_of_origin"].isin(keep)]
    dets = sorted(pool["detector_of_origin"].unique())
    knobs = AA.Knobs()
    out_dir = Path(args.out_root) / "artifact_audit"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.split

    print(f"\n# ARTIFACT-SIGNATURE AUDIT — split={args.split} detectors={dets} candidates={len(pool)} "
          f"volumes={pool['public_id'].nunique()} labels={pool['label'].value_counts().to_dict()}")
    print(f"# knobs: {dataclasses.asdict(knobs)}")

    _cache = {}

    def load_vol(vid: int):
        if vid not in _cache:
            _cache.clear(); _cache[vid] = K.open_vol(croot, vid)
        return _cache[vid]

    # ---- Q1 at the reported tau
    sig = AA.signature_table(load_vol, pool, knobs)
    # label-free context counts per (detector, volume) set over ALL candidates (pos/neg/ignore)
    ctx = np.zeros((len(sig), 4))
    pos_of = {idx: i for i, idx in enumerate(sig.index)}
    for _, g in sig.groupby(["detector_of_origin", "public_id"], sort=False):
        p = g[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float) * knobs.iso_mm
        cc = AA.context_counts(p, knobs)
        for row_idx, c in zip(g.index, cc):
            ctx[pos_of[row_idx]] = c
    for j, c in enumerate(AA.CONTEXT_FEATURES):
        sig[c] = ctx[:, j]
    sig.to_csv(out_dir / f"signatures_{tag}.csv", index=False)
    print(f"# signatures computed for {len(sig)} candidates in {time.time() - t0:.0f} s -> {out_dir / f'signatures_{tag}.csv'}")

    print("\n# Q1a — SIZE FIRST (the confound the old audit could not see)")
    size_rep = AA.contrast_report(sig, AA.SIZE_FEATURES)
    print_contrast(size_rep, "size")
    sig_rep = AA.contrast_report(sig, AA.SIGNATURE_FEATURES)
    print_contrast(sig_rep, f"Q1b — BEAM-LINE AND POSITION SIGNATURES (tau={knobs.tau})")
    for flag in ("dark_above", "reaches_floor"):
        g = sig[sig["label"].isin(["pos", "neg"])]
        fr = g.groupby("label")[flag].mean()
        print(f"  {flag:>18}: fraction TP {_fmt(fr.get('pos', np.nan), 6, 3)}  FP {_fmt(fr.get('neg', np.nan), 6, 3)}")
    part = AA.partition_report(sig)
    print_partition(part, f"Q1c — MECHANISM PARTITION (tau={knobs.tau}, run>={knobs.run_mm} mm, narrow<={knobs.stripe_narrow_mm} mm)")

    # ---- knob sweeps: re-assignment only (cheap) then tau (recompute)
    sweeps = {}
    print("\n# Q1d — KNOB SWEEPS (mean over detectors of the pooled fraction; FP | TP)")
    for run_mm in (4.0, 6.0, 10.0):
        for narrow in (3.0, 4.0, 6.0):
            kk = dataclasses.replace(knobs, run_mm=run_mm, stripe_narrow_mm=narrow)
            s2 = sig.copy(); s2["mechanism"] = [AA.assign_mechanism(r, kk) for r in s2.to_dict("records")]
            p2 = AA.partition_report(s2)
            fpm = p2[p2["label"] == "neg"].groupby("mechanism")["frac_pooled"].mean()
            tpm = p2[p2["label"] == "pos"].groupby("mechanism")["frac_pooled"].mean()
            key = f"run{run_mm:g}_narrow{narrow:g}"
            sweeps[key] = {"fp": fpm.to_dict(), "tp": tpm.to_dict()}
            print(f"  run>={run_mm:>4g} narrow<={narrow:g} | " + " ".join(
                f"{m[:6]} {fpm.get(m, 0):.2f}|{tpm.get(m, 0):.2f}" for m in AA.MECHANISMS))
    taus = [float(x) for x in args.tau_sweep.split(",")]
    for tau in taus:
        if abs(tau - knobs.tau) < 1e-12:
            continue
        kk = dataclasses.replace(knobs, tau=tau)
        s_t = AA.signature_table(load_vol, pool, kk)
        p_t = AA.partition_report(s_t)
        fpm = p_t[p_t["label"] == "neg"].groupby("mechanism")["frac_pooled"].mean()
        tpm = p_t[p_t["label"] == "pos"].groupby("mechanism")["frac_pooled"].mean()
        sweeps[f"tau{tau:g}"] = {"fp": fpm.to_dict(), "tp": tpm.to_dict(),
                                 "shadow_run_pervol_delta": float(np.median(AA.per_volume_delta(s_t, "shadow_run_mm")))}
        print(f"  tau={tau:<5g}            | " + " ".join(
            f"{m[:6]} {fpm.get(m, 0):.2f}|{tpm.get(m, 0):.2f}" for m in AA.MECHANISMS)
              + f"  shadow_run perVol delta {sweeps[f'tau{tau:g}']['shadow_run_pervol_delta']:+.3f}")

    # ---- Q2
    arr = arrangement_block(sig, knobs, args.n_perm, args.seed)
    print_arrangement(arr)

    # ---- Q3
    rank_rows = []
    if not args.skip_q3:
        gt = load_official_gt(args, args.split)
        rank_rows = ranking_block(sig, gt, per_detector=(args.split == "val"), k_folds=args.k_folds, seed=args.seed)
        print_ranking(rank_rows)

    if not args.no_figures:
        make_figures(sig, part, arr, rank_rows, load_vol, out_dir, tag, knobs)
        print(f"\n# figures -> {out_dir}/fig_*_{tag}.png")

    res = {"split": args.split, "detectors": dets, "n_candidates": int(len(pool)), "knobs": dataclasses.asdict(knobs),
           "size_report": size_rep, "signature_report": sig_rep, "partition": part, "sweeps": sweeps,
           "arrangement": arr, "ranking": rank_rows, "elapsed_s": time.time() - t0}
    (out_dir / f"artifact_audit_{tag}.json").write_text(json.dumps(_jsonable(res), indent=1))
    print(f"# json -> {out_dir / f'artifact_audit_{tag}.json'}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
