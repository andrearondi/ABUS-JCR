"""[P3U2.PD] Frozen candidate-pool deep diagnostics + visualization — informs Phase 4.

Reads ONLY the frozen record (``candidates_{train,val}.parquet``) — torch-free, no detector/cache. For
each split and each detector-of-origin pool it prints four analysis tables (feature discriminability,
ranking/rescoring headroom, pairwise geometry = the Axis-A test, set structure), writes a JSON summary,
and renders visualizations of the pool: aggregate stat PNGs, per-volume 2D-projection + 3D scatter PNGs,
and interactive plotly HTML (guarded) for a few auto-selected representative volumes — candidate centroids
coloured by score_max with TP/FP marker shapes, and the lesion location (best-IoU TP candidate) marked.

The top-k BOX figure overlays, per orthogonal plane: the **GT segmentation mask** (projected footprint
+ in-plane cross-section, cyan), the iso mask-hull box (lime), the OFFICIAL ``bbx_labels.csv`` box mapped
into iso space (white dashed — what ``iou_gt`` is scored against), and the top-k candidates
(TP=yellow, FP=red). Background defaults to the iso **slice through the GT centroid**: a max-intensity
projection (the pre-2026-07-27 default, still available via ``--bg mip``) washes the image out and
*inverts* a hypoechoic lesion, because the max along a ray is set by the brightest speckle on it.

The side table reports the recorded official ``IoU(GT)``, the ``IoU`` re-measured between the DRAWN iso
boxes, and ``box size / GT``. The last column is the answer to "the box clearly covers the lesion, why
is its IoU 0.07?": for a candidate that CONTAINS the GT, IoU is exactly ``1 / (box size / GT)``.

Usage (LOCAL-capable — needs no GPU, just the record):
    python scripts/phase3_pool_diagnostics.py \
        --out-root /home/maia-user/Andre2/outputs/phase3 --split both
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

from abus_jcr.candidates.record import read_candidate_record  # noqa: E402
from abus_jcr.probe import pool_diag as PD  # noqa: E402

G_LABELS = ["log|dx|/w", "log|dy|/h", "log|dz|/d", "log w_n/w_m", "log h_n/h_m", "log d_n/d_m"]


# ----------------------------------------------------------------------------- text tables
def _print_blocks(df: pd.DataFrame, tag: str, gt_by_pid: dict | None = None) -> dict:
    print(f"\n{'='*78}\n# POOL DIAGNOSTICS — {tag}  (n_cand={len(df)}, "
          f"n_vol={df['public_id'].nunique()}, pos={int((df.label=='pos').sum())}, "
          f"neg={int((df.label=='neg').sum())}, ignore={int((df.label=='ignore').sum())})\n")

    feat = PD.feature_discriminability(df)
    print("# 1. FEATURE DISCRIMINABILITY (TP vs FP; sorted by |Cliff's δ|)\n")
    print(f"  {'feature':>16} {'TP_med':>9} {'FP_med':>9} {'cliffs_δ':>9} {'balacc':>7} {'thresh':>9}")
    for _, r in feat.iterrows():
        print(f"  {r['feature']:>16} {r['tp_median']:>9.3f} {r['fp_median']:>9.3f} "
              f"{r['cliffs_delta']:>9.3f} {r['balacc']:>7.3f} {r['best_thresh']:>9.3f}")

    hr = PD.ranking_headroom(df)
    print("\n# 2. RANKING / RESCORING HEADROOM  (SET = one detector's candidates for one volume, Inv. 7)\n")
    print(f"  sets={hr['n_sets']}  TP-bearing sets={hr['n_pos_sets']}")
    print(f"  best-TP-rank: frac of TP-bearing sets where best TP is NOT already rank-1 = "
          f"{hr['frac_best_tp_not_rank1']:.3f}  (>0 ⇒ rescoring headroom)")
    print(f"  best-TP-rank histogram (1..10+): {hr['best_tp_rank_hist']}")
    print("  approx recall @ FP/vol (label-based; official CPM is B0'):")
    for b, r in sorted(hr["recall_at_fp"].items()):
        print(f"    {b:>6} FP/vol -> recall {r:.3f}")

    pg = PD.pairwise_geometry(df)
    print("\n# 3. PAIRWISE GEOMETRY g(m,n) — AXIS-A TEST\n")
    print(f"  pair counts: {pg['counts']}")
    print(f"  {'component':>14} {'TPTP_med':>9} {'TPFP_med':>9} {'FPFP_med':>9} "
          f"{'|δ|TPFP,FPFP':>13} {'|δ|TPTP,TPFP':>13}")
    for c in range(6):
        print(f"  {G_LABELS[c]:>14} {pg['median_per_component']['TP-TP'][c]:>9.3f} "
              f"{pg['median_per_component']['TP-FP'][c]:>9.3f} {pg['median_per_component']['FP-FP'][c]:>9.3f} "
              f"{pg['separability_per_component'][c]:>13.3f} {pg['separability_tptp_vs_tpfp'][c]:>13.3f}")
    max_sep = np.nanmax(pg["separability_per_component"]) if pg["separability_per_component"] else float("nan")
    max_sep2 = np.nanmax(pg["separability_tptp_vs_tpfp"]) if pg["separability_tptp_vs_tpfp"] else float("nan")
    print(f"  -> TP-FP vs FP-FP  max |δ| = {max_sep:.3f}  "
          f"({'SOME' if max_sep >= 0.15 else 'WEAK/NO'} per-pair FP-suppression signal)")
    print(f"  -> TP-TP vs TP-FP  max |δ| = {max_sep2:.3f}  "
          f"({'STRONG' if max_sep2 >= 0.4 else 'SOME' if max_sep2 >= 0.15 else 'WEAK'} co-location/consensus "
          f"signal — a TP's geometry to co-located TP peers vs to FPs; captured by jointness, nudgeable by Axis-A)")

    ss = PD.set_structure(df)
    a = ss["aggregate"]
    print("\n# 4. SET STRUCTURE  (per (detector, volume) set)\n")
    print(f"  sets={a['n_volumes']}  cands/set med={a['cands_per_vol_median']:.1f}  "
          f"pos/set med={a['pos_per_vol_median']:.1f}  neg/set med={a['neg_per_vol_median']:.1f}  "
          f"redundancy med={a['redundancy_median']:.1f}  (total pos {a['total_pos']} / neg {a['total_neg']})")

    ci = PD.confidence_iou_stats(df)
    print("\n# 5. CONFIDENCE vs IoU  (does score_max track localization quality?)\n")
    print(f"  corr(score_max, iou_gt): Pearson={ci['pearson_score_iou']:.3f}  Spearman={ci['spearman_score_iou']:.3f}")
    print(f"  TOP-{ci['k']} by score per set: mean IoU={ci['topk_mean_iou']:.3f}  "
          f"frac TP={ci['topk_frac_tp']:.3f}  top-1 mean IoU={ci['top1_mean_iou']:.3f}")

    lq = PD.localization_quality(df, gt_by_pid or {})
    if lq:
        print("\n# 6. LOCALIZATION QUALITY — is the box the right SIZE, or just in the right PLACE?\n")
        print("  (a candidate CONTAINING the GT scores IoU = 1 / size_ratio: 2.4x too big per axis -> 0.07)")
        print(f"  {'subset':>18} {'n':>7} {'size/GT p10':>12} {'med':>8} {'p90':>8} "
              f"{'centre off mm':>14} {'IoU med':>8}")
        for nm, key in (("all candidates", "all"), ("best-IoU per set", "best_iou_per_set"),
                        ("top-10 by score", "top10_by_score")):
            s = lq.get(key, {})
            if not s.get("n"):
                continue
            print(f"  {nm:>18} {s['n']:>7} {s['size_ratio_p10']:>12.2f} {s['size_ratio_med']:>8.2f} "
                  f"{s['size_ratio_p90']:>8.2f} {s['centre_mm_med']:>14.1f} {s['iou_med']:>8.3f}")

    return {"feature_discriminability": feat.to_dict(orient="records"), "ranking_headroom": hr,
            "confidence_iou": {k: v for k, v in ci.items() if k != "scatter"}, "_ci_scatter": ci["scatter"],
            "pairwise_geometry": {"counts": pg["counts"], "median_per_component": pg["median_per_component"],
                                  "separability_per_component": pg["separability_per_component"],
                                  "separability_tptp_vs_tpfp": pg["separability_tptp_vs_tpfp"]},
            "set_structure": ss, "localization_quality": lq}


# ----------------------------------------------------------------------------- viz helpers
def _select_viz_volumes(df: pd.DataFrame, k: int) -> list:
    """Auto-pick representative (detector, volume) SETS: TP-rich, FP-heavy, lowest-recall (fewest/no TP).

    A set = one detector's candidates for one volume (Inv. 7) — the unit the rescorer sees; do NOT mix seeds.
    """
    keys = ["detector_of_origin", "public_id"] if "detector_of_origin" in df.columns else ["public_id"]
    g = df.groupby(keys)
    stat = pd.DataFrame({"n": g.size(), "tp": g["label"].apply(lambda x: int((x == "pos").sum()))})
    picks = []
    if len(stat):
        picks = [("TP-rich", stat["tp"].idxmax()), ("FP-heavy", (stat["n"] - stat["tp"]).idxmax()),
                 ("low-recall", stat["tp"].idxmin())]
    seen, out = set(), []
    for name, key in picks:
        det, pid = (key if isinstance(key, tuple) else ("ALL", key))
        if (det, pid) not in seen:
            seen.add((det, pid)); out.append((name, det, pid))
        if len(out) >= k:
            break
    return out


def _gt_proxy(gvol: pd.DataFrame):
    """Lesion-location proxy for the wireframe: the box of the highest-IoU TP candidate (None if no TP)."""
    pos = gvol[gvol["label"] == "pos"]
    if len(pos) == 0:
        return None
    r = pos.loc[pos["iou_gt"].idxmax()]
    return (float(r.coordX), float(r.coordY), float(r.coordZ),
            float(r.x_length), float(r.y_length), float(r.z_length))


def _box_edges(cx, cy, cz, lx, ly, lz):
    """12 edges of an axis-aligned box as list of ((x0,x1),(y0,y1),(z0,z1))."""
    xs = [cx - lx / 2, cx + lx / 2]; ys = [cy - ly / 2, cy + ly / 2]; zs = [cz - lz / 2, cz + lz / 2]
    corners = [(xs[i], ys[j], zs[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if sum(corners[a][d] != corners[b][d] for d in range(3)) == 1:
                edges.append(tuple(zip(corners[a], corners[b])))
    return edges


def _plot_aggregate(diag: dict, ci_scatter, out_dir: Path, tag: str):
    feat = pd.DataFrame(diag["feature_discriminability"])
    hr, pg = diag["ranking_headroom"], diag["pairwise_geometry"]
    fig, axg = plt.subplots(2, 3, figsize=(19, 9)); fig.suptitle(f"Pool diagnostics — {tag}")
    ax = axg  # 2x3; panels (0,0..0,2) and (1,0..1,2)
    # (a) feature discriminability
    f2 = feat.iloc[::-1]
    ax[0, 0].barh(f2["feature"], f2["cliffs_delta"], color="tab:purple")
    ax[0, 0].axvline(0, color="k", lw=0.6); ax[0, 0].set_title("1. Feature discriminability (Cliff's δ, TP vs FP)")
    ax[0, 0].set_xlabel("Cliff's δ  (+ = higher on TP)")
    # (b) recall @ fp
    bs = sorted(hr["recall_at_fp"]); ax[0, 1].plot(bs, [hr["recall_at_fp"][b] for b in bs], "o-")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_title("2. Approx recall @ FP/vol (baseline ranking)")
    ax[0, 1].set_xlabel("FP/vol"); ax[0, 1].set_ylabel("recall"); ax[0, 1].set_ylim(0, 1)
    # (c) best-TP-rank hist
    h = hr["best_tp_rank_hist"]; ax[1, 0].bar([int(k) for k in h], [h[k] for k in h], color="tab:green")
    ax[1, 0].set_title("2. Best-TP rank per volume (1 = already top)"); ax[1, 0].set_xlabel("rank (10 = ≥10)")
    # (d) pairwise geometry separability — two contrasts
    xs = np.arange(6); w = 0.4
    ax[1, 1].bar(xs - w / 2, pg["separability_per_component"], w, color="tab:orange",
                 label="TP-FP vs FP-FP (FP-suppression)")
    ax[1, 1].bar(xs + w / 2, pg["separability_tptp_vs_tpfp"], w, color="tab:blue",
                 label="TP-TP vs TP-FP (co-location)")
    ax[1, 1].axhline(0.15, ls="--", color="grey"); ax[1, 1].set_xticks(xs)
    ax[1, 1].set_xticklabels(G_LABELS, rotation=30, ha="right", fontsize=8)
    ax[1, 1].set_title("3. Axis-A: pairwise geometry |δ| per g-component")
    ax[1, 1].set_ylabel("|Cliff's δ|"); ax[1, 1].legend(fontsize=7)
    # (e) confidence vs IoU scatter
    if ci_scatter and len(ci_scatter.get("score_max", [])):
        sm = np.asarray(ci_scatter["score_max"]); io = np.asarray(ci_scatter["iou_gt"])
        tp = np.asarray(ci_scatter["is_tp"], dtype=bool)
        ax[0, 2].scatter(sm[~tp], io[~tp], s=5, c="tab:red", alpha=0.25, label="FP")
        ax[0, 2].scatter(sm[tp], io[tp], s=6, c="tab:green", alpha=0.4, label="TP")
        ax[0, 2].axhline(0.3, ls="--", color="grey")
        ax[0, 2].set_title("5. score_max vs IoU(GT)"); ax[0, 2].set_xlabel("score_max")
        ax[0, 2].set_ylabel("IoU with GT"); ax[0, 2].legend(fontsize=7)
    else:
        ax[0, 2].axis("off")
    ax[1, 2].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / f"pool_diag_{tag}_aggregate.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def _plot_volume(gvol: pd.DataFrame, gt, out_dir: Path, tag: str, want_html: bool):
    is_pos = gvol["label"].to_numpy() == "pos"
    x, y, z = gvol.coordX.to_numpy(), gvol.coordY.to_numpy(), gvol.coordZ.to_numpy()
    s = gvol.score_max.to_numpy()
    written = []
    # static: 3 orthogonal 2D projections + 3D
    fig = plt.figure(figsize=(15, 4.2)); fig.suptitle(f"{tag}  (TP=○  FP=·  colour=score_max)")
    # NOTE: this figure plots the record's OFFICIAL (ITK) coordinates, where coordX = storage d2
    # (sweep), coordY = d1 (lateral), coordZ = d0 (depth) — see conventions.PERM_STORAGE_TO_ITK.
    # Panels are named for the storage axes so they agree with the top-k box figure (AX_NAME).
    projs = [("coronal: d2 sweep (X) × d1 lateral (Y)", x, y),
             ("B-mode frame: d1 lateral (Y) × d0 depth (Z)", y, z),
             ("depth × sweep: d2 (X) × d0 (Z)", x, z)]
    for i, (name, u, v) in enumerate(projs, 1):
        ax = fig.add_subplot(1, 4, i)
        ax.scatter(u[~is_pos], v[~is_pos], c=s[~is_pos], cmap="viridis", marker=".", s=18, vmin=0, vmax=max(s.max(), 0.3))
        sc = ax.scatter(u[is_pos], v[is_pos], c=s[is_pos], cmap="viridis", marker="o", s=60,
                        edgecolors="red", linewidths=1.1, vmin=0, vmax=max(s.max(), 0.3))
        ax.set_title(name, fontsize=9)
    fig.colorbar(sc, ax=fig.axes, fraction=0.02, pad=0.01, label="score_max")
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    ax3.scatter(x[~is_pos], y[~is_pos], z[~is_pos], c=s[~is_pos], cmap="viridis", marker=".", s=14)
    ax3.scatter(x[is_pos], y[is_pos], z[is_pos], c=s[is_pos], cmap="viridis", marker="o", s=50, edgecolors="red")
    if gt is not None:
        for (ex, ey, ez) in _box_edges(*gt):
            ax3.plot(ex, ey, ez, color="red", lw=0.8, alpha=0.7)
    ax3.set_title("3D (official ITK coords)", fontsize=9)
    ax3.set_xlabel("X = d2 sweep", fontsize=7); ax3.set_ylabel("Y = d1 lateral", fontsize=7)
    ax3.set_zlabel("Z = d0 depth", fontsize=7)
    p = out_dir / f"pool_diag_{tag}.png"; fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    # interactive plotly HTML (guarded)
    if want_html:
        try:
            import plotly.graph_objects as go
        except Exception as e:
            print(f"  (plotly unavailable: {type(e).__name__}; skipping HTML for {tag})")
            return written
        hov = [f"score={sc_:.3f} iou={iou:.2f} rank={int(rk)} {lb}"
               for sc_, iou, rk, lb in zip(gvol.score_max, gvol.iou_gt, gvol["rank"], gvol.label)]
        fig2 = go.Figure()
        for mask, nm, sym in [(~is_pos, "FP", "circle"), (is_pos, "TP", "diamond")]:
            fig2.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask], mode="markers", name=nm,
                marker=dict(size=[4 if nm == "FP" else 7] * int(mask.sum()), color=s[mask],
                            colorscale="Viridis", cmin=0, cmax=max(s.max(), 0.3), symbol=sym,
                            colorbar=dict(title="score_max"), line=dict(width=1, color="red" if nm == "TP" else "grey")),
                text=[hov[i] for i in np.where(mask)[0]], hoverinfo="text"))
        if gt is not None:
            for (ex, ey, ez) in _box_edges(*gt):
                fig2.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                            line=dict(color="red", width=3), showlegend=False, hoverinfo="skip"))
        fig2.update_layout(title=f"{tag} — candidate pool (TP vs FP, colour=score_max)")
        ph = out_dir / f"pool_diag_{tag}.html"; fig2.write_html(str(ph), include_plotlyjs="cdn"); written.append(ph)
    return written


# ----------------------------------------------------------------------------- exact GT + volume loaders
def _load_official_gt_map(split: str, data_root, phase1_out):
    """{public_id: (cx,cy,cz,lx,ly,lz)} EXACT official GT boxes; {} + warn if the data is unavailable."""
    try:
        import argparse as _ap
        from _phase3_common import load_official_gt, gt_official_tuple
        ns = _ap.Namespace(data_root=str(data_root), phase1_out=str(phase1_out))
        gt = load_official_gt(ns, split).set_index("public_id")
        return {int(v): gt_official_tuple(gt, int(v)) for v in gt.index}
    except Exception as e:
        print(f"  (exact GT unavailable for {split}: {type(e).__name__}: {e}; falling back to TP-candidate proxy)")
        return {}


def _load_iso_case(phase1_out, pid: int, want_volume: bool, official_gt=None):
    """``(iso_gt_hull, iso_mask, iso_volume, iso_official_gt)`` for one case; Nones on failure.

    - ``iso_gt_hull``     — tight hull of the ISO GT MASK (``mask_to_box_storage``), the lime box.
    - ``iso_mask``        — the iso GT mask itself, so the viz can draw the **segmentation footprint**
                            the box is supposed to enclose (the user-facing correspondence check).
    - ``iso_volume``      — memmapped (NOT materialised: the background only needs a few slices).
    - ``iso_official_gt`` — the OFFICIAL ``bbx_labels.csv`` box mapped native -> iso, i.e. exactly what
                            ``iou_gt`` is scored against, drawn in the picture's own coordinates. Its
                            agreement with ``iso_gt_hull`` IS the per-case reconstruction ceiling ([3.2]).
    """
    try:
        from abus_jcr import cache as K
        from abus_jcr.geometry import (mask_to_box_storage, official_box_to_storage,
                                       native_storage_to_iso_storage)
        croot = Path(phase1_out) / "cache"
        mask = np.asarray(K.open_mask(croot, int(pid))) > 0
        iso_gt = mask_to_box_storage(mask.astype(np.uint8))          # (min_d0,min_d1,min_d2,max_d0,max_d1,max_d2)
        vol = K.open_vol(croot, int(pid)) if want_volume else None   # memmap; sliced lazily
        iso_off = None
        if official_gt is not None:
            meta = K.read_meta(croot, int(pid))
            iso_off = native_storage_to_iso_storage(official_box_to_storage(tuple(official_gt)), meta)
        return iso_gt, mask, vol, iso_off
    except Exception as e:
        print(f"  (iso GT/volume unavailable for vol {pid}: {type(e).__name__}: {e}; boxes drawn without volume)")
        return None, None, None, None


# ----------------------------------------------------------------------------- top-10 BOX viz (iso space)
# Storage axes (conventions): d0 = depth/beam (skin at the top), d1 = lateral (in-plane),
# d2 = elevational sweep (SLICE_AXIS). The panels are labelled with these STORAGE names, never
# the official ITK (x,y,z) — where x=d2, y=d1, z=d0. Mixing the two is what made the old
# "axial x-y / sagittal y-z / coronal x-z" titles disagree with the centroid figure's own axes.
AX_NAME = {0: "d0 depth", 1: "d1 lateral", 2: "d2 sweep"}
# (projection axis, human title). The two remaining axes, ASCENDING, are (vertical, horizontal)
# — which is exactly the axis order numpy leaves after reducing `proj_axis`.
PROJS = [(2, "B-mode frame  d1(lateral) x d0(depth)"),
         (1, "depth x sweep  d2 x d0"),
         (0, "coronal  d2(sweep) x d1(lateral)")]


def _plane_axes(proj_axis: int):
    """(vertical_axis, horizontal_axis) left after reducing ``proj_axis`` — ascending order."""
    rem = [a for a in (0, 1, 2) if a != proj_axis]
    return rem[0], rem[1]


def _iso_box(row):
    """iso storage box (min_d0,min_d1,min_d2,max_d0,max_d1,max_d2) from a candidate's iso centre+extent."""
    return PD.iso_box_of_candidate(row)


def _rect(ax, box, ha, va, color, lw=1.6, ls="-"):
    """Draw the (ha, va) face of an iso box (axis indices 0=d0,1=d1,2=d2) as an unfilled rectangle."""
    mn, mx = box[:3], box[3:]
    ax.add_patch(mpatches.Rectangle((mn[ha], mn[va]), mx[ha] - mn[ha], mx[va] - mn[va],
                                    fill=False, edgecolor=color, lw=lw, ls=ls))


def _background(volume, proj_axis: int, centre_idx: int, mode: str, slab: int):
    """2D background for one plane + a caption.

    ``slice`` (DEFAULT) — the single iso slice through ``centre_idx`` (the GT centroid). This is what
    Phase-1/2 overlays draw and the ONLY mode in which a **hypoechoic** lesion is visible.
    ``mip`` — max-intensity projection. NEVER shows a dark lesion: the max along the ray is set by the
    brightest speckle anywhere on it, so a hypoechoic mass reads BRIGHTER than its surroundings
    (measured on val vol 122: MIP mean 0.612 vs volume mean 0.269; lesion +14% ABOVE background,
    vs darker on the slice). Kept only for a whole-extent overview of where the candidates sit.
    ``minip`` — min-intensity projection; keeps hypoechoic structure but over a long ray every column
    saturates dark. Use with ``--bg-slab`` to restrict it to a slab around the lesion.
    """
    n = int(volume.shape[proj_axis])
    c = int(np.clip(centre_idx, 0, n - 1))
    if mode == "slice":
        return np.asarray(np.take(volume, c, axis=proj_axis), dtype=np.float32), \
            f"{mode} @ {AX_NAME[proj_axis]}={c}"
    lo, hi = 0, n
    if slab and int(slab) > 0:
        h = max(1, int(slab) // 2)
        lo, hi = max(0, c - h), min(n, c + h + 1)
    sl = [slice(None)] * 3
    sl[proj_axis] = slice(lo, hi)
    sub = np.asarray(volume[tuple(sl)], dtype=np.float32)
    img = sub.min(axis=proj_axis) if mode == "minip" else sub.max(axis=proj_axis)
    span = "full" if (lo, hi) == (0, n) else f"{lo}..{hi - 1}"
    return img, f"{mode} over {AX_NAME[proj_axis]} [{span}]"


def _box_edges_iso(box):
    """12 edges of an iso min/max box as ((x0,x1),(y0,y1),(z0,z1)) with x=d1, y=d0, z=d2."""
    mn, mx = box[:3], box[3:]
    xs, ys, zs = [mn[1], mx[1]], [mn[0], mx[0]], [mn[2], mx[2]]
    corners = [(xs[i], ys[j], zs[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if sum(corners[a][d] != corners[b][d] for d in range(3)) == 1:
                edges.append(tuple(zip(corners[a], corners[b])))
    return edges


def _plot_top10_boxes(gset: pd.DataFrame, iso_gt, iso_mask, volume, iso_off, out_dir: Path, tag: str,
                      want_html: bool, k: int = 10, bg: str = "slice", bg_slab: int = 0):
    """Top-k candidates as BOXES over the ultrasound, with the GT SEGMENTATION MASK overlaid.

    Drawn per orthogonal plane: the projected GT mask footprint (cyan contour) + its cross-section at
    the displayed slice (cyan fill), the iso mask-hull box (lime), the OFFICIAL GT box mapped into iso
    (white dashed — what ``iou_gt`` is actually scored against), and the top-k candidate boxes
    (TP=yellow, FP=red). The side table adds ``IoU(iso)`` (measured between the DRAWN boxes) and
    ``size x GT`` so the picture and the record are auditable against each other.
    """
    top = gset.sort_values("score_max", ascending=False, kind="stable").head(k).reset_index(drop=True)
    audit = PD.box_audit(top, iso_gt)
    # GT centre in iso voxels — the slice the background is cut at (falls back to the volume centre)
    gt_c = ([(iso_gt[i] + iso_gt[i + 3]) / 2.0 for i in range(3)] if iso_gt is not None
            else ([s / 2.0 for s in volume.shape] if volume is not None else [0, 0, 0]))

    from abus_jcr.geometry import iou_storage
    recon_iou = iou_storage(iso_gt, iso_off) if (iso_gt is not None and iso_off is not None) else float("nan")

    fig = plt.figure(figsize=(18, 9.5))
    sub = (f"GT mask=cyan · mask-hull box=lime · official GT box (iso)=white dashed"
           f"{'' if not np.isfinite(recon_iou) else f'  [hull vs official IoU={recon_iou:.3f}]'}"
           f" · TP=yellow · FP=red")
    fig.suptitle(f"{tag} — top-{k} candidates as BOXES\n{sub}", fontsize=11)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1])
    for i, (proj_axis, name) in enumerate(PROJS):
        va, ha = _plane_axes(proj_axis)
        ax = fig.add_subplot(gs[0, i])
        cap = ""
        if volume is not None:
            img, cap = _background(volume, proj_axis, int(round(gt_c[proj_axis])), bg, bg_slab)
            vmin, vmax = np.percentile(img, [1, 99])
            ax.imshow(img, origin="lower", extent=[0, img.shape[1], 0, img.shape[0]],
                      cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        if iso_mask is not None:
            foot = np.asarray(iso_mask).any(axis=proj_axis)            # projected GT segmentation
            ax.contour(foot.astype(float), levels=[0.5], colors="cyan", linewidths=1.4,
                       origin="lower", extent=[0, foot.shape[1], 0, foot.shape[0]])
            if bg == "slice":                                          # + the in-plane cross-section
                cs = np.asarray(np.take(iso_mask, int(np.clip(round(gt_c[proj_axis]), 0,
                                                              iso_mask.shape[proj_axis] - 1)),
                                        axis=proj_axis)).astype(float)
                ax.contourf(cs, levels=[0.5, 1.5], colors=["cyan"], alpha=0.30,
                            origin="lower", extent=[0, cs.shape[1], 0, cs.shape[0]])
        if iso_off is not None:
            _rect(ax, iso_off, ha, va, "white", lw=1.6, ls="--")
        if iso_gt is not None:
            _rect(ax, iso_gt, ha, va, "lime", lw=2.2)
        for _, r in top.iterrows():
            _rect(ax, _iso_box(r), ha, va, "yellow" if r.label == "pos" else "red")
        ax.set_title(f"{name}\n{cap}", fontsize=8)
        ax.set_xlabel(AX_NAME[ha], fontsize=8); ax.set_ylabel(AX_NAME[va], fontsize=8)
    # 3D box wireframes (no volume — too heavy)
    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    if iso_gt is not None:
        for ex, ey, ez in _box_edges_iso(iso_gt):
            ax3.plot(ex, ey, ez, color="lime", lw=2.0)
    for _, r in top.iterrows():
        for ex, ey, ez in _box_edges_iso(_iso_box(r)):
            ax3.plot(ex, ey, ez, color=("gold" if r.label == "pos" else "red"), lw=0.9, alpha=0.8)
    ax3.set_title("3D boxes", fontsize=9)
    ax3.set_xlabel(AX_NAME[1]); ax3.set_ylabel(AX_NAME[0]); ax3.set_zlabel(AX_NAME[2])
    # IoU side table — official (recorded) AND iso (measured on the drawn boxes) + the size ratio
    axt = fig.add_subplot(gs[1, 1:]); axt.axis("off")
    cells = []
    for i, r in top.iterrows():
        a = audit.iloc[i] if len(audit) > i else None
        cells.append([i + 1, f"{r.score_max:.3f}", f"{r.iou_gt:.3f}",
                      "—" if a is None else f"{a.iou_iso:.3f}",
                      "—" if a is None else f"{a.size_ratio:.1f}x",
                      "TP" if r.label == "pos" else "FP"])
    tbl = axt.table(cellText=cells,
                    colLabels=["rank", "score_max", "IoU(GT, official)", "IoU(GT, iso)",
                               "box size / GT", "class"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.35)
    for i, r in top.iterrows():                                       # colour the class cell
        tbl[(i + 1, 5)].set_facecolor("#fff2b3" if r.label == "pos" else "#ffcccc")
    axt.set_title("a candidate that CONTAINS the GT scores IoU = 1 / (box size / GT) — "
                  "over-coverage costs exactly as much as a miss", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / f"pool_diag_{tag}_top{k}boxes.png"; fig.savefig(p, dpi=115); plt.close(fig)
    written = [p]

    # stdout audit — the numeric form of the picture (and the viz<->record consistency check)
    if len(audit):
        worst = float(np.nanmax(audit["iou_resid"].to_numpy(float))) if audit["iou_resid"].notna().any() else float("nan")
        print(f"  [{tag}] GT hull-vs-official IoU={recon_iou:.3f} (recon ceiling) | "
              f"top-{k} size/GT med={np.nanmedian(audit['size_ratio']):.1f}x "
              f"(min {np.nanmin(audit['size_ratio']):.1f}x, max {np.nanmax(audit['size_ratio']):.1f}x) | "
              f"contains-GT {int(audit['contains_gt'].sum())}/{len(audit)} | "
              f"max |IoU_iso - IoU_official| = {worst:.3f}"
              f"{'  <-- VIZ/RECORD MISMATCH, investigate' if np.isfinite(worst) and worst > 0.05 else ''}")

    if want_html:
        try:
            import plotly.graph_objects as go
        except Exception:
            return written
        figh = go.Figure()
        def _add_box(box, color, nm, meta):
            for ex, ey, ez in _box_edges_iso(box):
                figh.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color=color, width=4),
                                            name=nm, legendgroup=nm, showlegend=False, hovertext=meta, hoverinfo="text"))
        if iso_gt is not None:
            _add_box(iso_gt, "lime", "GT", "GT box (iso mask hull)")
        if iso_off is not None:
            _add_box(iso_off, "white", "GT-official", "official GT box mapped to iso (what iou_gt scores)")
        for i, r in top.iterrows():
            a = audit.iloc[i] if len(audit) > i else None
            _add_box(_iso_box(r), "gold" if r.label == "pos" else "red",
                     "TP" if r.label == "pos" else "FP",
                     f"rank {i+1}  score={r.score_max:.3f}  IoU_off={r.iou_gt:.3f}"
                     + ("" if a is None else f"  IoU_iso={a.iou_iso:.3f}  size={a.size_ratio:.1f}xGT")
                     + f"  {r.label}")
        figh.update_layout(title=f"{tag} — top-{k} boxes (TP=gold FP=red GT hull=lime official=white)",
                           scene=dict(xaxis_title=AX_NAME[1], yaxis_title=AX_NAME[0],
                                      zaxis_title=AX_NAME[2]))
        ph = out_dir / f"pool_diag_{tag}_top{k}boxes.html"; figh.write_html(str(ph), include_plotlyjs="cdn")
        written.append(ph)
    return written


# ----------------------------------------------------------------------------- main
def _assert_cache_matches_record(df: pd.DataFrame, phase1_out) -> None:
    """Warn loudly if the iso cache the viz reads is not the one the record was built from.

    The record carries ``preprocess_hash``; the cache dir is named by the CURRENT config's hash. If
    they differ, the drawn volume/mask/GT come from a different resampling than the candidate boxes —
    every overlay would be silently misaligned. Never fatal (the tables need no cache), always visible.
    """
    try:
        from abus_jcr.preprocess import preprocess_hash
        rec = sorted(set(str(h) for h in df["preprocess_hash"].dropna().unique()))
        cur = preprocess_hash()
        if rec and (len(rec) > 1 or rec[0] != cur):
            print(f"  ** WARNING: record preprocess_hash {rec} != current {cur} — the iso cache under "
                  f"{phase1_out}/cache does NOT match the record; overlays may be misaligned. **")
    except Exception as e:
        print(f"  (preprocess_hash check skipped: {type(e).__name__}: {e})")


def _run_split(record_base: Path, out_dir: Path, split: str, n_viz: int, want_html: bool,
               gt_off: dict, phase1_out, want_volume: bool, bg: str = "slice", bg_slab: int = 0):
    if not (record_base.with_suffix(".parquet").exists() or record_base.with_suffix(".csv").exists()):
        print(f"[skip {split}] no record at {record_base}.*")
        return
    df = read_candidate_record(record_base)
    _assert_cache_matches_record(df, phase1_out)
    pools = ["ALL"] + sorted(df["detector_of_origin"].unique())
    out = {}
    for pool in pools:
        sub = df if pool == "ALL" else df[df["detector_of_origin"] == pool].reset_index(drop=True)
        out[pool] = _print_blocks(sub, f"{split}/{pool}", gt_off)
    # visuals on the ALL pool (per-pool would multiply files; ALL is the informative aggregate)
    ci_scatter = out["ALL"].pop("_ci_scatter", None)
    for p in out:
        out[p].pop("_ci_scatter", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg_png = _plot_aggregate(out["ALL"], ci_scatter, out_dir, split)
        viz = _select_viz_volumes(df, n_viz)
        vfiles = []
        for name, det, pid in viz:
            gvol = df[(df["detector_of_origin"] == det) & (df["public_id"] == pid)].reset_index(drop=True)
            tag = f"{split}_{name}_{det}_vol{pid}"
            official_gt = gt_off.get(int(pid)) or _gt_proxy(gvol)     # EXACT GT if loaded, else proxy
            vfiles += _plot_volume(gvol, official_gt, out_dir, tag, want_html)
            iso_gt, iso_mask, vol, iso_off = _load_iso_case(
                phase1_out, pid, want_volume, gt_off.get(int(pid)))    # exact iso GT + mask + volume
            vfiles += _plot_top10_boxes(gvol, iso_gt, iso_mask, vol, iso_off, out_dir, tag,
                                        want_html, bg=bg, bg_slab=bg_slab)
    (out_dir / f"pool_diag_{split}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[{split}] aggregate PNG = {agg_png}")
    print(f"[{split}] per-set files = {[str(p.name) for p in vfiles]}")
    print(f"[{split}] json = {out_dir / f'pool_diag_{split}.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="[P3U2.PD] frozen-pool deep diagnostics + visualization")
    ap.add_argument("--out-root", required=True, help="phase3 out-root (contains candidates/)")
    ap.add_argument("--candidates-dir", default=None, help="override the candidates/ dir")
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--n-viz-volumes", type=int, default=3)
    ap.add_argument("--no-html", action="store_true", help="skip interactive plotly HTML")
    ap.add_argument("--no-volume", action="store_true", help="skip the ultrasound background entirely")
    ap.add_argument("--bg", default="slice", choices=["slice", "minip", "mip"],
                    help="box-viz background. 'slice' (DEFAULT) = the iso slice through the GT centroid "
                         "— the only mode where a HYPOECHOIC lesion is visible. 'mip' = the old "
                         "max-intensity projection: washes the image out and inverts a dark lesion "
                         "(it reads brighter than its surroundings). 'minip' = min-projection.")
    ap.add_argument("--bg-slab", type=int, default=0,
                    help="for --bg mip/minip: project only this many slices centred on the GT "
                         "(0 = the whole axis, the old behaviour)")
    ap.add_argument("--data-root", default="/home/maia-user/Andre2/data",
                    help="dataset root holding the split dirs (for EXACT official GT boxes)")
    ap.add_argument("--phase1-out", default="/home/maia-user/Andre2/outputs/phase1",
                    help="Phase-1 out (cache/ = iso volume + GT mask, for the volume background)")
    args = ap.parse_args()

    cand_dir = Path(args.candidates_dir) if args.candidates_dir else Path(args.out_root) / "candidates"
    out_dir = Path(args.out_root) / "pool_diag"; out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for sp in splits:
        gt_off = _load_official_gt_map(sp, args.data_root, args.phase1_out)
        _run_split(cand_dir / f"candidates_{sp}", out_dir, sp, args.n_viz_volumes, not args.no_html,
                   gt_off, args.phase1_out, not args.no_volume, bg=args.bg, bg_slab=args.bg_slab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
