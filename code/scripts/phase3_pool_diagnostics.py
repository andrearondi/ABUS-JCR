"""[P3U2.PD] Frozen candidate-pool deep diagnostics + visualization — informs Phase 4.

Reads ONLY the frozen record (``candidates_{train,val}.parquet``) — torch-free, no detector/cache. For
each split and each detector-of-origin pool it prints four analysis tables (feature discriminability,
ranking/rescoring headroom, pairwise geometry = the Axis-A test, set structure), writes a JSON summary,
and renders visualizations of the pool: aggregate stat PNGs, per-volume 2D-projection + 3D scatter PNGs,
and interactive plotly HTML (guarded) for a few auto-selected representative volumes — candidate centroids
coloured by score_max with TP/FP marker shapes, and the lesion location (best-IoU TP candidate) marked.

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

from abus_jcr.candidates.record import read_candidate_record  # noqa: E402
from abus_jcr.probe import pool_diag as PD  # noqa: E402

G_LABELS = ["log|dx|/w", "log|dy|/h", "log|dz|/d", "log w_n/w_m", "log h_n/h_m", "log d_n/d_m"]


# ----------------------------------------------------------------------------- text tables
def _print_blocks(df: pd.DataFrame, tag: str) -> dict:
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
    print("\n# 3. PAIRWISE GEOMETRY g(m,n) — AXIS-A TEST (TP-FP vs FP-FP separability per component)\n")
    print(f"  pair counts: {pg['counts']}")
    print(f"  {'component':>14} {'TPTP_med':>9} {'TPFP_med':>9} {'FPFP_med':>9} {'|δ|(TPFP,FPFP)':>15}")
    for c in range(6):
        print(f"  {G_LABELS[c]:>14} {pg['median_per_component']['TP-TP'][c]:>9.3f} "
              f"{pg['median_per_component']['TP-FP'][c]:>9.3f} {pg['median_per_component']['FP-FP'][c]:>9.3f} "
              f"{pg['separability_per_component'][c]:>15.3f}")
    max_sep = np.nanmax(pg["separability_per_component"]) if pg["separability_per_component"] else float("nan")
    print(f"  -> max |δ| across components = {max_sep:.3f} "
          f"({'SOME pairwise geometric signal' if max_sep >= 0.15 else 'WEAK/NO pairwise geometric signal'})")

    ss = PD.set_structure(df)
    a = ss["aggregate"]
    print("\n# 4. SET STRUCTURE  (per (detector, volume) set)\n")
    print(f"  sets={a['n_volumes']}  cands/set med={a['cands_per_vol_median']:.1f}  "
          f"pos/set med={a['pos_per_vol_median']:.1f}  neg/set med={a['neg_per_vol_median']:.1f}  "
          f"redundancy med={a['redundancy_median']:.1f}  (total pos {a['total_pos']} / neg {a['total_neg']})")

    return {"feature_discriminability": feat.to_dict(orient="records"), "ranking_headroom": hr,
            "pairwise_geometry": {"counts": pg["counts"], "median_per_component": pg["median_per_component"],
                                  "separability_per_component": pg["separability_per_component"]},
            "set_structure": ss}


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


def _plot_aggregate(diag: dict, out_dir: Path, tag: str):
    feat = pd.DataFrame(diag["feature_discriminability"])
    hr, pg = diag["ranking_headroom"], diag["pairwise_geometry"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); fig.suptitle(f"Pool diagnostics — {tag}")
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
    # (d) pairwise geometry separability
    ax[1, 1].bar(range(6), pg["separability_per_component"], color="tab:orange")
    ax[1, 1].axhline(0.15, ls="--", color="grey"); ax[1, 1].set_xticks(range(6))
    ax[1, 1].set_xticklabels(G_LABELS, rotation=30, ha="right", fontsize=8)
    ax[1, 1].set_title("3. Axis-A: |δ| TP-FP vs FP-FP per g-component"); ax[1, 1].set_ylabel("|Cliff's δ|")
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
    projs = [("axial X–Y", x, y), ("sagittal Y–Z", y, z), ("coronal X–Z", x, z)]
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
    ax3.set_title("3D", fontsize=9)
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


# ----------------------------------------------------------------------------- main
def _run_split(record_base: Path, out_dir: Path, split: str, n_viz: int, want_html: bool):
    if not (record_base.with_suffix(".parquet").exists() or record_base.with_suffix(".csv").exists()):
        print(f"[skip {split}] no record at {record_base}.*")
        return
    df = read_candidate_record(record_base)
    pools = ["ALL"] + sorted(df["detector_of_origin"].unique())
    out = {}
    for pool in pools:
        sub = df if pool == "ALL" else df[df["detector_of_origin"] == pool].reset_index(drop=True)
        out[pool] = _print_blocks(sub, f"{split}/{pool}")
    # visuals on the ALL pool (per-pool would multiply files; ALL is the informative aggregate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg_png = _plot_aggregate(out["ALL"], out_dir, split)
        viz = _select_viz_volumes(df, n_viz)
        vfiles = []
        for name, det, pid in viz:
            gvol = df[(df["detector_of_origin"] == det) & (df["public_id"] == pid)].reset_index(drop=True)
            vfiles += _plot_volume(gvol, _gt_proxy(gvol), out_dir, f"{split}_{name}_{det}_vol{pid}", want_html)
    (out_dir / f"pool_diag_{split}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[{split}] aggregate PNG = {agg_png}")
    print(f"[{split}] per-volume files = {[str(p.name) for p in vfiles]}")
    print(f"[{split}] json = {out_dir / f'pool_diag_{split}.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="[P3U2.PD] frozen-pool deep diagnostics + visualization")
    ap.add_argument("--out-root", required=True, help="phase3 out-root (contains candidates/)")
    ap.add_argument("--candidates-dir", default=None, help="override the candidates/ dir")
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--n-viz-volumes", type=int, default=3)
    ap.add_argument("--no-html", action="store_true", help="skip interactive plotly HTML")
    ap.add_argument("--data-root", default=None, help="unused (kept for runbook symmetry)")
    args = ap.parse_args()

    cand_dir = Path(args.candidates_dir) if args.candidates_dir else Path(args.out_root) / "candidates"
    out_dir = Path(args.out_root) / "pool_diag"; out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for sp in splits:
        _run_split(cand_dir / f"candidates_{sp}", out_dir, sp, args.n_viz_volumes, not args.no_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
