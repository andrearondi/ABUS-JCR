"""[SHADOW] Do candidate positions follow the volume's SHADOW structure? — 2D and 3D.

What this answers that [P3U2.PD] does not
----------------------------------------
The pool diagnostics describe each candidate on its own (score stats, its own extents, its
own anisotropy). The per-set projection figures show the candidate cloud lining up along
an axis, and the ABUS-CAD literature says that is expected: in an 846-patient study **86%
of CAD false positives were pseudo-lesions**, led by **marginal shadowing (39.1%)** and
**Cooper's-ligament shadowing (26.8%)**. If the FP cloud is organised by acoustic shadows,
that is a *joint*, *contextual* signal — the kind a set rescorer can use and a
per-candidate feature structurally cannot.

This probe measures the correspondence directly, in nine blocks:

  0. BEAM-AXIS AUDIT     — re-derives the beam axis from image content and checks it against
                           the declared convention. Runs first because every other block is
                           meaningless if the axes are mislabelled.
  1. SHADOW FIELD        — per-volume shadow fraction, tissue coverage, sanity numbers.
  2. AXIS MARGINALS      — side-by-side 1-D profiles: mean intensity, shadow mass, TP-centroid
                           density, FP-centroid density, per axis. The plain visual answer to
                           "do candidates sit where the shadows are?".
  3. PLANE CORRESPONDENCE— the same question per orthogonal plane, as 2-D maps plus their
                           rank correlation, with TP and FP kept separate.
  4. PER-CANDIDATE       — shadow descriptors per candidate and the TP-vs-FP Cliff's delta.
                           THE actionable block: these are the columns a rescorer would take.
  5. RAY STRUCTURE       — are candidates strung along shared beam lines (shadow rays)?
                           Permutation-referenced, plus a null-free k-NN direction anisotropy.
  6. SLICE CONCENTRATION — do candidates pile onto a few slices, and are those the shadowed
                           ones? (the striping visible in the per-set figures)
  7. PERIODICITY         — the 'multiple parallel dark stripes' question: is the banding
                           regular, and do candidates inherit the same period?
  8. ANISOTROPY RECHECK  — recomputes the deployed per-candidate anisotropy feature about the
                           MEASURED beam axis alongside the deployed one, so an axis mistake
                           cannot masquerade as 'no signal'.

Everything is measured in ISO-VOXEL INDEX units and fractions of the volume's own extent.
That is deliberate: it is the only frame that does not depend on ``SPACING_STORAGE_MM``,
whose per-axis assignment is exactly what block 0 exists to question. No millimetre figure
is printed anywhere in this script.

Usage (needs the frozen record + the Phase-1 iso cache; no GPU, no torch):
    python scripts/phase3_shadow_candidate_probe.py \
        --out-root   /home/maia-user/Andre2/outputs/phase3 \
        --phase1-out /home/maia-user/Andre2/outputs/phase1 \
        --split both
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
import matplotlib.patches as mpatches  # noqa: E402

from abus_jcr import conventions as C  # noqa: E402
from abus_jcr import cache as K  # noqa: E402
from abus_jcr.candidates.record import read_candidate_record  # noqa: E402
from abus_jcr.probe import shadow as SH  # noqa: E402

AX_LONG = {0: "d0", 1: "d1", 2: "d2"}
ROLE = {}          # filled once the beam axis is known


def _roles(beam_axis: int) -> dict:
    """Human names for the three storage axes given a measured beam axis.

    The two non-beam axes are the in-plane lateral axis and the elevational sweep axis.
    The sweep axis is ``conventions.SLICE_AXIS`` (the 2.5D stack axis); whatever is left
    is lateral. Naming them from the *measured* beam axis rather than from a constant is
    the whole point of this script.
    """
    sweep = C.SLICE_AXIS
    if beam_axis == sweep:                      # would break Inv. 1; reported loudly upstream
        others = [a for a in (0, 1, 2) if a != beam_axis]
        return {beam_axis: "depth/beam", others[0]: "in-plane A", others[1]: "in-plane B"}
    lateral = [a for a in (0, 1, 2) if a not in (beam_axis, sweep)][0]
    return {beam_axis: "depth/beam", lateral: "lateral", sweep: "sweep/elevational"}


def _plane_name(pair, beam_axis: int, roles: dict) -> str:
    """Anatomical name of the plane spanned by ``pair``, derived from the measured beam axis.

    The plane that EXCLUDES the beam axis is the ABUS coronal / C-plane. Of the two planes
    that contain it, the one also containing the lateral axis is the axial/transverse
    B-mode frame; the other is sagittal.
    """
    if beam_axis not in pair:
        return "CORONAL (C-plane)"
    other = [a for a in pair if a != beam_axis][0]
    return "AXIAL (B-mode frame)" if roles.get(other) == "lateral" else "SAGITTAL"


def _iso_boxes(g: pd.DataFrame) -> np.ndarray:
    """(n, 6) iso storage boxes from the record's frozen ``cen_d*`` / ``ext_d*`` columns."""
    cen = g[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float)
    ext = g[["ext_d0", "ext_d1", "ext_d2"]].to_numpy(float)
    return np.hstack([cen - ext / 2.0, cen + ext / 2.0])


# ============================================================================ block 0
def beam_axis_audit(croot: Path, pids, declared_beam: int, sub: int = 4) -> dict:
    """Re-derive the beam axis from image content on every available volume.

    Prints the per-axis attenuation signature and the vote. A disagreement with the
    declared convention is reported as a hard finding, not a warning: the augmentation
    policy, the anisotropy feature and every mm-valued constant hang off this axis.
    """
    votes, rows = [], []
    for pid in pids:
        try:
            vol = K.open_vol(croot, int(pid))
        except Exception as e:                                   # noqa: BLE001
            print(f"  (vol {pid} unavailable: {type(e).__name__}: {e})")
            continue
        st = SH.axis_attenuation_stats(np.asarray(vol), sub=sub)
        b = int(min((0, 1, 2), key=lambda a: st[a]["spearman"]))
        votes.append(b)
        rows.append({"public_id": int(pid), "beam_axis": b,
                     **{f"sp_d{a}": st[a]["spearman"] for a in (0, 1, 2)},
                     **{f"asym_d{a}": st[a]["asym"] for a in (0, 1, 2)},
                     **{f"peak_d{a}": st[a]["peak_pos_frac"] for a in (0, 1, 2)}})
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    winner = int(df["beam_axis"].mode().iloc[0])
    unanimity = float((df["beam_axis"] == winner).mean())

    print("\n# 0. BEAM-AXIS AUDIT — measured from image content, no spacing constant used\n")
    print("  The beam axis is the ONLY axis along which mean intensity decays monotonically")
    print("  from a bright near-field entrance (attenuation the TGC never fully cancels).\n")
    print(f"  {'axis':>6} {'role (declared)':>20} {'spearman med':>13} {'asym med':>10} "
          f"{'peak pos med':>13} {'votes':>7}")
    for a in (0, 1, 2):
        role = _roles(declared_beam).get(a, "?")
        print(f"  {AX_LONG[a]:>6} {role:>20} {df[f'sp_d{a}'].median():>13.3f} "
              f"{df[f'asym_d{a}'].median():>10.2f} {df[f'peak_d{a}'].median():>13.3f} "
              f"{int((df['beam_axis'] == a).sum()):>7}")
    print(f"\n  MEASURED beam axis = d{winner}  ({unanimity:.0%} of {len(df)} volumes agree)")
    print(f"  DECLARED beam axis = d{declared_beam}  "
          f"(conventions.FP_PROBE_ANISO_DEPTH_AXIS / IN_PLANE_ROW_AXIS)")
    if winner != declared_beam:
        print("\n  *** MISMATCH — the declared beam axis is NOT the axis the data attenuates along. ***")
        print("      Downstream this touches, at minimum: the augmentation flip policy (Inv. 13),")
        print("      the per-candidate anisotropy feature, and the per-axis SPACING_STORAGE_MM map.")
        print("      Escalate before trusting any depth-relative number in this report.")
    else:
        print("  -> convention CONFIRMED by the data.")
    return {"n": int(len(df)), "measured": winner, "declared": int(declared_beam),
            "unanimity": unanimity, "agree": bool(winner == declared_beam),
            "per_volume": df.to_dict(orient="records")}


# ============================================================================ per volume
def analyse_volume(croot: Path, pid: int, g: pd.DataFrame, beam_axis: int,
                   args) -> dict | None:
    """All per-volume shadow <-> candidate measurements for one (volume, candidate-set)."""
    try:
        vol = np.asarray(K.open_vol(croot, int(pid)))
    except Exception as e:                                       # noqa: BLE001
        print(f"  (vol {pid} unavailable: {type(e).__name__}: {e})")
        return None

    field = SH.shadow_field(vol, beam_axis=beam_axis, dark_z=args.dark_z,
                            tail_z=args.tail_z, far_frac=args.far_frac,
                            near_ok_z=args.near_ok_z, sub=args.sub)
    shape = vol.shape
    cen = g[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float)
    is_tp = (g["label"].to_numpy() == "pos")
    is_fp = (g["label"].to_numpy() == "neg")

    feats = SH.candidate_shadow_features(field, _iso_boxes(g), beam_axis=beam_axis,
                                         distal_depth=args.distal_depth)

    marg = SH.axis_marginals(field, vol, beam_axis, n_bins=args.n_bins)
    planes = SH.plane_shadow_maps(field, beam_axis)

    plane_corr = {}
    for pair, smap in planes.items():
        drop = [a for a in (0, 1, 2) if a not in pair][0]
        out_shape = (args.map_bins, args.map_bins)
        s_small = SH.downsample_map(smap, out_shape)
        for tag, mask in (("fp", is_fp), ("tp", is_tp), ("all", np.ones(len(g), bool))):
            dens = SH.centroid_density_map(cen[mask], drop_axis=drop, shape=shape,
                                           out_shape=out_shape, sigma_bins=args.smooth)
            plane_corr[f"{pair}|{tag}"] = SH.map_correlation(s_small, dens)

    shadow_axis_prof = {a: np.asarray(marg[a]["shadow"]) for a in (0, 1, 2)}
    conc = {}
    for a in (0, 1, 2):
        prof_full = np.asarray(field["shadow"]).mean(
            axis=tuple(x for x in (0, 1, 2) if x != a))
        conc[a] = {tag: SH.slice_concentration(cen[m], axis=a, n_slices=shape[a],
                                               shadow_profile=prof_full)
                   for tag, m in (("fp", is_fp), ("tp", is_tp))}
        for tag in conc[a]:
            conc[a][tag].pop("counts", None)

    ray = {tag: SH.ray_colinearity(cen[m], beam_axis=beam_axis,
                                   coronal_radius=args.coronal_radius,
                                   min_depth_sep=args.min_depth_sep, n_perm=args.n_perm)
           for tag, m in (("fp", is_fp), ("tp", is_tp))}
    aniso_dir = {tag: SH.knn_direction_anisotropy(cen[m], shape=shape, k=args.knn)
                 for tag, m in (("fp", is_fp), ("tp", is_tp))}

    # periodicity of the shadow field and of FP density along the two NON-beam axes
    period = {}
    for a in (0, 1, 2):
        if a == beam_axis:
            continue
        sp = np.asarray(field["shadow"]).mean(axis=tuple(x for x in (0, 1, 2) if x != a))
        cnt = np.bincount(np.clip(cen[is_fp][:, a].astype(int), 0, shape[a] - 1),
                          minlength=shape[a]).astype(float)
        period[a] = {"shadow": {k: v for k, v in SH.dominant_period(sp).items() if k != "autocorr"},
                     "fp": {k: v for k, v in SH.dominant_period(cnt).items() if k != "autocorr"}}

    return {"public_id": int(pid), "shape": list(shape),
            "shadow_frac": field["shadow_frac"], "n_tissue_lines": field["n_tissue_lines"],
            "frac_weak_lines": field["frac_weak_lines"],
            "n_cand": int(len(g)), "n_tp": int(is_tp.sum()), "n_fp": int(is_fp.sum()),
            "features": {k: v.tolist() for k, v in feats.items()},
            "labels": g["label"].tolist(),
            "marginals": {a: {"intensity": np.asarray(marg[a]["intensity"]).tolist(),
                              "shadow": np.asarray(marg[a]["shadow"]).tolist()}
                          for a in (0, 1, 2)},
            "centroids": cen.tolist(), "is_tp": is_tp.tolist(), "is_fp": is_fp.tolist(),
            "boxes": _iso_boxes(g).tolist(), "scores": g["score_max"].tolist(),
            "plane_corr": {k: v for k, v in plane_corr.items()},
            "slice_concentration": conc, "ray": ray, "knn_anisotropy": aniso_dir,
            "periodicity": period,
            "_field": field if args.keep_fields else None,
            "_vol_shape": shape}


# ============================================================================ reporting
def report(per_vol: list, beam_axis: int, roles: dict, tag: str) -> dict:
    print(f"\n{'='*78}\n# SHADOW <-> CANDIDATE CORRESPONDENCE — {tag}  "
          f"(volumes={len(per_vol)})\n")

    # --- 1 shadow field ---
    sf = np.array([v["shadow_frac"] for v in per_vol], float)
    print("# 1. SHADOW FIELD\n")
    print(f"  shadowed fraction of tissue voxels: med={np.median(sf):.4f}  "
          f"p10={np.percentile(sf, 10):.4f}  p90={np.percentile(sf, 90):.4f}")
    print(f"  volumes with a non-degenerate field (0 < frac < 0.5): "
          f"{int(((sf > 0) & (sf < 0.5)).sum())}/{len(sf)}")
    wk = np.array([v.get("frac_weak_lines", np.nan) for v in per_vol], float)
    if np.isfinite(wk).any():
        print(f"  weakly-coupled beam lines excluded (dim already in the near field): "
              f"med={np.nanmedian(wk):.3f}  p90={np.nanpercentile(wk, 90):.3f}")
        print("    -> these are the FOV margin / poor-contact columns, NOT posterior shadows;")
        print("       a large value means much of the volume carries no usable acoustic signal.")

    # --- 4 per-candidate (printed early: it is the actionable block) ---
    feat_names = ["shadow_frac", "z_mean", "distal_z", "proximal_z", "line_shadow",
                  "tissue_frac", "weak_line_frac"]
    tp_all = {k: [] for k in feat_names}
    fp_all = {k: [] for k in feat_names}
    for v in per_vol:
        lab = np.asarray(v["labels"])
        for k in feat_names:
            arr = np.asarray(v["features"][k], float)
            tp_all[k].append(arr[lab == "pos"])
            fp_all[k].append(arr[lab == "neg"])
    tp_all = {k: np.concatenate(v) if v else np.zeros(0) for k, v in tp_all.items()}
    fp_all = {k: np.concatenate(v) if v else np.zeros(0) for k, v in fp_all.items()}

    print("\n# 4. PER-CANDIDATE SHADOW FEATURES (TP vs FP; the block a rescorer would consume)\n")
    print("  |delta| >= 0.15 is the same 'worth a feature block' bar the Axis-A test uses.")
    print("  POOLED delta counts every candidate once. It OVERSTATES its own precision: the")
    print("  split is single-lesion dominant, so a volume's ~20 TPs are ~20 overlapping tubes")
    print("  on ONE lesion, not 20 independent observations. One unusual lesion with many")
    print("  tubes can carry the pooled number on its own.")
    print("  PER-VOLUME delta is computed inside each volume and then aggregated, so every")
    print("  volume counts once. 'sign' = the fraction of volumes agreeing with the pooled")
    print("  direction. Trust a feature when the two deltas agree AND sign is near 1.0.\n")
    print(f"  {'feature':>14} {'TP_med':>9} {'FP_med':>9} {'pooled_d':>9} {'perVol_d':>9} "
          f"{'sign':>6} {'n_TP':>7} {'n_FP':>7} {'nVol':>5}")
    feat_rows = []
    for k in feat_names:
        d = SH.cliffs_delta(tp_all[k], fp_all[k])
        per_v = []
        for v in per_vol:
            lab = np.asarray(v["labels"])
            arr = np.asarray(v["features"][k], float)
            a, b = arr[lab == "pos"], arr[lab == "neg"]
            if len(a) and len(b):
                dv = SH.cliffs_delta(a, b)
                if np.isfinite(dv):
                    per_v.append(dv)
        d_v = float(np.median(per_v)) if per_v else float("nan")
        if per_v and np.isfinite(d) and d != 0:
            sign = float(np.mean([np.sign(x) == np.sign(d) for x in per_v]))
        else:
            sign = float("nan")
        print(f"  {k:>14} {np.nanmedian(tp_all[k]) if len(tp_all[k]) else float('nan'):>9.3f} "
              f"{np.nanmedian(fp_all[k]) if len(fp_all[k]) else float('nan'):>9.3f} "
              f"{d:>9.3f} {d_v:>9.3f} {sign:>6.2f} {len(tp_all[k]):>7} {len(fp_all[k]):>7} "
              f"{len(per_v):>5}")
        feat_rows.append({"feature": k, "cliffs_delta": d, "cliffs_delta_per_volume": d_v,
                          "sign_agreement": sign, "n_volumes": len(per_v),
                          "tp_median": float(np.nanmedian(tp_all[k])) if len(tp_all[k]) else None,
                          "fp_median": float(np.nanmedian(fp_all[k])) if len(fp_all[k]) else None})

    def _score(r):
        """Rank on the WEAKER of the two deltas — a feature only counts if it survives both."""
        a, b = abs(r["cliffs_delta"]), abs(r["cliffs_delta_per_volume"])
        if not np.isfinite(a):
            return -1.0
        return min(a, b) if np.isfinite(b) else a

    best = max(feat_rows, key=_score, default=None)
    if best and _score(best) >= 0:
        s = _score(best)
        consistent = np.isfinite(best["sign_agreement"]) and best["sign_agreement"] >= 0.75
        verdict = ("CARRIES SIGNAL" if s >= 0.15 and consistent else
                   "NOT CONSISTENT ACROSS VOLUMES" if s >= 0.15 else "NO USABLE SIGNAL")
        print(f"\n  -> strongest (on the weaker of the two deltas): {best['feature']} "
              f"pooled={abs(best['cliffs_delta']):.3f} perVol={abs(best['cliffs_delta_per_volume']):.3f} "
              f"sign={best['sign_agreement']:.2f}  => {verdict}")

    # --- 3 plane correspondence ---
    print("\n# 3. PLANE CORRESPONDENCE — does candidate density track shadow density?\n")
    print("  Spearman between the 2-D shadow map and the 2-D centroid-density map, per plane.")
    print("  The CORONAL plane is the one that excludes the beam axis: a shadow ray is a POINT")
    print("  there, so it is the plane where a real correspondence must show up most strongly.\n")
    print(f"  {'plane':>10} {'anatomical':>22} {'rho FP':>9} {'rho TP':>9} {'rho all':>9} {'n vol':>7}")
    plane_rows = []
    for pair in [(0, 1), (0, 2), (1, 2)]:
        row = {"plane": f"d{pair[0]}xd{pair[1]}", "name": _plane_name(pair, beam_axis, roles)}
        for tagk in ("fp", "tp", "all"):
            vals = [v["plane_corr"].get(f"{pair}|{tagk}", {}).get("spearman", np.nan)
                    for v in per_vol]
            vals = np.asarray(vals, float)
            row[f"rho_{tagk}"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
            row[f"n_{tagk}"] = int(np.isfinite(vals).sum())
        print(f"  {row['plane']:>10} {row['name']:>22} {row['rho_fp']:>9.3f} "
              f"{row['rho_tp']:>9.3f} {row['rho_all']:>9.3f} {row['n_fp']:>7}")
        plane_rows.append(row)

    # --- 5 ray structure ---
    print("\n# 5. RAY STRUCTURE — are candidates strung along shared beam lines?\n")
    print("  enrichment = observed / permutation-null pairs that are coronally close but")
    print("  far apart in depth. >1 means candidates stack along the beam.\n")
    print(f"  {'subset':>8} {'enrichment med':>15} {'p25':>8} {'p75':>8} {'n vol':>7}")
    ray_rows = {}
    for tagk in ("fp", "tp"):
        e = np.array([v["ray"][tagk]["enrichment"] for v in per_vol], float)
        e = e[np.isfinite(e)]
        if len(e):
            print(f"  {tagk:>8} {np.median(e):>15.3f} {np.percentile(e, 25):>8.3f} "
                  f"{np.percentile(e, 75):>8.3f} {len(e):>7}")
            ray_rows[tagk] = {"median": float(np.median(e)), "n": int(len(e))}

    print("\n  k-NN displacement anisotropy (fraction of local displacement variance per axis;")
    print("  isotropic = 0.333 each; the beam axis dominating means ray-like alignment):\n")
    print(f"  {'subset':>8} " + " ".join(f"{'d'+str(a)+' ('+roles.get(a,'?')[:9]+')':>20}" for a in (0, 1, 2)))
    aniso_rows = {}
    for tagk in ("fp", "tp"):
        vals = {a: np.array([v["knn_anisotropy"][tagk].get(a, np.nan) for v in per_vol], float)
                for a in (0, 1, 2)}
        meds = {a: float(np.nanmedian(vals[a])) if np.isfinite(vals[a]).any() else float("nan")
                for a in (0, 1, 2)}
        print(f"  {tagk:>8} " + " ".join(f"{meds[a]:>20.3f}" for a in (0, 1, 2)))
        aniso_rows[tagk] = meds

    # --- 6 slice concentration ---
    print("\n# 6. SLICE CONCENTRATION — do candidates pile onto a few slices, and are those\n"
          "     the shadowed ones? (gini 0 = uniform; rho = per-slice count vs shadow mass)\n")
    print(f"  {'axis':>6} {'role':>20} {'gini FP':>9} {'top10% FP':>11} {'rho FP':>9} {'gini TP':>9}")
    conc_rows = []
    for a in (0, 1, 2):
        gf = np.nanmedian([v["slice_concentration"][a]["fp"]["gini"] for v in per_vol])
        tf = np.nanmedian([v["slice_concentration"][a]["fp"]["top10pct_share"] for v in per_vol])
        rf = np.nanmedian([v["slice_concentration"][a]["fp"]["spearman_vs_shadow"] for v in per_vol])
        gt = np.nanmedian([v["slice_concentration"][a]["tp"]["gini"] for v in per_vol])
        print(f"  {AX_LONG[a]:>6} {roles.get(a,'?'):>20} {gf:>9.3f} {tf:>11.3f} {rf:>9.3f} {gt:>9.3f}")
        conc_rows.append({"axis": a, "role": roles.get(a), "gini_fp": float(gf),
                          "top10_fp": float(tf), "rho_fp": float(rf), "gini_tp": float(gt)})

    # --- 7 periodicity ---
    print("\n# 7. PERIODICITY — 'multiple parallel dark stripes': is the banding regular?\n")
    print(f"  {'axis':>6} {'role':>20} {'shadow period':>15} {'strength':>9} "
          f"{'FP period':>11} {'strength':>9}")
    per_rows = []
    for a in (0, 1, 2):
        vals = [v["periodicity"].get(a) for v in per_vol if v["periodicity"].get(a)]
        if not vals:
            continue
        sp = np.nanmedian([x["shadow"]["period"] for x in vals])
        ss = np.nanmedian([x["shadow"]["strength"] for x in vals])
        fp_ = np.nanmedian([x["fp"]["period"] for x in vals])
        fs = np.nanmedian([x["fp"]["strength"] for x in vals])
        print(f"  {AX_LONG[a]:>6} {roles.get(a,'?'):>20} {sp:>15.1f} {ss:>9.3f} "
              f"{fp_:>11.1f} {fs:>9.3f}")
        per_rows.append({"axis": a, "shadow_period": float(sp), "shadow_strength": float(ss),
                         "fp_period": float(fp_), "fp_strength": float(fs)})
    print("\n  A high strength at a consistent period = regular banding (ribs / Cooper families /")
    print("  reverberation). A matching FP period means the detector is firing ON the bands.")

    return {"shadow_field": {"frac_median": float(np.median(sf))},
            "per_candidate": feat_rows, "plane_corr": plane_rows, "ray": ray_rows,
            "knn_anisotropy": aniso_rows, "slice_concentration": conc_rows,
            "periodicity": per_rows}


# ============================================================================ anisotropy recheck
def anisotropy_recheck(df: pd.DataFrame, measured_beam: int, declared_beam: int) -> dict:
    """Recompute the per-candidate anisotropy feature about the MEASURED beam axis.

    ``probe.fp_structure`` and ``probe.pool_diag`` both define anisotropy as
    ``ext_d<declared> / mean(other two)`` with ``declared = FP_PROBE_ANISO_DEPTH_AXIS``.
    If that constant names the wrong axis, the feature measures elongation along a
    direction with no acoustic meaning — and a null result says nothing about whether
    depth-elongation separates TP from FP. This prints both so the two cannot be confused.
    """
    print("\n# 8. ANISOTROPY RECHECK — deployed axis vs measured beam axis\n")
    tp = df[df["label"] == "pos"]
    fp = df[df["label"] == "neg"]
    rows = []
    for name, ax in (("deployed (d%d)" % declared_beam, declared_beam),
                     ("measured (d%d)" % measured_beam, measured_beam)):
        others = [a for a in (0, 1, 2) if a != ax]

        def _agg(sub):
            num = sub[f"ext_d{ax}"].to_numpy(float)
            den = (sub[f"ext_d{others[0]}"].to_numpy(float)
                   + sub[f"ext_d{others[1]}"].to_numpy(float)) / 2.0
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

        a_tp, a_fp = _agg(tp), _agg(fp)
        d = SH.cliffs_delta(a_tp, a_fp)
        rows.append({"variant": name, "axis": ax, "tp_median": float(np.nanmedian(a_tp)),
                     "fp_median": float(np.nanmedian(a_fp)), "cliffs_delta": d})
    print(f"  {'variant':>16} {'TP_med':>9} {'FP_med':>9} {'cliffs_d':>9}")
    for r in rows:
        print(f"  {r['variant']:>16} {r['tp_median']:>9.3f} {r['fp_median']:>9.3f} "
              f"{r['cliffs_delta']:>9.3f}")
    if measured_beam != declared_beam:
        a, b = abs(rows[0]["cliffs_delta"]), abs(rows[1]["cliffs_delta"])
        print(f"\n  -> the deployed feature is computed about d{declared_beam}, which this run measured")
        print(f"     is NOT the beam axis. |delta| {a:.3f} (deployed) vs {b:.3f} (measured beam).")
        if b > a:
            print("     Recomputing about the real beam axis IMPROVES separation — the recorded")
            print("     'anisotropy carries no signal' result was at least partly an axis artifact.")
        else:
            print("     Recomputing about the real beam axis does NOT improve separation — the")
            print("     null result stands on its own merits.")
    return {"rows": rows}


# ============================================================================ figures
def figures(per_vol: list, beam_axis: int, roles: dict, out_dir: Path, tag: str) -> list:
    written = []

    # --- FIG 1: axis marginals, side by side (the user-facing histogram comparison) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex="col")
    for a in (0, 1, 2):
        inten = np.nanmean([v["marginals"][a]["intensity"] for v in per_vol], axis=0)
        shad = np.nanmean([v["marginals"][a]["shadow"] for v in per_vol], axis=0)
        x = np.linspace(0, 1, len(inten))
        ax = axes[0, a]
        ax.plot(x, inten / max(np.nanmax(inten), 1e-9), color="tab:grey", lw=2,
                label="mean intensity (norm.)")
        ax.plot(x, shad / max(np.nanmax(shad), 1e-9), color="tab:blue", lw=2,
                label="shadow mass (norm.)")
        ax.set_title(f"d{a} — {roles.get(a,'?')}" + ("   [BEAM]" if a == beam_axis else ""),
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylabel("normalised")

        ax2 = axes[1, a]
        tp_c, fp_c = [], []
        for v in per_vol:
            cen = np.asarray(v["centroids"], float)
            n = v["_vol_shape"][a]
            tp_c += list(cen[np.asarray(v["is_tp"], bool)][:, a] / max(n, 1))
            fp_c += list(cen[np.asarray(v["is_fp"], bool)][:, a] / max(n, 1))
        bins = np.linspace(0, 1, 41)
        if fp_c:
            ax2.hist(fp_c, bins=bins, color="tab:red", alpha=0.55, density=True, label=f"FP (n={len(fp_c)})")
        if tp_c:
            ax2.hist(tp_c, bins=bins, color="tab:green", alpha=0.55, density=True, label=f"TP (n={len(tp_c)})")
        ax2.plot(x, shad / max(np.nanmean(shad), 1e-9), color="tab:blue", lw=2, label="shadow mass")
        ax2.set_xlabel(f"normalised position along d{a}")
        ax2.set_ylabel("density")
        ax2.legend(fontsize=7)
    fig.suptitle(f"{tag} — intensity / shadow / candidate marginals per storage axis "
                 f"(beam axis measured = d{beam_axis})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / f"shadow_marginals_{tag}.png"
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    # --- FIG 2: per-candidate feature distributions TP vs FP ---
    feat_names = ["shadow_frac", "z_mean", "distal_z", "proximal_z", "line_shadow",
                  "tissue_frac", "weak_line_frac"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for i, k in enumerate(feat_names):
        ax = axes[i // 3, i % 3]
        tp, fp = [], []
        for v in per_vol:
            lab = np.asarray(v["labels"])
            arr = np.asarray(v["features"][k], float)
            tp += list(arr[lab == "pos"]); fp += list(arr[lab == "neg"])
        tp = np.asarray(tp, float); fp = np.asarray(fp, float)
        tp = tp[np.isfinite(tp)]; fp = fp[np.isfinite(fp)]
        if len(tp) and len(fp):
            lo = float(min(np.percentile(tp, 1), np.percentile(fp, 1)))
            hi = float(max(np.percentile(tp, 99), np.percentile(fp, 99)))
            bins = np.linspace(lo, hi, 45) if hi > lo else 20
            ax.hist(fp, bins=bins, color="tab:red", alpha=0.55, density=True, label="FP")
            ax.hist(tp, bins=bins, color="tab:green", alpha=0.55, density=True, label="TP")
            ax.set_title(f"{k}   |δ|={abs(SH.cliffs_delta(tp, fp)):.3f}", fontsize=10)
            ax.legend(fontsize=7)
    for j in range(len(feat_names), 9):
        axes[j // 3, j % 3].axis("off")
    fig.suptitle(f"{tag} — per-candidate shadow descriptors, TP vs FP", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out_dir / f"shadow_features_{tag}.png"
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)
    return written


def iso_mm_per_voxel(beam_axis: int, roles: dict) -> np.ndarray:
    """Physical size of one ISO-cache voxel along each storage axis, in millimetres.

    Needed because the cache is **not** isotropic. ``preprocess`` resampled axis ``a`` by
    ``SPACING_STORAGE_MM[a] / ISO_SPACING_MM`` using the *declared* spacing map, so the
    real size of a cache voxel is ``true_spacing[a] * ISO_SPACING_MM /
    SPACING_STORAGE_MM[a]``. With the deployed constants that is roughly
    (1.10, 0.15, 0.40) mm — a 7.5x in-plane distortion. Drawing any plane at
    ``aspect="equal"`` in voxels, or at ``aspect="auto"``, therefore shows anatomically
    false proportions.

    ``true_spacing`` is assigned by MEASURED ROLE, not by stored index: the official spec
    gives 0.073 mm along depth, 0.200 mm laterally and 0.475674 mm between sweep frames,
    and this run measured which storage axis plays which role. See results/AXIS_AUDIT.md.
    """
    by_role = {"depth/beam": 0.073, "lateral": 0.200, "sweep/elevational": 0.475674}
    mm = np.ones(3, dtype=float)
    for a in (0, 1, 2):
        true_sp = by_role.get(roles.get(a, ""), C.SPACING_STORAGE_MM[a])
        mm[a] = true_sp * C.ISO_SPACING_MM / C.SPACING_STORAGE_MM[a]
    return mm


def exemplar_figure(vrec: dict, field: dict, beam_axis: int, roles: dict,
                    out_dir: Path, tag: str, volume=None, mask=None,
                    top_k: int = 10) -> Path:
    """One volume, three planes: real B-mode anatomy with the shadow field overlaid.

    Deliberately built like the [P3U2.PD] top-k box figure rather than as a heat map,
    because a heat map of *projected* shadow mass turned out to be close to unreadable: it
    averages along the dropped axis, so it corresponds to no anatomical image, shows no
    tissue, and reduces candidates to bare centroids. Here each panel is an actual iso
    **slice** through the lesion, the shadow field for that same slice is laid over it in
    red, and candidates are drawn as **boxes** so their extent is visible.

    Panels are drawn at their true physical aspect (see :func:`iso_mm_per_voxel`), so the
    three planes have different shapes — as they must: this volume is about
    173 x 50 x 166 mm, and the depth axis is by far the shortest.
    """
    cen = np.asarray(vrec["centroids"], float)
    boxes = np.asarray(vrec["boxes"], float)
    scores = np.asarray(vrec["scores"], float)
    is_tp = np.asarray(vrec["is_tp"], bool)
    shadow = np.asarray(field["shadow"], bool)
    vol = None if volume is None else np.asarray(volume, dtype=np.float32)
    msk = None if mask is None else np.asarray(mask) > 0
    mm = iso_mm_per_voxel(beam_axis, roles)

    # centre the slices on the GT lesion when there is one, else on the candidate cloud
    if msk is not None and msk.any():
        w = np.where(msk)
        centre = [int(np.mean(w[a])) for a in (0, 1, 2)]
    elif len(cen):
        centre = [int(np.median(cen[:, a])) for a in (0, 1, 2)]
    else:
        centre = [s // 2 for s in shadow.shape]

    order = np.argsort(-scores)[:top_k]
    # panel widths in MILLIMETRES, so the three planes share one physical scale bar and
    # the true 173 x 50 x 166 mm proportions are visible at a glance
    planes = [(0, 1), (0, 2), (1, 2)]
    widths = [shadow.shape[p[0]] * mm[p[0]] for p in planes]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5),
                             gridspec_kw={"width_ratios": widths})
    for i, pair in enumerate(planes):
        drop = [a for a in (0, 1, 2) if a not in pair][0]
        c = int(np.clip(centre[drop], 0, shadow.shape[drop] - 1))
        sh2 = np.take(shadow, c, axis=drop)
        ax = axes[i]
        ext = [0, sh2.shape[0], sh2.shape[1], 0]
        aspect = mm[pair[1]] / mm[pair[0]]        # true anatomical proportions

        if vol is not None:
            bg = np.take(vol, c, axis=drop)
            ax.imshow(bg.T, origin="upper", aspect=aspect, cmap="gray", extent=ext,
                      vmin=float(np.percentile(bg, 1)), vmax=float(np.percentile(bg, 99)))
        # shadow for THIS slice, as a translucent red wash
        overlay = np.zeros(sh2.shape + (4,), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = np.where(sh2, 0.45, 0.0)
        ax.imshow(np.transpose(overlay, (1, 0, 2)), origin="upper", aspect=aspect, extent=ext)

        if msk is not None and msk.any():
            m2 = np.take(msk, c, axis=drop).astype(float)
            if m2.any():
                ax.contour(np.linspace(0, m2.shape[0], m2.shape[0]),
                           np.linspace(0, m2.shape[1], m2.shape[1]),
                           m2.T, levels=[0.5], colors="cyan", linewidths=1.6)

        for n in order:
            lo0, hi0 = boxes[n][pair[0]], boxes[n][pair[0] + 3]
            lo1, hi1 = boxes[n][pair[1]], boxes[n][pair[1] + 3]
            col = "yellow" if is_tp[n] else "red"
            ax.add_patch(mpatches.Rectangle((lo0, lo1), hi0 - lo0, hi1 - lo1, fill=False,
                                            edgecolor=col, lw=1.3, alpha=0.9))
        ax.set_xlim(0, sh2.shape[0]); ax.set_ylim(sh2.shape[1], 0)
        ax.set_title(f"{_plane_name(pair, beam_axis, roles)}   slice @ d{drop}={c}\n"
                     f"d{pair[0]} ({roles.get(pair[0],'?')}) x d{pair[1]} ({roles.get(pair[1],'?')})",
                     fontsize=10)
        ax.set_xlabel(f"d{pair[0]} (iso vox, {mm[pair[0]]:.2f} mm each)", fontsize=8)
        ax.set_ylabel(f"d{pair[1]} (iso vox, {mm[pair[1]]:.2f} mm each)", fontsize=8)

    fig.suptitle(f"{tag} vol {vrec['public_id']} — shadow field (red) on the B-mode slice; "
                 f"top-{top_k} candidates as boxes (TP=yellow, FP=red), GT mask=cyan\n"
                 f"panels drawn at TRUE physical aspect; iso voxel = "
                 f"({mm[0]:.2f}, {mm[1]:.2f}, {mm[2]:.2f}) mm — the cache is NOT isotropic",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = out_dir / f"shadow_exemplar_{tag}_vol{vrec['public_id']}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description="[SHADOW] shadow <-> candidate correspondence probe")
    ap.add_argument("--out-root", required=True, help="phase3 out-root (contains candidates/)")
    ap.add_argument("--phase1-out", required=True, help="Phase-1 out (cache/ = iso volumes)")
    ap.add_argument("--candidates-dir", default=None)
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--detector", default=None,
                    help="restrict to one detector_of_origin (default: the first found, so the "
                         "unit stays one rescorer SET per volume — Inv. 7)")
    ap.add_argument("--max-volumes", type=int, default=0, help="0 = all")
    ap.add_argument("--n-exemplars", type=int, default=3)
    ap.add_argument("--exemplar-top-k", type=int, default=10,
                    help="how many candidates (by score_max) to draw as boxes on the exemplar")
    # shadow-field knobs
    ap.add_argument("--dark-z", type=float, default=-0.6)
    ap.add_argument("--tail-z", type=float, default=-0.4)
    ap.add_argument("--far-frac", type=float, default=0.30,
                    help="fraction of each beam line, measured from the deepest end, whose "
                         "mean residual is the PERSISTENCE test (a shadow darkens the far "
                         "field; an isolated hypoechoic mass does not)")
    ap.add_argument("--near-ok-z", type=float, default=-0.35,
                    help="a beam line whose NEAR-FIELD residual is below this is weakly "
                         "coupled / outside the usable field, not shadowed, and is excluded "
                         "from shadow flagging (reported as frac_weak_lines)")
    ap.add_argument("--distal-depth", type=int, default=24)
    ap.add_argument("--sub", type=int, default=4, help="stride for the per-depth baseline estimate")
    # analysis knobs
    ap.add_argument("--n-bins", type=int, default=60)
    ap.add_argument("--map-bins", type=int, default=24)
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--coronal-radius", type=float, default=12.0)
    ap.add_argument("--min-depth-sep", type=float, default=25.0)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--keep-fields", action="store_true",
                    help="retain shadow fields in memory for the exemplar figures")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / "shadow_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    croot = Path(args.phase1_out) / "cache"
    cdir = Path(args.candidates_dir) if args.candidates_dir else Path(args.out_root) / "candidates"

    declared_beam = int(C.FP_PROBE_ANISO_DEPTH_AXIS)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    summary = {}

    for split in splits:
        rec = cdir / f"candidates_{split}.parquet"
        if not rec.exists():
            print(f"\n(no record for split '{split}' at {rec} — skipped)")
            continue
        df = read_candidate_record(rec)
        if args.detector:
            df = df[df["detector_of_origin"] == args.detector]
        elif "detector_of_origin" in df.columns and df["detector_of_origin"].nunique() > 1:
            first = sorted(df["detector_of_origin"].unique())[0]
            print(f"\n[{split}] multiple detectors present; restricting to '{first}' so each "
                  f"volume contributes ONE rescorer set (override with --detector)")
            df = df[df["detector_of_origin"] == first]

        pids = sorted(df["public_id"].unique())
        if args.max_volumes:
            pids = pids[: args.max_volumes]
            df = df[df["public_id"].isin(pids)]

        audit = beam_axis_audit(croot, pids, declared_beam, sub=args.sub)
        beam = int(audit.get("measured", declared_beam))
        roles = _roles(beam)

        per_vol, fields = [], {}
        for pid in pids:
            g = df[df["public_id"] == pid]
            if not len(g):
                continue
            r = analyse_volume(croot, int(pid), g, beam, args)
            if r is None:
                continue
            f = r.pop("_field", None)
            if f is not None:                       # only when --keep-fields was passed
                fields[int(pid)] = f
            per_vol.append(r)
            print(f"  [{split}] vol {pid}: {r['n_cand']} cand "
                  f"({r['n_tp']} TP / {r['n_fp']} FP)  shadow_frac={r['shadow_frac']:.4f}")

        if not per_vol:
            print(f"[{split}] no analysable volumes — skipped")
            continue

        summ = report(per_vol, beam, roles, f"{split}")
        summ["beam_axis_audit"] = {k: v for k, v in audit.items() if k != "per_volume"}
        summ["anisotropy_recheck"] = anisotropy_recheck(df, beam, declared_beam)

        figs = figures(per_vol, beam, roles, out_dir, split)
        # exemplars: the most FP-heavy volumes
        order = sorted(per_vol, key=lambda v: -v["n_fp"])[: args.n_exemplars]
        for v in order:
            f = fields.get(v["public_id"])
            try:
                vol = np.asarray(K.open_vol(croot, v["public_id"]))
                if f is None:
                    f = SH.shadow_field(vol, beam_axis=beam, dark_z=args.dark_z,
                                        tail_z=args.tail_z, far_frac=args.far_frac,
                                        near_ok_z=args.near_ok_z, sub=args.sub)
            except Exception as e:                                # noqa: BLE001
                print(f"  (exemplar vol {v['public_id']} skipped: {type(e).__name__}: {e})")
                continue
            try:
                msk = np.asarray(K.open_mask(croot, v["public_id"]))
            except Exception:                                     # noqa: BLE001
                msk = None                                        # GT unavailable -> no contour
            figs.append(exemplar_figure(v, f, beam, roles, out_dir, split,
                                        volume=vol, mask=msk, top_k=args.exemplar_top_k))
        print(f"\n[{split}] figures: {[p.name for p in figs]}")

        drop_keys = {"features", "labels", "centroids", "is_tp", "is_fp", "marginals", "_vol_shape"}
        summ["per_volume"] = [{k: v for k, v in r.items() if k not in drop_keys} for r in per_vol]
        summary[split] = summ

    jp = out_dir / "shadow_probe.json"
    jp.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\njson = {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
