"""Intensity <-> candidate-localisation probe. READ-ONLY: writes only figures + one JSON.

Two stages, both descriptive. There is no verdict, no pass/fail gate and no
auto-generated conclusion: the script prints numbers and draws figures, and the reading
is written by hand afterwards. That is deliberate — a probe that grades itself invites
the grader, not the data, to decide the answer.

**Stage 1 — which storage axis is the beam/depth axis.** Measured from image content
only. No constant from ``conventions`` is an INPUT; the declared values are printed
alongside purely so any disagreement is visible.

**Stage 2 — does candidate placement track dark regions.** Per plane (2-D) and per
candidate box (3-D), for one detector at a time so a set stays (detector, volume).

Everything runs in **native** NRRD space. The Phase-1 iso cache is never opened, so no
resampling assumption is inherited.

Usage (server):
    python scripts/phase3_intensity_candidate_probe.py \
        --out-root  $WORK/outputs/phase3 \
        --data-root $WORK/data --split val

Usage (laptop, Validation only):
    python scripts/phase3_intensity_candidate_probe.py \
        --data-root "/Users/andrearondi/Desktop/KTH/Tesi/Dataset" --split val \
        --candidates ~/Desktop/KTH/Tesi/Dataset/pool/candidates_val.parquet \
        --out-root /tmp/intensity_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.io_nrrd import discover_cases, load_array
from abus_jcr.probe import intensity_geom as ig
from _phase3_common import add_phase3_paths, split_root, profile_banner

# The three spacings the official challenge description quotes, as an unordered SET.
# Which one belongs to which axis is exactly what Stage 1 determines; nothing here
# assumes an assignment.
OFFICIAL_SPACINGS_MM = (0.073, 0.200, 0.475674)

# GE Invenia ABUS acquisition envelope, from the GoldSeal Invenia ABUS 2.0 service
# documentation (768 elements, 0.20 mm element pitch -> 15.3 cm aperture; 16.9 cm
# transducer travel; 2.5-6.0 cm depth setting). Used ONLY as a plausibility window on
# the physical extent of each axis — the window is deliberately generous, because a
# wrong assignment misses it by a factor, not by a few percent.
SCANNER_ENVELOPE_MM = {"depth": (20.0, 70.0), "lateral": (120.0, 200.0), "sweep": (140.0, 190.0)}

# What each spacing VALUE physically is, from the same GE documentation:
#   0.200 mm    = the transducer's element pitch  -> LATERAL, across the array face
#   0.475674 mm = 169 mm travel / 355 frames      -> SWEEP, between B-mode frames
#   0.073 mm    = the remaining one               -> AXIAL/DEPTH, along the beam
# This says which spacing means what. It does NOT say which ARRAY axis is which —
# that is what the measurement and the extent enumeration below decide.
SPACING_ROLE = {0.073: "depth", 0.2: "lateral", 0.475674: "sweep"}

AX = ("d0", "d1", "d2")
# Grid cell for every 2-D correspondence map. One knob, stated once, and the figures
# are drawn on the SAME grid the correlations are computed on, so the number and the
# picture can be checked against each other.
CELL_MM = 5.0


# ==========================================================================
# Stage 1
# ==========================================================================

def stage1(cases, out_dir: Path, max_volumes=None, n_exemplars: int = 3):
    ids = sorted(cases)[: max_volumes or len(cases)]
    rows, profiles = [], {a: [] for a in range(3)}
    exemplars = {}

    for vid in ids:
        vol, _ = load_array(cases[vid].data)
        rec = {"public_id": vid, "shape": tuple(int(s) for s in vol.shape)}
        for a in range(3):
            p = ig.axis_profile(vol, a)
            st = ig.profile_stats(p)
            rec[f"spearman_{a}"] = st["spearman"]
            rec[f"peak_pos_{a}"] = st["peak_pos"]
            rec[f"decile_ratio_{a}"] = st["decile_ratio"]
            rec[f"adjcorr_{a}"] = ig.adjacent_correlation(vol, a)
            profiles[a].append(p / max(p.max(), 1e-9))
        rec["beam_vote"] = int(np.argmin([rec[f"spearman_{a}"] for a in range(3)]))
        rows.append(rec)
        if len(exemplars) < n_exemplars:
            mid = [s // 2 for s in vol.shape]
            # keyed by the SORTED pair of axes the plane spans
            exemplars[vid] = {(0, 1): np.asarray(vol[:, :, mid[2]]),
                              (0, 2): np.asarray(vol[:, mid[1], :]),
                              (1, 2): np.asarray(vol[mid[0], :, :])}
        del vol

    df = pd.DataFrame(rows)
    votes = df["beam_vote"].value_counts().to_dict()
    beam = int(df["beam_vote"].mode().iloc[0])
    med_shape = np.median(np.array([r["shape"] for r in rows], dtype=float), axis=0)

    # The remaining two spacings are fixed by PHYSICAL EXTENT, not by another heuristic:
    # enumerate all 6 assignments of the three official spacings to the three axes and
    # keep the ones whose extents land inside the scanner envelope. Exactly one survives
    # (the wrong ones miss by a factor of 2-6, not by a few percent), so no tie-break
    # rule is needed and the whole enumeration is printed for inspection.
    from itertools import permutations as _perms
    feasible = []
    for perm in _perms(OFFICIAL_SPACINGS_MM):
        ext = [med_shape[a] * perm[a] for a in range(3)]
        roles = [SPACING_ROLE[round(s, 6)] for s in perm]
        ok = all(SCANNER_ENVELOPE_MM[roles[a]][0] <= ext[a] <= SCANNER_ENVELOPE_MM[roles[a]][1]
                 for a in range(3))
        feasible.append({"spacing": list(perm), "extent_mm": ext, "roles": roles,
                         "in_envelope": bool(ok),
                         "beam_agrees": bool(roles[beam] == "depth")})
    # Neither criterion resolves it alone: the envelope leaves 2 permutations standing
    # (both put a plausible breast in the box), and the attenuation vote fixes only which
    # axis is depth. Their INTERSECTION is what identifies the map — two independent
    # lines of evidence, one of which is a measurement on this very data.
    by_envelope = [f for f in feasible if f["in_envelope"]]
    surviving = [f for f in by_envelope if f["beam_agrees"]]
    chosen = surviving[0] if len(surviving) == 1 else None
    spacing = list(chosen["spacing"]) if chosen else list(C.SPACING_STORAGE_MM)
    # Sweep = the coarsest of the axes that are NOT the measured beam axis. Taking a plain
    # argmax would collapse sweep onto beam in the unresolved fallback (where `spacing` is
    # the declared map and need not agree with the measurement), leaving one axis with no
    # role at all. When the enumeration DID resolve, this is identical to the argmax.
    sweep = max((a for a in range(3) if a != beam), key=lambda a: spacing[a])
    lateral = [a for a in range(3) if a not in (beam, sweep)][0]

    print("\n" + "=" * 78)
    print("# STAGE 1 — WHICH AXIS IS THE BEAM/DEPTH AXIS?  (measured from image content)\n")
    print("  The beam axis is the one along which mean intensity decays monotonically from")
    print("  a bright near-field entrance. Spacing-free: it uses array indices only.\n")
    print(f"  {'axis':>5} {'spearman med':>13} {'peak pos med':>13} {'decile ratio':>13} "
          f"{'adj corr med':>13} {'votes':>6}")
    for a in range(3):
        print(f"  {AX[a]:>5} {df[f'spearman_{a}'].median():>13.3f} {df[f'peak_pos_{a}'].median():>13.3f} "
              f"{df[f'decile_ratio_{a}'].median():>13.2f} {df[f'adjcorr_{a}'].median():>13.3f} "
              f"{votes.get(a, 0):>6}")
    print(f"\n  MEASURED beam axis = {AX[beam]}  ({votes.get(beam, 0)}/{len(df)} volumes agree)")
    print("\n  NOTE on 'adj corr': adjacent-plane correlation does NOT order the axes by")
    print("  sampling pitch here — the finest axis scores LOWEST, because axial speckle")
    print("  decorrelates within a pulse length and the attenuation gradient adds a")
    print("  plane-to-plane trend. It is printed as a negative control, not as evidence.")

    print("\n# 1b. PER-AXIS SAMPLE COUNTS (a hardware-fixed axis takes ONE value across the")
    print("      split; a per-patient setting takes a few; an acquisition length varies freely)\n")
    for a in range(3):
        vals = sorted({r["shape"][a] for r in rows})
        show = vals if len(vals) <= 8 else [vals[0], "...", vals[-1]]
        print(f"    {AX[a]}: n_unique={len(vals):>2}  values={show}")

    print("\n# 1c. SPACING ASSIGNMENT BY PHYSICAL EXTENT — all 6 permutations enumerated.")
    print("      Envelope (GE Invenia ABUS 2.0 documentation): depth "
          f"{SCANNER_ENVELOPE_MM['depth']}, lateral {SCANNER_ENVELOPE_MM['lateral']}, "
          f"sweep {SCANNER_ENVELOPE_MM['sweep']} mm.")
    print("      0.200 mm = element pitch (lateral); 0.475674 mm = 169 mm travel / ~355 frames")
    print("      (sweep); 0.073 mm = the remaining, axial/depth.\n")
    print(f"  {'d0':>10} {'d1':>10} {'d2':>10} | {'ext d0':>8} {'ext d1':>8} {'ext d2':>8} | "
          f"{'in envelope':>12} {'beam agrees':>12}")
    for f in feasible:
        print("  " + " ".join(f"{s:>10.6g}" for s in f["spacing"]) + " | "
              + " ".join(f"{e:>8.1f}" for e in f["extent_mm"]) + " | "
              + f"{'YES' if f['in_envelope'] else 'no':>12} "
              + f"{'YES' if f['beam_agrees'] else 'no':>12}")

    print(f"\n  funnel: 6 permutations -> {len(by_envelope)} inside the scanner envelope "
          f"-> {len(surviving)} that ALSO put depth on the measured beam axis {AX[beam]}")
    if chosen is None:
        print(f"  *** {len(surviving)} survive — NOT resolved. Falling back to the declared "
              "map; treat every mm figure below as provisional.")
    else:
        print(f"  EXACTLY ONE permutation survives both: "
              f"({spacing[0]:.6g}, {spacing[1]:.6g}, {spacing[2]:.6g}) mm  ->  "
              f"{AX[0]}={chosen['roles'][0]}, {AX[1]}={chosen['roles'][1]}, {AX[2]}={chosen['roles'][2]}")
        print("\n  Third, independent line — does the sample-count structure fit these roles?")
        for a in range(3):
            vals = sorted({r["shape"][a] for r in rows})
            kind = ("FIXED across the split" if len(vals) == 1 else
                    f"takes {len(vals)} discrete values" if len(vals) <= 3 else "varies freely")
            expect = {"lateral": "a fixed transducer -> expect FIXED",
                      "depth": "an operator depth setting (2.5-6.0 cm) -> expect a FEW values",
                      "sweep": "a trimmed acquisition length -> expect FREE variation"}[chosen["roles"][a]]
            print(f"    {AX[a]} = {chosen['roles'][a]:>7}: {kind:<26} | {expect}")
    print(f"\n  declared conventions.SPACING_STORAGE_MM = "
          f"({C.SPACING_STORAGE_MM[0]:.6g}, {C.SPACING_STORAGE_MM[1]:.6g}, {C.SPACING_STORAGE_MM[2]:.6g})")
    agree = bool(np.allclose(spacing, C.SPACING_STORAGE_MM))
    print(f"  agreement with the measured map = "
          f"{'YES' if agree else 'NO — declared and measured DISAGREE (nothing is changed by this script)'}")

    _fig_axis_profiles(profiles, beam, out_dir)
    for vid, planes in exemplars.items():
        _fig_planes(vid, planes, beam, sweep, lateral, spacing, out_dir)

    return {"beam_axis": beam, "sweep_axis": sweep, "lateral_axis": lateral,
            "spacing_measured_mm": spacing, "votes": {AX[a]: int(votes.get(a, 0)) for a in range(3)},
            "agrees_with_declared": agree, "resolved": chosen is not None,
            "permutation_table": feasible, "per_volume": df.to_dict("records")}


def _fig_axis_profiles(profiles, beam, out_dir: Path):
    plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for a, ax in enumerate(axes):
        stack = [p for p in profiles[a]]
        for p in stack:
            ax.plot(np.linspace(0, 1, len(p)), p, color="tab:blue", alpha=0.25, lw=0.8)
        L = min(len(p) for p in stack)
        med = np.median(np.stack([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(p)), p)
                                  for p in stack]), axis=0)
        ax.plot(np.linspace(0, 1, L), med, color="k", lw=2)
        ax.set_title(f"{AX[a]}" + ("  <- MEASURED BEAM AXIS" if a == beam else ""),
                     fontsize=10, fontweight=("bold" if a == beam else "normal"))
        ax.set_xlabel("position along axis (fraction)")
    axes[0].set_ylabel("mean intensity (norm.)")
    fig.suptitle("Stage 1 — per-axis mean-intensity profile, every volume (median in black)",
                 fontsize=11)
    _save(fig, out_dir / "fig1_axis_profiles.png")


def _oriented(planes, row_ax: int, col_ax: int) -> np.ndarray:
    """The stored mid-plane spanning {row_ax, col_ax}, transposed so rows = ``row_ax``."""
    img = planes[tuple(sorted((row_ax, col_ax)))]
    return img if row_ax < col_ax else img.T


def _fig_planes(vid, planes, beam, sweep, lateral, spacing, out_dir: Path):
    """Orthogonal mid-planes in TRUE physical proportions, depth vertical with 0 at the top.

    Drawn the way a reader would see them: if the measured roles are right, the axial
    panel is an ordinary B-mode frame (bright skin/coupling line along the top, tissue
    fading with depth, shadows descending) and the coronal panel is the en-face C-plane
    (no skin line, no descending shadows). That is the check no statistic can replace.
    """
    plt = _plt()
    if plt is None:
        return
    panels = [("AXIAL / transverse (native B-mode frame)", beam, lateral),
              ("CORONAL (C-plane, en face)", sweep, lateral),
              ("SAGITTAL", beam, sweep)]
    role = {beam: "depth", sweep: "sweep", lateral: "lateral"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, r_ax, c_ax) in zip(axes, panels):
        img = _oriented(planes, r_ax, c_ax)
        h_mm = img.shape[0] * spacing[r_ax]
        w_mm = img.shape[1] * spacing[c_ax]
        ax.imshow(img, cmap="gray", origin="upper", aspect="equal", extent=(0, w_mm, h_mm, 0))
        ax.set_title(f"{name}\n{w_mm:.0f} x {h_mm:.0f} mm", fontsize=9)
        ax.set_xlabel(f"{AX[c_ax]} = {role[c_ax]} (mm)", fontsize=8)
        ax.set_ylabel(f"{AX[r_ax]} = {role[r_ax]} (mm)", fontsize=8)
    fig.suptitle(f"Stage 1 — case {vid}: orthogonal mid-planes at TRUE physical scale, under the "
                 f"MEASURED map (beam={AX[beam]}, sweep={AX[sweep]}, lateral={AX[lateral]})",
                 fontsize=10)
    _save(fig, out_dir / f"fig2_planes_case{vid}.png")


# ==========================================================================
# Stage 2
# ==========================================================================

def stage2(cases, pool: pd.DataFrame, geom: dict, out_dir: Path, split: str,
           detector: str, max_volumes=None, n_exemplars: int = 3):
    beam, sweep, lateral = geom["beam_axis"], geom["sweep_axis"], geom["lateral_axis"]
    spacing = np.asarray(geom["spacing_measured_mm"], dtype=float)

    planes = [(beam, "CORONAL (C-plane)", (lateral, sweep)),
              (sweep, "AXIAL (B-mode frame)", (lateral, beam)),
              (lateral, "SAGITTAL", (beam, sweep))]

    vids = [v for v in sorted(pool["public_id"].unique()) if int(v) in cases]
    vids = vids[: max_volumes or len(vids)]
    per_vol, cand_rows = [], []

    # Exemplars are the most FP-heavy volumes — where localisation structure, if there is
    # any, has the best chance of being visible. Chosen from the pool alone, so their
    # figures are drawn inside the single volume pass below rather than re-reading.
    fp_by_vid = pool[pool["label"] == "neg"].groupby("public_id").size()
    exemplar_ids = {int(v) for v in fp_by_vid.sort_values(ascending=False).index[:n_exemplars]}

    # marginals (Fig 3) accumulate here so no extra pass over the volumes is needed
    NB = 40
    marg_inten = {a: [] for a in range(3)}
    marg_hist = {(a, t): np.zeros(NB) for a in range(3) for t in ("tp", "fp")}

    for vid in vids:
        # Inv. 11 ignore band (IoU 0.10-0.30) is DROPPED, not folded into the FPs. Those are
        # partial hits sitting on or beside the lesion, so counting them as false positives
        # would drag the FP population toward the TPs and mute any real difference.
        g = pool[(pool["public_id"] == vid) & (pool["label"].isin(("pos", "neg")))]
        tp = g[g["label"] == "pos"]
        fp = g[g["label"] == "neg"]
        if len(fp) == 0 and len(tp) == 0:
            continue
        vol, _ = load_array(cases[int(vid)].data)
        rec = {"public_id": int(vid), "n_tp": len(tp), "n_fp": len(fp)}

        for a in range(3):
            p = ig.axis_profile(vol, a)
            marg_inten[a].append(np.interp(np.linspace(0, 1, NB), np.linspace(0, 1, len(p)), p))
        for t, lab in (("tp", "pos"), ("fp", "neg")):
            sub = g[g["label"] == lab]
            if len(sub):
                cen = ig.storage_boxes(sub)["cen"]
                for a in range(3):
                    h, _ = np.histogram(cen[:, a] / vol.shape[a], bins=NB, range=(0, 1))
                    marg_hist[(a, t)] += h

        b_all = ig.storage_boxes(g, spacing=spacing)
        st = ig.box_intensity_stats(vol, b_all["cen"], b_all["ext"], depth_axis=beam)
        is_tp = (g["label"].to_numpy() == "pos")
        for k in ("inside", "depth_baseline", "contrast", "distal_contrast"):
            rec[f"{k}_tp_med"] = float(np.nanmedian(st[k][is_tp])) if is_tp.any() else np.nan
            rec[f"{k}_fp_med"] = float(np.nanmedian(st[k][~is_tp])) if (~is_tp).any() else np.nan
        rec["d_contrast"] = ig.cliffs_delta(st["contrast"][is_tp], st["contrast"][~is_tp])
        rec["d_distal"] = ig.cliffs_delta(st["distal_contrast"][is_tp], st["distal_contrast"][~is_tp])
        cand_rows.append(pd.DataFrame({
            "public_id": int(vid), "label": np.where(is_tp, "TP", "FP"),
            "contrast": st["contrast"], "distal_contrast": st["distal_contrast"],
            "inside": st["inside"], "depth_baseline": st["depth_baseline"]}))

        # per-plane correspondence, on a shared CELL_MM grid
        for drop, pname, (r_ax, c_ax) in planes:
            shape2d = (vol.shape[r_ax], vol.shape[c_ax])
            gshape = ig.grid_shape(shape2d, (spacing[r_ax], spacing[c_ax]), CELL_MM)
            proj = np.asarray(vol).mean(axis=drop, dtype=np.float64)
            if r_ax > c_ax:
                proj = proj.T
            imap = ig.block_mean(proj, gshape)
            maps = {}
            for tag, sub in (("tp", tp), ("fp", fp)):
                pts = ig.storage_boxes(sub)["cen"][:, [r_ax, c_ax]] if len(sub) else np.zeros((0, 2))
                maps[tag] = ig.plane_count_map(pts, shape2d, gshape)
                # NEGATED intensity, so rho > 0 == candidates where the plane is DARK
                rec[f"rho_{pname.split()[0].lower()}_{tag}"] = ig.map_spearman(-imap, maps[tag])
            # Restricted to cells holding at least one candidate. Over ALL cells the
            # statistic is dominated by tissue-vs-dark-margin, where nothing can be
            # detected anyway; this asks the question actually posed — among the places
            # candidates land, do FPs prefer the darker ones?
            occ = (maps["tp"] + maps["fp"]) > 0
            rec[f"rho_{pname.split()[0].lower()}_fp_occ"] = ig.map_spearman(-imap, maps["fp"], where=occ)
            rec[f"n_occ_{pname.split()[0].lower()}"] = int(occ.sum())

        # alignment, normalised by the volume's own physical extent so the answer is not
        # just "the breast is 173 mm wide and 50 mm deep"
        extent_mm = [vol.shape[a] * spacing[a] for a in range(3)]
        for tag, sub in (("tp", tp), ("fp", fp)):
            f = ig.spread_fractions(ig.storage_boxes(sub, spacing=spacing)["cen_mm"], extent_mm) \
                if len(sub) >= 2 else np.full(3, np.nan)
            for a in range(3):
                rec[f"spread_{tag}_{AX[a]}"] = float(f[a])
        if len(fp) >= 2:
            cs = ig.coronal_stacking(ig.storage_boxes(fp, spacing=spacing)["cen_mm"], beam, CELL_MM)
            cc = cs["cell_counts"]
            rec["fp_cells"] = int(len(cc))
            rec["fp_frac_in_cells_ge3"] = float(cc[cc >= 3].sum() / cc.sum()) if cc.sum() else np.nan
            rec["fp_depth_spread_med_mm"] = float(np.nanmedian(cs["cell_depth_spread"])) \
                if np.isfinite(cs["cell_depth_spread"]).any() else np.nan

        per_vol.append(rec)
        if int(vid) in exemplar_ids:          # drawn now, while the volume is in memory
            _fig_correspondence(vol, g, geom, out_dir, int(vid), split)
            _fig_banding(vol, g, geom, out_dir, int(vid), split)
        del vol

    dfv = pd.DataFrame(per_vol)
    dfc = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame()

    _print_stage2(dfv, dfc, split, detector)
    _fig_marginals(marg_inten, marg_hist, geom, out_dir, split, detector)
    _fig_local_intensity(dfc, out_dir, split, detector)
    _fig_alignment(dfv, geom, out_dir, split, detector)

    return {"per_volume": dfv.to_dict("records"),
            "n_volumes": int(len(dfv)), "detector": detector, "split": split}


def _print_stage2(dfv: pd.DataFrame, dfc: pd.DataFrame, split: str, detector: str):
    print("\n" + "=" * 78)
    print(f"# STAGE 2 — INTENSITY <-> CANDIDATE CORRESPONDENCE  ({split} / {detector} / "
          f"{len(dfv)} volumes)\n")

    print(f"# 2a. PLANE CORRESPONDENCE — does candidate density sit where the plane is DARK?")
    print(f"     Spearman(-intensity, candidate count) on a shared {CELL_MM:.0f} mm grid. "
          "POSITIVE = dark.")
    print("     Per volume, then summarised; 'sign' = fraction of volumes with rho > 0.")
    print("     ALL   = every grid cell. Dominated by tissue-vs-dark-margin, where no candidate")
    print("             could ever land, so it mostly restates 'candidates are in tissue'.")
    print("     OCC   = only cells holding >=1 candidate. This is the question actually asked:")
    print("             among the places candidates go, do FPs prefer the darker ones?\n")
    print(f"  {'plane':>10} {'rho FP ALL':>11} {'sign':>6} | {'rho FP OCC':>11} {'p25':>7} {'p75':>7} "
          f"{'sign':>6} {'cells':>7} | {'rho TP ALL':>11}")
    for pname in ("axial", "coronal", "sagittal"):
        f, t, o = dfv.get(f"rho_{pname}_fp"), dfv.get(f"rho_{pname}_tp"), dfv.get(f"rho_{pname}_fp_occ")
        if f is None:
            continue
        f = f.dropna(); t = t.dropna() if t is not None else pd.Series(dtype=float)
        o = o.dropna() if o is not None else pd.Series(dtype=float)
        nocc = dfv.get(f"n_occ_{pname}")
        print(f"  {pname.upper():>10} {f.median():>11.3f} {(f > 0).mean():>6.2f} | "
              f"{(o.median() if len(o) else float('nan')):>11.3f} "
              f"{(o.quantile(.25) if len(o) else float('nan')):>7.3f} "
              f"{(o.quantile(.75) if len(o) else float('nan')):>7.3f} "
              f"{((o > 0).mean() if len(o) else float('nan')):>6.2f} "
              f"{(nocc.median() if nocc is not None else float('nan')):>7.0f} | "
              f"{(t.median() if len(t) else float('nan')):>11.3f}")

    print("\n# 2b. 3-D INTENSITY AT THE CANDIDATE (depth-matched; negative = darker than its")
    print("      own depth band). delta = Cliff's TP vs FP, per volume then pooled.\n")
    if len(dfc):
        for col, lab in (("contrast", "inside the box"), ("distal_contrast", "slab distal to it")):
            tp = dfc.loc[dfc["label"] == "TP", col]
            fp = dfc.loc[dfc["label"] == "FP", col]
            pv = dfv[f"d_{'contrast' if col == 'contrast' else 'distal'}"].dropna()
            print(f"  {lab:>20}  TP med={tp.median():>8.2f}  FP med={fp.median():>8.2f}  "
                  f"pooled delta={ig.cliffs_delta(tp, fp):>6.3f}  perVol med={pv.median():>6.3f}  "
                  f"sign={(np.sign(pv) == np.sign(pv.median())).mean():>4.2f}")

    print("\n# 2c. ALIGNMENT — share of centroid spread per axis, normalised by the volume's own")
    print("      extent (a cloud filling the volume scores 0.33 each, whatever its shape).")
    print("      Without that normalisation every cloud looks flat in depth, because the")
    print("      volume is ~173 x 50 x 168 mm — a fact about the FOV, not the candidates.\n")
    print(f"  {'subset':>8} " + " ".join(f"{AX[a]:>10}" for a in range(3)))
    for tag in ("tp", "fp"):
        cols = [f"spread_{tag}_{AX[a]}" for a in range(3)]
        if all(c in dfv for c in cols):
            print(f"  {tag.upper():>8} " + " ".join(f"{dfv[c].median():>10.3f}" for c in cols))

    if "fp_frac_in_cells_ge3" in dfv:
        print("\n# 2d. BEAM-LINE STACKING — FPs binned into 5 mm CORONAL cells (the plane a")
        print("      beam-direction shadow ray collapses to a point in).\n")
        print(f"  frac of FPs in cells holding >=3: med={dfv['fp_frac_in_cells_ge3'].median():.3f} "
              f"p25={dfv['fp_frac_in_cells_ge3'].quantile(.25):.3f} "
              f"p75={dfv['fp_frac_in_cells_ge3'].quantile(.75):.3f}")
        print(f"  depth spread within a multi-occupancy cell (p90-p10, mm): "
              f"med={dfv['fp_depth_spread_med_mm'].median():.1f}")


# ------------------------------------------------------------------ figures

def _fig_marginals(inten, hist, geom, out_dir: Path, split: str, detector: str):
    """The side-by-side comparison: intensity along each axis, above candidate density.

    Read each column top-vs-bottom. If candidates were driven by darkness, the FP peak would
    sit where the intensity curve is low. The intensity panel is shown precisely so the
    depth/attenuation confound is visible rather than hidden inside a summary statistic.
    """
    plt = _plt()
    if plt is None or not inten[0]:
        return
    beam = geom["beam_axis"]
    role = _roles(geom)
    nb = len(next(iter(hist.values())))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6), sharex=True)
    x = np.linspace(0, 1, nb)
    for a in range(3):
        st = np.stack(inten[a])
        ax = axes[0, a]
        ax.fill_between(x, np.percentile(st, 25, 0), np.percentile(st, 75, 0),
                        color="0.75", alpha=0.7)
        ax.plot(x, np.median(st, 0), "k", lw=2)
        ax.set_title(f"{AX[a]} — {role[a]}" + ("  (BEAM)" if a == beam else ""),
                     fontsize=10, fontweight=("bold" if a == beam else "normal"))
        if a == 0:
            ax.set_ylabel("mean intensity\n(median + IQR)")
        ax2 = axes[1, a]
        for t, col in (("fp", "tab:red"), ("tp", "tab:blue")):
            h = hist[(a, t)]
            ax2.step(x, h / max(h.sum(), 1), where="mid", color=col, label=t.upper())
        ax2.set_xlabel("position along axis (fraction)")
        if a == 0:
            ax2.set_ylabel("candidate density")
        ax2.legend(fontsize=8)
    fig.suptitle(f"Fig 3 — intensity vs candidate marginals, {split}/{detector}. "
                 "Read the columns TOP-vs-BOTTOM: does the candidate peak sit where intensity is low?",
                 fontsize=10)
    _save(fig, out_dir / f"fig3_marginals_{split}_{detector}.png")


def _fig_correspondence(vol, g, geom, out_dir: Path, vid: int, split: str):
    plt = _plt()
    if plt is None:
        return
    beam, sweep, lateral = geom["beam_axis"], geom["sweep_axis"], geom["lateral_axis"]
    spacing = np.asarray(geom["spacing_measured_mm"], float)
    tp, fp = g[g["label"] == "pos"], g[g["label"] == "neg"]

    # Same plane order and the same row/col choice as Fig 2, so the two figures can be laid
    # side by side. rho is unaffected by the choice (transposing both maps together permutes
    # the same paired cells) — this is purely so the reader is not re-orienting each time.
    panels = [(sweep, "AXIAL (B-mode frame)", (beam, lateral)),
              (beam, "CORONAL (C-plane)", (sweep, lateral)),
              (lateral, "SAGITTAL", (beam, sweep))]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    for j, (drop, pname, (r_ax, c_ax)) in enumerate(panels):
        proj = np.asarray(vol).mean(axis=drop, dtype=np.float64)
        if r_ax > c_ax:
            proj = proj.T
        shape2d = (vol.shape[r_ax], vol.shape[c_ax])
        gshape = ig.grid_shape(shape2d, (spacing[r_ax], spacing[c_ax]), CELL_MM)

        # Both rows are drawn in millimetres at TRUE physical aspect. Stretching a panel to
        # fill its axes would make "these candidates line up along an axis" a statement about
        # the layout rather than about the anatomy — the exact question this figure exists to
        # answer honestly.
        h_mm = shape2d[0] * spacing[r_ax]
        w_mm = shape2d[1] * spacing[c_ax]
        ext = (0.0, w_mm, h_mm, 0.0)

        ax = axes[0, j]
        ax.imshow(proj, cmap="gray", aspect="equal", origin="upper", extent=ext)
        for sub, style in ((fp, dict(s=9, c="tab:red", alpha=0.65, marker=".")),
                           (tp, dict(s=42, facecolors="none", edgecolors="cyan", lw=1.2))):
            if len(sub):
                cen = ig.storage_boxes(sub)["cen"]
                ax.scatter(cen[:, c_ax] * spacing[c_ax], cen[:, r_ax] * spacing[r_ax], **style)
        ax.set_title(f"{pname}\nmean intensity + centroids (red=FP, cyan=TP)", fontsize=9)
        ax.set_xlabel(f"{AX[c_ax]} = {_roles(geom)[c_ax]} (mm)", fontsize=8)
        ax.set_ylabel(f"{AX[r_ax]} = {_roles(geom)[r_ax]} (mm)", fontsize=8)

        ax = axes[1, j]
        pts = ig.storage_boxes(fp)["cen"][:, [r_ax, c_ax]] if len(fp) else np.zeros((0, 2))
        cmap_ = ig.plane_count_map(pts, shape2d, gshape)
        tmap = ig.plane_count_map(
            ig.storage_boxes(tp)["cen"][:, [r_ax, c_ax]] if len(tp) else np.zeros((0, 2)),
            shape2d, gshape)
        imap = ig.block_mean(proj, gshape)
        rho = ig.map_spearman(-imap, cmap_)
        rho_occ = ig.map_spearman(-imap, cmap_, where=(cmap_ + tmap) > 0)
        ax.imshow(cmap_, cmap="magma", aspect="equal", origin="upper", extent=ext)
        # Both numbers, because the map's huge black background is exactly what makes the
        # ALL-cells version look like a result when it is really "candidates are in tissue".
        ax.set_title(f"FP density ({CELL_MM:.0f} mm cells)\nrho(dark, FP): all cells {rho:+.3f}  |  "
                     f"candidate cells {rho_occ:+.3f}", fontsize=9)
        ax.set_xlabel(f"{AX[c_ax]} (mm)", fontsize=8); ax.set_ylabel(f"{AX[r_ax]} (mm)", fontsize=8)
    fig.suptitle(f"Fig 4 — case {vid} ({split}): a beam-direction shadow is a POINT in the "
                 f"coronal panel and a STRIPE in the other two.", fontsize=10)
    _save(fig, out_dir / f"fig4_correspondence_{split}_vol{vid}.png")


def _fig_local_intensity(dfc: pd.DataFrame, out_dir: Path, split: str, detector: str):
    plt = _plt()
    if plt is None or not len(dfc):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(
            axes, ("contrast", "distal_contrast"),
            ("inside the candidate box", "slab directly distal (posterior shadow)")):
        # centre each volume on its own median so volumes with different dynamic
        # range can share one histogram; the TP-vs-FP comparison is unaffected.
        d = dfc.copy()
        d[col] = d[col] - d.groupby("public_id")[col].transform("median")
        lo, hi = np.nanpercentile(d[col], [1, 99])
        bins = np.linspace(lo, hi, 45)
        for lab, col_ in (("FP", "tab:red"), ("TP", "tab:blue")):
            v = d.loc[d["label"] == lab, col].dropna()
            ax.hist(v, bins=bins, density=True, alpha=0.5, color=col_, label=f"{lab} (n={len(v)})")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{title}\ndepth-matched, per-volume centred", fontsize=9)
        ax.set_xlabel("intensity − depth-matched baseline (centred)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle(f"Fig 5 — is a candidate darker than its own depth band? {split}/{detector}",
                 fontsize=10)
    _save(fig, out_dir / f"fig5_local_intensity_{split}_{detector}.png")


def _fig_alignment(dfv: pd.DataFrame, geom, out_dir: Path, split: str, detector: str):
    plt = _plt()
    if plt is None or not len(dfv):
        return
    role = _roles(geom)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    ax = axes[0]
    for i, tag in enumerate(("tp", "fp")):
        for a in range(3):
            c = f"spread_{tag}_{AX[a]}"
            if c not in dfv:
                continue
            v = dfv[c].dropna()
            x = a + (i - 0.5) * 0.28
            ax.scatter(np.full(len(v), x) + np.random.default_rng(a).normal(0, 0.03, len(v)),
                       v, s=14, alpha=0.6, color=("tab:blue" if tag == "tp" else "tab:red"),
                       label=(tag.upper() if a == 0 else None))
            ax.plot([x - 0.1, x + 0.1], [v.median()] * 2, color="k", lw=2)
    ax.axhline(1 / 3, ls="--", color="0.5", lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{AX[a]}\n{role[a]}" for a in range(3)], fontsize=8)
    ax.set_ylabel("fraction of centroid variance (mm)")
    ax.set_title("per-volume spread per axis (dashed = isotropic)", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    if "fp_frac_in_cells_ge3" in dfv:
        ax.scatter(dfv["fp_frac_in_cells_ge3"], dfv["fp_depth_spread_med_mm"],
                   s=22, color="tab:red", alpha=0.7)
        ax.set_xlabel(f"frac of FPs in {CELL_MM:.0f} mm coronal cells holding >=3")
        ax.set_ylabel("depth spread within those cells (mm)")
        ax.set_title("beam-line stacking: many FPs per coronal cell,\nspread far in depth = a ray",
                     fontsize=9)
    fig.suptitle(f"Fig 6 — do candidates align along an axis? {split}/{detector}", fontsize=10)
    _save(fig, out_dir / f"fig6_alignment_{split}_{detector}.png")


def _detector_grid_period_native(shape, lateral_axis: int) -> float:
    """The FPN P3 anchor stride expressed in NATIVE voxels along ``lateral_axis``.

    A peak here in the FP-position spectrum is the network's own sampling lattice, not
    tissue. Two rescalings sit between the network and the native volume, and both must be
    undone: native -> iso cache, then iso slice -> the detector's resized input.

    Note this deliberately uses ``conventions.SPACING_STORAGE_MM`` — the DECLARED map — and
    not the measured one, because the declared map is what ``preprocess`` actually applied
    when the cache was built. That is a statement of what happened, not an endorsement of it.
    """
    iso = [int(round(shape[a] * C.SPACING_STORAGE_MM[a] / C.ISO_SPACING_MM)) for a in range(3)]
    h, w = iso[C.IN_PLANE_ROW_AXIS], iso[C.IN_PLANE_COL_AXIS]   # the 2.5D slice the detector sees
    scale = min(C.DET_MIN_SIZE / max(min(h, w), 1), C.DET_MAX_SIZE / max(max(h, w), 1))
    iso_per_native = max(iso[lateral_axis], 1) / max(shape[lateral_axis], 1)
    return 8.0 / max(scale, 1e-9) / max(iso_per_native, 1e-9)


def _fig_banding(vol, g, geom, out_dir: Path, vid: int, split: str):
    plt = _plt()
    if plt is None:
        return
    beam, sweep, lateral = geom["beam_axis"], geom["sweep_axis"], geom["lateral_axis"]
    fp = g[g["label"] == "neg"]

    # one axial B-mode frame: the slice holding the most FPs
    cen = ig.storage_boxes(g)["cen"]
    s = int(np.median(cen[:, sweep])) if len(cen) else vol.shape[sweep] // 2
    frame = np.asarray(vol.take(s, axis=sweep))
    # take() leaves the two surviving axes in ASCENDING storage order, so transpose exactly
    # when depth is the larger of the two — otherwise the "distal half" slice below would be
    # taken along lateral and the profile would answer a different question entirely.
    if beam > lateral:
        frame = frame.T                                    # rows = beam(depth), cols = lateral
    n_depth = frame.shape[0]
    lat_prof = frame[n_depth // 2:, :].mean(axis=0, dtype=np.float64)   # distal half

    spacing = np.asarray(geom["spacing_measured_mm"], float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    axes[0].imshow(frame, cmap="gray", aspect="equal", origin="upper",
                   extent=(0, frame.shape[1] * spacing[lateral], frame.shape[0] * spacing[beam], 0))
    near = fp[np.abs(ig.storage_boxes(fp)["cen"][:, sweep] - s) < 12] if len(fp) else fp
    if len(near):
        c = ig.storage_boxes(near)["cen"]
        axes[0].scatter(c[:, lateral] * spacing[lateral], c[:, beam] * spacing[beam],
                        s=14, c="tab:red", alpha=0.8)
    axes[0].set_xlabel(f"{AX[lateral]} = lateral (mm)", fontsize=8)
    axes[0].set_ylabel(f"{AX[beam]} = depth (mm)", fontsize=8)
    axes[0].set_title(f"case {vid}: axial B-mode frame at {AX[sweep]}={s} "
                      f"(rows={AX[beam]} depth, cols={AX[lateral]} lateral); red = FP centroids "
                      f"within 12 slices", fontsize=9)

    axes[1].plot(lat_prof, color="k", lw=1)
    axes[1].set_title("lateral mean-intensity profile over the DISTAL half of the depth range "
                      "— parallel shadows appear as periodic dips", fontsize=9)
    axes[1].set_xlabel(f"{AX[lateral]} (lateral, voxels)")

    per_i, pow_i = ig.power_spectrum_1d(lat_prof)
    axes[2].plot(per_i, pow_i, color="k", lw=1, label="intensity profile")
    if len(fp) >= 8:
        h, _ = np.histogram(ig.storage_boxes(fp)["cen"][:, lateral],
                            bins=len(lat_prof), range=(0, vol.shape[lateral]))
        per_f, pow_f = ig.power_spectrum_1d(h.astype(float))
        if len(per_f):
            axes[2].plot(per_f, pow_f, color="tab:red", lw=1, alpha=0.8, label="FP lateral positions")
    grid = _detector_grid_period_native(vol.shape, lateral)
    axes[2].axvline(grid, color="tab:green", ls="--", lw=1.2,
                    label=f"detector grid period ~{grid:.0f} vox (FPN P3 stride 8)")
    axes[2].set_xlim(2, min(200, len(lat_prof) / 2))
    axes[2].set_xlabel("period (lateral voxels)"); axes[2].set_ylabel("normalised power")
    axes[2].set_title("periodicity: a shared peak = the detector fires on the bands; "
                      "a peak at the green line is the anchor grid, not anatomy", fontsize=9)
    axes[2].legend(fontsize=8)
    _save(fig, out_dir / f"fig7_banding_{split}_vol{vid}.png")


# ------------------------------------------------------------------ utils

def _roles(geom) -> dict:
    r = {}
    r[geom["beam_axis"]] = "depth/beam"
    r[geom["sweep_axis"]] = "sweep/elev"
    r[geom["lateral_axis"]] = "lateral"
    return r


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:      # pragma: no cover
        print(f"(figures skipped: {type(e).__name__}: {e})")
        return None


def _save(fig, path: Path):
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=115)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  figure -> {path}")


def main() -> int:
    # The single knob in Stage 2, rebound once from --cell-mm so the printed tables and the
    # figures are guaranteed to be drawn on the same grid the correlations were computed on.
    global CELL_MM
    ap = argparse.ArgumentParser(description="Intensity <-> candidate probe (read-only)")
    add_phase3_paths(ap)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--candidates", default=None,
                    help="path to the candidate record (parquet/csv). Default: "
                         "<out-root>/candidates/candidates_<split>")
    ap.add_argument("--detector", default=None,
                    help="detector_of_origin to analyse; default = the one with most volumes")
    ap.add_argument("--max-volumes", type=int, default=None)
    ap.add_argument("--n-exemplars", type=int, default=3)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--cell-mm", type=float, default=CELL_MM,
                    help=f"grid cell for the 2-D correspondence maps (default {CELL_MM} mm). "
                         "Sweep it to check the plane correlations are not a binning artefact.")
    args = ap.parse_args()
    profile_banner()          # Inv. 6: name the substrate that produced this output
    CELL_MM = float(args.cell_mm)

    out_dir = Path(args.out_root) / "intensity_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(split_root(args, args.split))
    print(f"[{args.split}] {len(cases)} cases under {split_root(args, args.split)}")

    geom = stage1(cases, out_dir, args.max_volumes, args.n_exemplars)
    result = {"stage1": geom}

    if not args.stage1_only:
        from abus_jcr.candidates.record import read_candidate_record
        cpath = Path(args.candidates) if args.candidates else \
            Path(args.out_root) / "candidates" / f"candidates_{args.split}"
        try:
            pool = read_candidate_record(cpath.with_suffix(""))
        except FileNotFoundError as e:
            # read_candidate_record tries .parquet then falls back to .csv, swallowing the
            # parquet error. The usual cause on a laptop is a pyarrow version older than the
            # one that wrote the file, which reports a bare FileNotFoundError for the CSV.
            raise SystemExit(
                f"could not read the candidate record at {cpath.with_suffix('')}.\n"
                f"  {type(e).__name__}: {e}\n"
                f"  If the .parquet is present but the .csv is not, the parquet was most likely\n"
                f"  written by a NEWER pyarrow than this machine has "
                f"(local pyarrow reads it as 'Repetition level histogram size mismatch').\n"
                f"  Fix: copy the CSV mirror that sits next to it on the server, e.g.\n"
                f"    scp thesis-server:.../candidates/candidates_{args.split}.csv {cpath.parent}/") from e
        pool = pool[pool["split"] == args.split]
        dets = pool.groupby("detector_of_origin")["public_id"].nunique().sort_values()
        det = args.detector or str(dets.index[-1])
        if len(dets) > 1:
            print(f"[{args.split}] detectors present: {list(dets.index)} -> using '{det}' "
                  f"(override with --detector) so each volume contributes ONE set (Inv. 7)")
        pool = pool[pool["detector_of_origin"] == det]
        if pool.empty:
            # Otherwise every downstream table prints its headers over zero rows, which
            # reads exactly like a genuine "no signal" result. Fail loudly instead.
            raise SystemExit(f"no candidates for detector '{det}' in split '{args.split}'. "
                             f"Available: {list(dets.index)}")
        result["stage2"] = stage2(cases, pool, geom, out_dir, args.split, det,
                                  args.max_volumes, args.n_exemplars)

    jp = out_dir / f"intensity_probe_{args.split}.json"
    jp.write_text(json.dumps(result, indent=2, default=str))
    print(f"\njson = {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
