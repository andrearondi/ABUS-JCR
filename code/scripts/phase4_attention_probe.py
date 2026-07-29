"""[4.9] Interpretability — data-driven, not decorative (exit check 9).

On the already-identified FP-heavy val sets (``val_FP-heavy_full_seed2_vol122`` plus the two
other [P3U2.PD] exemplars), report for A1/FULL vs B2/B1:

* the **mean attention mass a TP places on its co-located TP peers vs on FPs** — the direct
  test of the recorded TP-TP-vs-TP-FP ``|δ| ≈ 0.94`` structure (a TP's geometry to its
  co-located TP peers is highly distinctive; FP clusters run 13.5/vol vs 1.0 for TPs);
* the **refined-score change on the FP cluster** relative to B0's ``score_max`` ranking.

The printed verdict states plainly whether the geometry-aware rungs down-weight the cluster
more. **B1 has no attention at all** — that is the point of the contrast, and it is reported
as ``attention: none (per-candidate MLP)`` rather than silently skipped.

Usage:
    python scripts/phase4_attention_probe.py --device cuda --out-root ... \\
        --sets full_seed2:122 full_seed0:110 full_seed1:121
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.probe.candidate_diag import cluster_counts

from _phase4_common import (add_phase4_paths, assert_device, boxes_of, dump_json,
                            load_deployed_model, load_variant_inputs, seed_detector)


def _parse_sets(specs):
    out = []
    for s in specs:
        det, pid = s.split(":")
        if not det.startswith("full_seed"):
            raise SystemExit(f"--sets takes val sets (full_seed<r>:<public_id>); got {s!r}")
        out.append((det, int(pid), int(det.replace("full_seed", ""))))
    return out


def _forward_set(model, feats, coord, length, device):
    """Returns ``(mean attention (N,N) or None, probability (N,))`` for ONE set."""
    import torch

    with torch.no_grad():
        f = torch.as_tensor(feats[None], dtype=torch.float32, device=device)
        c = torch.as_tensor(coord[None], dtype=torch.float32, device=device)
        l = torch.as_tensor(length[None], dtype=torch.float32, device=device)
        m = torch.ones(1, feats.shape[0], dtype=torch.bool, device=device)
        logits, attns = model(f, c, l, m, return_attn=True)
        prob = torch.sigmoid(logits).clamp(0.0, 1.0 - C.RESC_PROB_EPS).squeeze(0).cpu().numpy()
    if not attns:
        return None, prob
    return np.mean([a.squeeze(0).mean(0).cpu().numpy() for a in attns], axis=0), prob


def _attention_mass(attn, labels):
    """Mean attention a TP row places on its TP peers vs on FPs (self-attention excluded)."""
    is_tp = labels == "pos"
    is_fp = labels == "neg"
    if attn is None or not is_tp.any():
        return None
    rows = attn[is_tp]
    diag = np.diag(attn)[is_tp]
    return {"tp_to_tp_peers": float(np.mean(rows[:, is_tp].sum(axis=1) - diag)),
            "tp_to_fp": float(np.mean(rows[:, is_fp].sum(axis=1))),
            "tp_self": float(np.mean(diag)),
            "n_tp": int(is_tp.sum()), "n_fp": int(is_fp.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.9] attention interpretability probe")
    add_phase4_paths(ap)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sets", nargs="+",
                    default=["full_seed2:122", "full_seed0:110", "full_seed1:121"],
                    help="detector:public_id pairs — the [P3U2.PD] FP-heavy exemplars")
    ap.add_argument("--variants", nargs="+", default=["B1", "B2", "A1", "FULL"])
    ap.add_argument("--cluster-radius", type=float, default=C.FP_PROBE_CLUSTER_RADIUS)
    args = ap.parse_args()
    assert_device(args.device)

    targets = _parse_sets(args.sets)
    seeds = sorted({t[2] for t in targets})
    inputs = {s: load_variant_inputs(args, s, C.RESC_TOKEN_BLOCKS) for s in seeds}
    models = {s: {v: load_deployed_model(args, v, s, inputs[s]["d_in"], args.device)[0]
                  for v in args.variants} for s in seeds}

    report = []
    for det, pid, seed in targets:
        rec = inputs[seed]["rec_va"]
        if det != seed_detector(seed):
            raise SystemExit(f"{det} does not match seed {seed}'s detector {seed_detector(seed)}")
        sel = np.flatnonzero(rec["public_id"].to_numpy().astype(int) == pid)
        if not len(sel):
            raise SystemExit(f"volume {pid} not present in the {det} val pool")
        sub = rec.iloc[sel]
        labels = sub["label"].to_numpy()
        coord, length = boxes_of(sub)
        feats = np.ascontiguousarray(inputs[seed]["Zva"][sel], dtype=np.float32)

        # the FP cluster this probe is about ([P3U2.13]: 13.5 FP clusters/vol vs 1.0 for TPs)
        fp_mask = labels == "neg"
        n_cl, _, redund = cluster_counts(coord[fp_mask], args.cluster_radius) if fp_mask.any() \
            else (0, 0, float("nan"))
        base = sub["score_max"].to_numpy(float)
        base_fp_mean = float(base[fp_mask].mean()) if fp_mask.any() else float("nan")

        entry = {"set": f"{det}:{pid}", "seed": seed, "n_cands": int(len(sub)),
                 "n_tp": int((labels == "pos").sum()), "n_fp": int(fp_mask.sum()),
                 "n_ignore": int((labels == "ignore").sum()),
                 "fp_clusters": int(n_cl), "fp_cluster_redundancy": float(redund),
                 "b0_fp_mean_score": base_fp_mean, "rungs": {}}

        print(f"\n{'='*78}\n# [4.9] set {det}:{pid} — {entry['n_cands']} candidates "
              f"({entry['n_tp']} TP / {entry['n_fp']} FP / {entry['n_ignore']} ignore), "
              f"{n_cl} FP clusters @ r={args.cluster_radius}\n")
        print(f"  {'rung':<6} {'TP->TP peers':>13} {'TP->FP':>9} {'TP self':>9} "
              f"{'FP mean prob':>13} {'TP mean prob':>13}")

        for variant in args.variants:
            attn, prob = _forward_set(models[seed][variant], feats, coord, length, args.device)
            mass = _attention_mass(attn, labels)
            fp_mean = float(prob[fp_mask].mean()) if fp_mask.any() else float("nan")
            tp_mean = float(prob[labels == "pos"].mean()) if (labels == "pos").any() else float("nan")
            entry["rungs"][variant] = {
                "attention": mass, "fp_mean_prob": fp_mean, "tp_mean_prob": tp_mean,
                "tp_minus_fp_prob": tp_mean - fp_mean,
                "has_attention": attn is not None,
            }
            if mass is None:
                print(f"  {variant:<6} {'none (per-candidate MLP)':>33} "
                      f"{fp_mean:>13.4f} {tp_mean:>13.4f}")
            else:
                print(f"  {variant:<6} {mass['tp_to_tp_peers']:>13.4f} {mass['tp_to_fp']:>9.4f} "
                      f"{mass['tp_self']:>9.4f} {fp_mean:>13.4f} {tp_mean:>13.4f}")

        geo = [v for v in ("A1", "FULL") if v in entry["rungs"]]
        plain = [v for v in ("B2", "B1") if v in entry["rungs"]]
        if geo and plain:
            g = np.mean([entry["rungs"][v]["fp_mean_prob"] for v in geo])
            p = np.mean([entry["rungs"][v]["fp_mean_prob"] for v in plain])
            entry["geometry_downweights_fp_cluster"] = bool(g < p)
            print(f"\n  VERDICT: geometry-aware rungs {geo} give the FP cluster mean prob "
                  f"{g:.4f} vs {p:.4f} for {plain} -> "
                  f"{'they DO down-weight it' if g < p else 'they do NOT down-weight it'}")
        report.append(entry)

    print(f"\n# Read this against the recorded pairwise prior: TP-TP vs TP-FP |delta| ~ 0.94 "
          f"(strong, the density/consensus signal the set attention captures natively) but "
          f"TP-FP vs FP-FP max |delta| = 0.082 (weak — no per-pair FP-suppression signal). "
          f"A null geometry effect here is a PRE-REGISTERED outcome, not a bug.")
    dump_json({"sets": report, "cluster_radius": args.cluster_radius,
               "variants": args.variants},
              Path(args.out_root) / "grid" / "attention_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
