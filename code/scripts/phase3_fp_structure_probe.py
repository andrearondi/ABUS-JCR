"""[3.7] Phase-0b FP-structure probe on the Val RetinaNet candidate pool.

Runs ``probe.fp_structure.fp_structure_probe`` on the frozen Val record and writes the
FP-vs-TP clustering/anisotropy table + a one-paragraph verdict. The verdict decides the
Phase-4 geometry-term claim scope: structure PRESENT -> "relational"; ABSENT ->
"set-level contextual calibration".

Usage:
    python scripts/phase3_fp_structure_probe.py --out-root /home/maia-user/Andre2/outputs/phase3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from abus_jcr.candidates.record import read_candidate_record
from abus_jcr.probe.fp_structure import fp_structure_probe
from _phase3_common import add_phase3_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.7] FP-structure probe (Val)")
    add_phase3_paths(parser)
    args = parser.parse_args()

    pool = read_candidate_record(Path(args.out_root) / "candidates" / "candidates_val")
    res = fp_structure_probe(pool, split_filter="val")

    fp, tp, v = res["fp"], res["tp"], res["verdict"]
    print("# [3.7] Phase-0b FP-structure probe (Val, iso space)\n")
    print(f"{'group':>4} {'n':>6} {'aniso med':>10} {'NN med':>9} {'clusters/vol med':>18}")
    print(f"{'FP':>4} {fp['n']:>6} {fp['anisotropy_median']:>10.3f} {fp['nn_dist_median']:>9.2f} "
          f"{fp['clusters_per_vol_median']:>18.2f}")
    print(f"{'TP':>4} {tp['n']:>6} {tp['anisotropy_median']:>10.3f} {tp['nn_dist_median']:>9.2f} "
          f"{tp['clusters_per_vol_median']:>18.2f}")
    print(f"\neffect (Cliff's delta, FP vs TP): anisotropy={res['effect']['anisotropy_cliffs_delta']:.3f}, "
          f"NN-dist={res['effect']['nn_dist_cliffs_delta']:.3f}")

    present = v["structure_present"]
    verdict = (
        "FP geometry shows EXPLOITABLE STRUCTURE: false positives are more depth-elongated "
        f"(anisotropy {fp['anisotropy_median']:.2f} vs TP {tp['anisotropy_median']:.2f}) and more "
        f"spatially clustered (NN {fp['nn_dist_median']:.1f} vs {tp['nn_dist_median']:.1f}, "
        f">1 cluster/vol) than true positives. Phase-4 geometry term -> RELATIONAL."
    ) if present else (
        "FP geometry shows NO clear structure vs TPs (not jointly more elongated AND more "
        "clustered). Phase-4 geometry term -> SET-LEVEL CONTEXTUAL CALIBRATION."
    )
    print(f"\nVERDICT: {verdict}")

    out_dir = Path(args.out_root) / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_out = {**res, "verdict_text": verdict}
    (out_dir / "fp_structure.json").write_text(json.dumps(res_out, indent=2, default=str))

    # anisotropy scatter (FP vs TP centroids, depth extent) — best-effort
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        val = pool[pool["split"] == "val"]
        fig, ax = plt.subplots(figsize=(5, 4))
        for lab, color in [("neg", "tab:red"), ("pos", "tab:blue")]:
            g = val[val["label"] == lab]
            aniso = g["ext_d0"] / ((g["ext_d1"] + g["ext_d2"]) / 2.0).replace(0, float("nan"))
            ax.scatter(g["cen_d2"], aniso, s=8, alpha=0.4, color=color,
                       label=("FP" if lab == "neg" else "TP"))
        ax.axhline(1.0, ls="--", color="k", alpha=0.4)
        ax.set_xlabel("iso slice (d2)"); ax.set_ylabel("anisotropy ext_d0 / mean(d1,d2)")
        ax.set_title("[3.7] FP vs TP depth-anisotropy"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "fp_structure.png", dpi=120); plt.close(fig)
        print(f"figure = {out_dir / 'fp_structure.png'}")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")
    print(f"json = {out_dir / 'fp_structure.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
