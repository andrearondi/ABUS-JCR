"""[3.6] Baseline FROC ladder B0 — the Phase-4 floor (Inv. 3, 8, 12).

Per Val seed pool (``detector_of_origin == full_seed{s}``): rank the frozen pool by the
detector's own aggregated score (``score_max``), score it through the official oracle
(``eval/froc.evaluate_froc``), and read CPM (``average_recall``) + the recall ceiling
(``max_recall``, Inv. 8). Report mean+-std over the 3 seeds and a volume-level
``bootstrap_cpm_ci`` per seed. Spot-checks that the pred CSV ``probability`` equals the
record ``score_max`` by row-aligned join.

Usage (server or local with the Val record present):
    python scripts/phase3_baseline_froc.py --out-root /home/maia-user/Andre2/outputs/phase3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.candidates.record import read_candidate_record, to_official_pred_csv
from abus_jcr.eval.froc import evaluate_froc, cpm, recall_ceiling, bootstrap_cpm_ci
from _phase3_common import add_phase3_paths, load_official_gt


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.6] baseline FROC (B0) over the Val seed pools")
    add_phase3_paths(parser)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    rec_base = Path(args.out_root) / "candidates" / "candidates_val"
    pool = read_candidate_record(rec_base)
    gt_df = load_official_gt(args, "val")

    seeds = sorted(pool["detector_of_origin"].unique())
    print(f"# [3.6] Baseline FROC B0 (Val, {len(seeds)} seed pools)\n")
    per_seed = []
    for det in seeds:
        sub = pool[pool["detector_of_origin"] == det].reset_index(drop=True)
        pred = to_official_pred_csv(sub, prob_col="score_max")  # row order preserved
        # spot-check row alignment: probability == record score_max
        assert np.allclose(pred["probability"].to_numpy(), sub["score_max"].to_numpy()), \
            f"{det}: pred.probability != record.score_max (row-alignment broken)"

        res = evaluate_froc(gt_df, pred)
        this_cpm = cpm(res)
        ceiling = recall_ceiling(res)
        ci = bootstrap_cpm_ci(gt_df, pred, n_boot=args.n_boot, seed=0)
        per_seed.append({"detector": det, "cpm": this_cpm, "recall_ceiling": ceiling,
                         "cpm_ci_lo": ci["lo"], "cpm_ci_hi": ci["hi"],
                         "n_candidates": int(len(sub))})
        print(f"{det}: CPM={this_cpm:.4f}  ceiling(max_recall)={ceiling:.4f}  "
              f"95% CI=[{ci['lo']:.4f}, {ci['hi']:.4f}]  n_cand={len(sub)}")

    cpms = np.array([r["cpm"] for r in per_seed])
    ceils = np.array([r["recall_ceiling"] for r in per_seed])
    print(f"\nCPM  mean+-std over {len(seeds)} seeds = {cpms.mean():.4f} +- {cpms.std(ddof=0):.4f}")
    print(f"ceiling mean+-std               = {ceils.mean():.4f} +- {ceils.std(ddof=0):.4f}")
    print("  ^ the recall ceiling is THE key Inv.-8 number; every Phase-4 curve re-ranks this pool.")

    out_dir = Path(args.out_root) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"per_seed": per_seed, "cpm_mean": float(cpms.mean()), "cpm_std": float(cpms.std(ddof=0)),
               "ceiling_mean": float(ceils.mean()), "ceiling_std": float(ceils.std(ddof=0))}
    (out_dir / "baseline_froc.json").write_text(json.dumps(payload, indent=2))
    print(f"json = {out_dir / 'baseline_froc.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
