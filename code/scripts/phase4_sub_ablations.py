"""[4.8] Sub-ablations on FULL + the λ = 0 pure-ranking diagnostics.

**Token-block on/off** (on FULL, at its selected λ, 3 seeds each): ``rank``, ``score_stats``,
``tube_geom``. The tube-geom block is the split pair — on the promoted val pool ``area_cv``
carries signal (Cliff's δ **0.474**) while ``centroid_jitter`` is near-dead and
wrong-signed (δ **−0.292**) — so this ablation reports whether the block earns its place
rather than assuming it does.

**λ = 0 diagnostics** on A2 and FULL: the pure-per-volume-ranking endpoint. The calibration
analysis PREDICTS it loses to B2 (smooth-AP is invariant to a per-set constant, and
cross-volume calibration is **78.7 %** of the headroom on the promoted pool, [F.8]).
Reported whichever way it lands, and **excluded from the selection budget**.

This is a driver over ``phase4_train_variant.py``, so the schedule, the trial budget and the
selection rule are literally the same code as the main grid.

Usage:
    python scripts/phase4_sub_ablations.py --device cuda --b2-alpha 0.25 --b2-lr 1e-3 \\
        --full-lambda 1.0 --out-root ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from abus_jcr import conventions as C
from abus_jcr.rescore.variants import SUB_ABLATION_BLOCKS

from _phase4_common import add_phase4_paths, dump_json


def _run(args, variant: str, seed: int, blocks, tag: str, lam=None) -> dict:
    cmd = [sys.executable, str(Path(__file__).with_name("phase4_train_variant.py")),
           "--variant", variant, "--seed", str(seed), "--device", args.device,
           "--epochs", str(args.epochs), "--n-boot", str(args.n_boot),
           "--phase1-out", args.phase1_out, "--phase3-out", args.phase3_out,
           "--data-root", args.data_root, "--out-root", args.out_root,
           "--b2-alpha", str(args.b2_alpha), "--b2-lr", str(args.b2_lr),
           "--tag", tag, "--blocks", *blocks]
    if lam is not None:
        cmd += ["--lambda-override", str(lam)]
    if args.record_suffix:
        cmd += ["--record-suffix", args.record_suffix]
    print(f"\n{'='*78}\n# [4.8] {variant} seed{seed} {tag}\n{'='*78}", flush=True)
    subprocess.run(cmd, check=True)
    p = Path(args.out_root) / "variants" / f"{variant}_{tag}_seed{seed}.json"
    rep = json.loads(p.read_text())
    return {"variant": variant, "seed": seed, "tag": tag, "blocks": list(blocks),
            "lambda": lam, "val_cpm": rep["deployed"]["val_cpm"],
            "selected_epoch": rep["deployed"]["selected_epoch"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.8] FULL sub-ablations + lambda=0 diagnostics")
    add_phase4_paths(ap)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=C.RESC_SET_EPOCHS)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.RESC_SEEDS))
    ap.add_argument("--b2-alpha", type=float, required=True, help="B2's selected alpha ([4.6])")
    ap.add_argument("--b2-lr", type=float, required=True, help="B2's selected lr ([4.6])")
    ap.add_argument("--full-lambda", type=float, required=True,
                    help="FULL's selected lambda ([4.6]) — the sub-ablations run AT it")
    ap.add_argument("--skip-blocks", action="store_true")
    ap.add_argument("--skip-lambda0", action="store_true")
    args = ap.parse_args()

    results = {"block_ablations": [], "lambda0_diagnostics": []}

    if not args.skip_blocks:
        for block in SUB_ABLATION_BLOCKS:
            keep = [b for b in C.RESC_TOKEN_BLOCKS if b != block]
            for seed in args.seeds:
                results["block_ablations"].append(
                    _run(args, "FULL", seed, keep, f"no{block}", lam=args.full_lambda))

    if not args.skip_lambda0:
        # FULL-P added 2026-09-03 (the loop predated the -P wiring): the lambda=0 endpoint reads
        # DIFFERENTLY on the two objectives — per-set smooth-AP alone is shift-invariant and must
        # collapse, the pooled surrogate is not and may hold — and RB [4.8] reports them side by
        # side. The -P machinery (warm start, reference table, froc loss) engages automatically
        # through run_variant_trial.
        for variant in ("A2", "FULL", "FULL-P"):
            for seed in args.seeds:
                results["lambda0_diagnostics"].append(
                    _run(args, variant, seed, list(C.RESC_TOKEN_BLOCKS), "lam0", lam=0.0))

    print(f"\n{'='*78}\n# [4.8] SUB-ABLATION SUMMARY (val CPM, mean over seeds)\n")
    import numpy as np
    for name, rows in results.items():
        if not rows:
            continue
        print(f"  {name}:")
        for tag in sorted({r["tag"] + "|" + r["variant"] for r in rows}):
            t, v = tag.split("|")
            sel = [r["val_cpm"] for r in rows if r["tag"] == t and r["variant"] == v]
            print(f"    {v:<5} {t:<14} {np.mean(sel):.4f} +/- {np.std(sel):.4f}  (n={len(sel)})")
    print("\n# Compare each against FULL's own number from [4.7] with a PAIRED interval; the "
          "lambda=0 rows are the pure-ranking endpoint the calibration analysis predicts will "
          "LOSE to B2 — record whichever way it lands.")

    dump_json(results, Path(args.out_root) / "grid" / "sub_ablations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
