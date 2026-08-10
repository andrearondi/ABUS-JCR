"""[4.2] ONE-TIME crop-geometry selection (seed 0 only) — then FROZEN for everything.

The brief's literal 48³ voxel crop is inadequate on this dataset (Train union-box in-plane
diag p50 = 54.1 / p99 = 231.1 iso px; locally 5/6 Val lesions exceed 48 iso voxels
laterally, up to 183), which is exactly the "size audit shows lesions routinely exceeding
this" contingency PHASE_4.md §1.1 anticipated. This step runs the two sanctioned options
head to head on **B1 val CPM** and freezes the winner for every variant, every seed and
(Phase 6) every detector:

* **(a) adaptive** — cubic ROI, side ``clip(1.5 * max(ext), 48, 160)`` iso vox → 48³ (DEFAULT);
* **(b) fixed control** — a fixed 128³ iso ROI → 48³.

It is a thin driver over ``phase4_pretrain_encoder.py``: the two runs differ ONLY in the
crop geometry, so nothing else can explain the gap.

Usage:
    python scripts/phase4_select_crop_geometry.py --device cuda \\
        --phase1-out ... --phase3-out ... --out-root ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from abus_jcr import conventions as C

from _phase4_common import add_phase4_paths, crops_dir, dump_json, encoder_dir


def child_cmd(args, crop_side, extra_out: Path) -> list:
    """The child ``phase4_pretrain_encoder.py`` invocation for one arm.

    ``--out-root`` is redirected per arm so the two encoders never overwrite each other, but
    ``--crops-root`` stays on the SHARED [4.1] cache. Conflating the two is what made the
    adaptive arm die with ``no CROP_META.json under .../crop_geometry/adaptive/crops/...``
    (the fixed arm cannot expose it — ``--crop-side`` re-extracts and reads no cache).
    """
    cmd = [sys.executable, str(Path(__file__).with_name("phase4_pretrain_encoder.py")),
           "--seed", "0", "--device", args.device,
           "--epochs", str(args.epochs),
           "--phase1-out", args.phase1_out, "--phase3-out", args.phase3_out,
           "--data-root", args.data_root, "--out-root", str(extra_out),
           "--crops-root", str(crops_dir(args))]
    if args.record_suffix:
        cmd += ["--record-suffix", args.record_suffix]
    if crop_side is not None:
        cmd += ["--crop-side", str(crop_side)]
    return cmd


def _run(args, tag: str, crop_side, extra_out: Path) -> dict:
    cmd = child_cmd(args, crop_side, extra_out)
    print(f"\n{'='*78}\n# [4.2] {tag}: {' '.join(cmd)}\n{'='*78}", flush=True)
    subprocess.run(cmd, check=True)
    report = json.loads((encoder_dir(argparse.Namespace(out_root=str(extra_out)), 0)
                         / "pretrain_report.json").read_text())
    sel = report["epochs"][report["selected_epoch"]]
    return {"tag": tag, "crop_side": crop_side, "selected_epoch": report["selected_epoch"],
            "val_cpm": float(sel["val_cpm"]), "val_ci_lo": float(sel["val_ci_lo"]),
            "val_ci_hi": float(sel["val_ci_hi"]), "out_root": str(extra_out)}


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.2] freeze the crop geometry on B1 val CPM")
    add_phase4_paths(ap)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=C.RESC_ENC_EPOCHS)
    ap.add_argument("--control-side", type=float, default=128.0,
                    help="the fixed-ROI control side in iso voxels (default 128)")
    args = ap.parse_args()

    root = Path(args.out_root) / "crop_geometry"
    adaptive = _run(args, "adaptive (clip(1.5*max(ext), 48, 160))", None, root / "adaptive")
    fixed = _run(args, f"fixed {args.control_side:.0f}^3 iso", args.control_side, root / "fixed")

    winner = adaptive if adaptive["val_cpm"] >= fixed["val_cpm"] else fixed
    print(f"\n{'='*78}\n# [4.2] RESULT — crop geometry frozen on B1 val CPM (seed 0)\n")
    for r in (adaptive, fixed):
        mark = "  <-- WINNER" if r is winner else ""
        print(f"  {r['tag']:<40} val CPM {r['val_cpm']:.4f} "
              f"[{r['val_ci_lo']:.4f}, {r['val_ci_hi']:.4f}]  ep{r['selected_epoch']}{mark}")
    print(f"\n# FROZEN: {winner['tag']}")
    if winner is not adaptive:
        print("# ACTION REQUIRED: the fixed control won. Set RESC_CROP_CONTEXT/MIN/MAX in "
              "conventions.py so roi_side_iso returns the fixed side, rebuild the crop cache "
              "([4.1]), and record the change in RESULTS_PHASE_4 [4.2]. Do NOT proceed with a "
              "mixed geometry.")
    else:
        print("# No change needed: conventions.py already encodes the adaptive ROI.")
    print("# This choice is now CONSTANT for every variant, every seed and (Phase 6) every "
          "detector — exit check 3.")

    dump_json({"adaptive": adaptive, "fixed": fixed, "winner": winner["tag"],
               "decided_on": "B1 val CPM, seed 0", "epochs": args.epochs},
              root / "crop_geometry_choice.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
