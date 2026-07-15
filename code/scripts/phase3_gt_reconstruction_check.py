"""[3.2] GEOMETRY GATE — GT reconstruction consistency over all 100 Train volumes.

Detector-INDEPENDENT: it exercises only the linking/coord path (per-slice iso mask
boxes -> one iso tube -> native -> official) against the official ``bbx_labels`` box, so
a coordinate/affine bug is caught before any candidate is trusted. The native-hull
control (native mask -> official, no resampling) is ~1.0 (Phase-0 residual 0); the
gap to it in the iso path is resampling quantization, NOT a bug — hence the tolerance
is the Phase-1 measured fidelity, never ≈1.0.

Asserts: >= RECON_IOU_WARN_FRAC of cases clear RECON_IOU_SOFT (0.85) AND none below
RESAMPLE_IOU_FLOOR (0.50). Must PASS before [3.5].

Usage (server):
    python scripts/phase3_gt_reconstruction_check.py \
        --phase1-out /home/maia-user/Andre2/outputs/phase1 \
        --out-root  /home/maia-user/Andre2/outputs/phase3 --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.geometry import mask_to_official_box, iou_official
from abus_jcr.link.reconstruct import gt_reconstruction_consistency
from _phase3_common import add_phase3_paths, cache_root, load_manifest, load_official_gt, split_root, gt_official_tuple


def main() -> int:
    parser = argparse.ArgumentParser(description="[3.2] GT reconstruction geometry gate")
    add_phase3_paths(parser)
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="split to gate (train = the exit check; val = local precedent)")
    args = parser.parse_args()

    manifest = load_manifest(args)
    croot = cache_root(args)
    gt_idx = load_official_gt(args, args.split).set_index("public_id")
    vids = sorted(int(v) for v in manifest[manifest["split"] == args.split]["volume_id"])

    # native masks for the hull control (io only; no resampling)
    from abus_jcr.io_nrrd import discover_cases, load_array
    cases = discover_cases(split_root(args, args.split))

    rows = []
    for vid in vids:
        meta = K.read_meta(croot, vid)
        mask_iso = np.asarray(K.open_mask(croot, vid))
        gt_official = gt_official_tuple(gt_idx, vid)
        recon_iou = gt_reconstruction_consistency(mask_iso, gt_official, meta)

        native_mask, _ = load_array(cases[vid].mask)
        native_mask = (np.asarray(native_mask) > 0).astype(np.uint8)
        hull_iou = iou_official(mask_to_official_box(native_mask), gt_official)
        rows.append({"public_id": vid, "recon_iou": recon_iou, "hull_iou": hull_iou})

    recon = np.array([r["recon_iou"] for r in rows], dtype=float)
    n = len(recon)
    n_clear_soft = int((recon >= C.RECON_IOU_SOFT).sum())
    n_below_floor = int((recon < C.RESAMPLE_IOU_FLOOR).sum())
    frac_clear = n_clear_soft / n if n else float("nan")

    out_dir = Path(args.out_root) / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"gt_reconstruction_{args.split}.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(table_path, index=False)

    print(f"# [3.2] GT reconstruction gate ({args.split}, n={n})\n")
    print("per-case recon IoU (iso->native->official) vs native-hull control (should be ~1.0):")
    for r in sorted(rows, key=lambda x: x["recon_iou"]):
        flag = "  <-- BELOW FLOOR (BUG)" if r["recon_iou"] < C.RESAMPLE_IOU_FLOOR else (
            "  (< soft)" if r["recon_iou"] < C.RECON_IOU_SOFT else "")
        print(f"  vol {r['public_id']:<4} recon={r['recon_iou']:.4f}  hull={r['hull_iou']:.4f}{flag}")
    print()
    print(f"recon IoU: min={recon.min():.4f} p10={np.percentile(recon,10):.4f} "
          f"median={np.median(recon):.4f} max={recon.max():.4f}")
    print(f"cleared soft ({C.RECON_IOU_SOFT}): {n_clear_soft}/{n} = {frac_clear:.3f} "
          f"(gate needs >= {C.RECON_IOU_WARN_FRAC})")
    print(f"below hard floor ({C.RESAMPLE_IOU_FLOOR}): {n_below_floor} (gate needs 0)")
    print(f"table = {table_path}")

    summary = {
        "split": args.split, "n": n, "min": float(recon.min()),
        "median": float(np.median(recon)), "frac_clear_soft": frac_clear,
        "n_below_floor": n_below_floor,
    }
    (out_dir / f"gt_reconstruction_{args.split}_summary.json").write_text(json.dumps(summary, indent=2))

    ok = (frac_clear >= C.RECON_IOU_WARN_FRAC) and (n_below_floor == 0)
    print(f"\nGATE {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("A FAIL means a linking/coord bug OR the resampling tolerance is wrong — STOP and inspect.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
