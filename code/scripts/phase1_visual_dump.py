"""Phase 1 — overlay derived 2D box + mask contour on ISO slices (eye-check).

For a few cases, renders several lesion-bearing slices with the mask contour and
the mask-derived 2D box drawn on the isotropic B-mode frame (d0 vertical/depth,
d1 horizontal/lateral). The mask is ALWAYS shown alongside the box (invariant of
the visual checks). Saves PNGs under <out-root>/figures/.

Usage:
    python scripts/phase1_visual_dump.py --split Validation --out-root ./_p1_out --cases 100 104 116
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.conventions import SLICE_AXIS
from abus_jcr.slice_labels import boxes_for_slice

DEFAULT_OUT_ROOT = "$WORK/outputs/phase1"


def _lesion_slices(mask_iso: np.ndarray):
    return [z for z in range(mask_iso.shape[SLICE_AXIS]) if np.take(mask_iso, z, axis=SLICE_AXIS).any()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 visual dump")
    parser.add_argument("--split", default="Validation")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cases", type=int, nargs="+", default=[100, 104, 116])
    parser.add_argument("--per-case", type=int, default=3, help="slices per case")
    args = parser.parse_args()

    cache_root = Path(args.out_root) / "cache"
    K.assert_hash(cache_root)
    fig_dir = Path(args.out_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for cid in args.cases:
        vol = np.asarray(K.open_vol(cache_root, cid))
        mask = np.asarray(K.open_mask(cache_root, cid))
        zs = _lesion_slices(mask)
        if not zs:
            print(f"case {cid}: no lesion slices, skip")
            continue
        # sample per-case slices evenly across the z-run
        picks = np.linspace(0, len(zs) - 1, min(args.per_case, len(zs))).round().astype(int)
        for k in picks:
            z = zs[int(k)]
            frame = np.take(vol, z, axis=SLICE_AXIS)      # (d0, d1)
            mframe = np.take(mask, z, axis=SLICE_AXIS)
            boxes = boxes_for_slice(mframe)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(frame, cmap="gray", origin="upper")  # d0 down = depth
            ax.contour(mframe, levels=[0.5], colors="lime", linewidths=1.0)
            for (r0, c0, r1, c1) in boxes:
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), (c1 - c0 + 1), (r1 - r0 + 1),
                                       fill=False, edgecolor="red", linewidth=1.2))
            # Labels from the MEASURED physical axis roles (results/AXIS_CHECK.md), not from
            # the historical "d0 = depth" assumption they carried until 2026-08-09 — that was
            # the inverted map, and a mislabelled orientation figure is exactly the hazard this
            # dump exists to catch. The PIXELS are unchanged (no transpose), so every figure
            # already in report/REPORT_PHASE_1.md is still the same image; only the axis names
            # are corrected. Consequence to expect: `d0` is LATERAL and is the vertical axis
            # here, so depth runs LEFT->RIGHT and the bright skin/coupling line is at the LEFT
            # edge, not the top. It is a transposed B-mode frame, on BOTH profiles.
            row_role = "depth" if C.DEPTH_AXIS == C.IN_PLANE_ROW_AXIS else "lateral"
            col_role = "depth" if C.DEPTH_AXIS == C.IN_PLANE_COL_AXIS else "lateral"
            ax.set_title(f"case {cid}  z={z}  [{C.AXIS_PROFILE}]  "
                         f"(d0={row_role} down, d1={col_role})")
            ax.set_xlabel(f"d1 ({col_role})")
            ax.set_ylabel(f"d0 ({row_role})")
            out = fig_dir / f"overlay_{args.split}_{cid}_z{z}.png"
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
            print(f"case {cid} z={z}: {len(boxes)} box(es) -> {out}")

    print(f"\n**DONE** — {len(written)} overlay PNG(s) under {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
