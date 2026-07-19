"""[P3-UPDATE L1] Derive the tube drift caps from Train GT — standalone, no detector needed.

Prints ``LINK_MAX_TUBE_ZSPAN`` / ``LINK_MAX_CENTROID_DRIFT`` (iso space) from the Phase-1
Train union GT boxes, so they can be SET in conventions.py and a linker re-check run BEFORE
the full [3.3'] freeze (which needs the retrained fold detectors). Reuses the exact
``derive_link_caps`` that [3.3'] uses, so the values match.

Usage:
    python scripts/phase3_derive_caps.py --phase1-out /home/maia-user/Andre2/outputs/phase1
"""

from __future__ import annotations

import argparse
import sys

from _phase2_common import load_slice_boxes
from _phase3_common import derive_link_caps, add_phase3_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="[L1] derive tube drift caps from Train GT")
    add_phase3_paths(parser)
    args = parser.parse_args()

    caps = derive_link_caps(load_slice_boxes(args, "Train"))
    print("# [L1] Train-GT drift-cap derivation (SET these in conventions.py, then re-run [P3U.0a])\n")
    print(f"  zspan_p99            = {caps['zspan_p99']:.2f} iso slices")
    print(f"  inplane_extent_p99   = {caps['inplane_extent_p99']:.2f} iso px\n")
    print(f"  LINK_MAX_TUBE_ZSPAN     = {caps['LINK_MAX_TUBE_ZSPAN']}   # round({caps['zspan_safety']} * zspan_p99)")
    print(f"  LINK_MAX_CENTROID_DRIFT = {caps['LINK_MAX_CENTROID_DRIFT']}   # round({caps['drift_safety']} * inplane_extent_p99)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
