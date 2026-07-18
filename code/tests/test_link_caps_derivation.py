"""[P3-UPDATE L1] Train-GT drift-cap derivation (torch-free).

``derive_link_caps`` reads the Phase-1 iso union GT boxes and produces the frozen tube caps
from per-volume lesion z-extent (p99) and per-box in-plane extent (p99), scaled by safety.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _phase3_common import derive_link_caps  # noqa: E402


def _sb(rows):
    return pd.DataFrame(rows, columns=["volume_id", "slice_z", "r0", "c0", "r1", "c1"])


def test_caps_scale_from_percentiles():
    # vol 1 spans slices 0..9 (z-extent 10); vol 2 spans 0..4 (z-extent 5).
    rows = []
    for z in range(10):
        rows.append({"volume_id": 1, "slice_z": z, "r0": 0, "c0": 0, "r1": 20, "c1": 30})  # h=21,w=31 -> 31
    for z in range(5):
        rows.append({"volume_id": 2, "slice_z": z, "r0": 0, "c0": 0, "r1": 10, "c1": 10})  # 11
    caps = derive_link_caps(_sb(rows), zspan_safety=1.8, drift_safety=1.5)
    # p99 z-extent ~ 10 (dominated by vol 1); in-plane p99 ~ 31
    assert caps["LINK_MAX_TUBE_ZSPAN"] == int(round(1.8 * caps["zspan_p99"]))
    assert caps["LINK_MAX_CENTROID_DRIFT"] == int(round(1.5 * caps["inplane_extent_p99"]))
    assert caps["zspan_p99"] >= 9.0
    assert caps["inplane_extent_p99"] >= 30.0


def test_caps_are_positive_ints():
    rows = [{"volume_id": 1, "slice_z": z, "r0": 5, "c0": 5, "r1": 15, "c1": 25} for z in range(3)]
    caps = derive_link_caps(_sb(rows))
    assert isinstance(caps["LINK_MAX_TUBE_ZSPAN"], int) and caps["LINK_MAX_TUBE_ZSPAN"] > 0
    assert isinstance(caps["LINK_MAX_CENTROID_DRIFT"], int) and caps["LINK_MAX_CENTROID_DRIFT"] > 0
