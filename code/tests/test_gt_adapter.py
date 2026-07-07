"""The GT column adapter maps documented names to the official schema, and the
per-case double-check compares the recomputed whole-mask box to the CSV row.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.conventions import GT_COLUMNS
from abus_jcr.gt_labels import to_official_gt, doublecheck_case


def _documented_frame():
    # documented header id,c_x,c_y,c_z,len_x,len_y,len_z (row = DATA_INFO case 100)
    return pd.DataFrame(
        [(100, 189.0, 242.0, 314.0, 72, 368, 302)],
        columns=["id", "c_x", "c_y", "c_z", "len_x", "len_y", "len_z"],
    )


def test_to_official_gt_yields_exactly_official_columns():
    out = to_official_gt(_documented_frame())
    assert list(out.columns) == GT_COLUMNS


def test_to_official_gt_maps_documented_row_to_expected_official_row():
    out = to_official_gt(_documented_frame()).iloc[0]
    assert int(out["public_id"]) == 100
    assert (out["coordX"], out["coordY"], out["coordZ"]) == (189.0, 242.0, 314.0)
    assert (out["x_length"], out["y_length"], out["z_length"]) == (72, 368, 302)


def test_doublecheck_case_zero_residual_for_matching_mask():
    # a mask whose inclusive hull is storage (5,3,2)-(9,7,5) -> official
    # (3.5, 5.0, 7.0, 3, 4, 4)
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:10, 3:8, 2:6] = 1
    gt_row = {"coordX": 3.5, "coordY": 5.0, "coordZ": 7.0,
              "x_length": 3.0, "y_length": 4.0, "z_length": 4.0}
    residual = doublecheck_case(mask, gt_row)
    assert max(residual.values()) == 0


def test_doublecheck_case_reports_nonzero_residual_on_mismatch():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:10, 3:8, 2:6] = 1
    gt_row = {"coordX": 3.5, "coordY": 5.0, "coordZ": 7.0,
              "x_length": 3.0, "y_length": 4.0, "z_length": 5.0}  # z off by 1
    residual = doublecheck_case(mask, gt_row)
    assert residual["z_length"] == pytest.approx(1.0)
    assert max(residual.values()) > 0
