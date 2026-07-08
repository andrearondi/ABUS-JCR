"""The FROZEN common per-slice detection schema (Phase 2 -> Phase 3).

Data-independent: columns, dtypes, the half-open coordinate contract, and a
write/read round-trip. No torch, no cache — runs in the laptop env.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.detect import schema as S


def _good_df():
    return pd.DataFrame({
        "volume_id": [100, 100],
        "slice_z": [10, 11],
        "x1": [1.0, 2.0], "y1": [3.0, 4.0],
        "x2": [5.0, 6.0], "y2": [7.0, 8.0],
        "score": [0.9, 0.1],
    })


def test_columns_are_frozen_and_ordered():
    assert S.DETECTION_COLUMNS == ["volume_id", "slice_z", "x1", "y1", "x2", "y2", "score"]


def test_validate_accepts_wellformed_frame():
    df = S.validate_detections(_good_df())
    assert list(df.columns[:len(S.DETECTION_COLUMNS)]) == S.DETECTION_COLUMNS


def test_validate_rejects_nonpositive_width():
    df = _good_df()
    df.loc[0, "x2"] = df.loc[0, "x1"]  # x2 <= x1
    with pytest.raises(ValueError):
        S.validate_detections(df)


def test_validate_rejects_nonpositive_height():
    df = _good_df()
    df.loc[1, "y2"] = df.loc[1, "y1"] - 1.0  # y2 < y1
    with pytest.raises(ValueError):
        S.validate_detections(df)


def test_validate_rejects_out_of_range_score():
    df = _good_df()
    df.loc[0, "score"] = 1.5
    with pytest.raises(ValueError):
        S.validate_detections(df)


def test_validate_rejects_missing_column():
    df = _good_df().drop(columns=["score"])
    with pytest.raises(ValueError):
        S.validate_detections(df)


def test_empty_detections_is_valid_and_zero_rows():
    df = S.empty_detections()
    assert len(df) == 0
    assert list(df.columns) == S.DETECTION_COLUMNS
    S.validate_detections(df)  # must not raise


def test_write_read_round_trip(tmp_path):
    df = _good_df()
    path = tmp_path / "det.parquet"
    S.write_detections(df, path)
    back = S.read_detections(path)
    # same coordinates and scores survive the round-trip
    for col in S.DETECTION_COLUMNS:
        np.testing.assert_allclose(
            back[col].to_numpy(dtype=float), df[col].to_numpy(dtype=float)
        )
