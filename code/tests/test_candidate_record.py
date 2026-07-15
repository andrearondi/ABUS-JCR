"""Candidate-record persistence + official-pred-CSV row alignment (Phase 3).

The two frozen artefacts: the candidate feature record (Parquet + CSV mirror) and
the official prediction CSV that ``evaluate()`` consumes. The load-bearing invariant
is ROW ALIGNMENT — candidate ``k`` in the record must be row ``k`` in the pred CSV
(the CSV drops ``candidate_id``; the join is by construction-preserved order).
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.candidates.record import (
    CANDIDATE_COLUMNS,
    write_candidate_record,
    read_candidate_record,
    to_official_pred_csv,
)
from abus_jcr.eval.froc import PRED_COLUMNS


def _record(n=3, score_max=None):
    score_max = score_max if score_max is not None else [0.9, 0.5, 0.2]
    rows = []
    for k in range(n):
        rows.append({
            "public_id": 100, "candidate_id": f"full_seed0:100:{k}",
            "detector_of_origin": "full_seed0", "split": "val", "fold": -1,
            "coordX": 10.0 + k, "coordY": 20.0, "coordZ": 30.0,
            "x_length": 5.0, "y_length": 6.0, "z_length": 7.0,
            "score_max": score_max[k], "score_mean": 0.4, "score_std": 0.1,
            "score_min": 0.1, "slice_count": 3, "z_span": 4, "fill_ratio": 0.75,
            "rank": k + 1, "rank_norm": (k + 1) / n,
            "label": "pos" if k == 0 else "neg", "iou_gt": 0.5 if k == 0 else 0.0,
            "cen_d0": 1.0, "cen_d1": 2.0, "cen_d2": 3.0,
            "ext_d0": 4.0, "ext_d1": 5.0, "ext_d2": 6.0,
            "preprocess_hash": "deadbeef",
        })
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def test_record_roundtrip(tmp_path):
    df = _record()
    fmt = write_candidate_record(df, tmp_path / "cand_val")
    assert fmt in ("parquet+csv", "csv-only")
    back = read_candidate_record(tmp_path / "cand_val")
    assert list(back.columns) == CANDIDATE_COLUMNS
    pd.testing.assert_frame_equal(back.reset_index(drop=True), df.reset_index(drop=True),
                                  check_dtype=False)


def test_to_official_pred_csv_row_alignment(tmp_path):
    df = _record()
    pred_path = tmp_path / "pred.csv"
    pred = to_official_pred_csv(df, prob_col="score_max", path=pred_path)
    assert list(pred.columns) == PRED_COLUMNS
    # candidate k in the record <-> row k in the CSV
    assert len(pred) == len(df)
    assert pred["probability"].tolist() == pytest.approx(df["score_max"].tolist())
    for col in ("coordX", "coordY", "coordZ", "x_length", "y_length", "z_length"):
        assert pred[col].tolist() == pytest.approx(df[col].tolist())
    # re-read from disk preserves order
    on_disk = pd.read_csv(pred_path)
    assert on_disk["probability"].tolist() == pytest.approx(df["score_max"].tolist())


def test_to_official_pred_csv_rejects_probability_ge_one(tmp_path):
    df = _record(score_max=[0.9, 1.0, 0.2])  # 1.0 is out of [0,1)
    with pytest.raises(ValueError):
        to_official_pred_csv(df, prob_col="score_max", path=tmp_path / "bad.csv")
