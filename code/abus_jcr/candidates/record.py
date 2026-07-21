"""The two frozen Phase-3 artefacts: the candidate feature record + the pred CSV.

1. **Candidate record** (Parquet + CSV mirror) — the per-candidate feature row Phase 4
   consumes: official scoring box, the frozen score-stats vector, within-volume rank,
   the ignore-band label, the iso centre/extents (Phase-4 crop handle), and the cache
   ``preprocess_hash`` (join to the iso cache). Column order is FROZEN.
2. **Official prediction CSV** — exactly what ``eval/froc.evaluate`` consumes. Built by
   selecting the official ``PRED_COLUMNS`` with ``probability = record[prob_col]``,
   **preserving row order** so candidate ``k`` in the record is row ``k`` in the CSV
   (the CSV carries no ``candidate_id``; the join is by construction-preserved order).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..eval.froc import PRED_COLUMNS, write_pred_csv

# FROZEN candidate-record schema (consumed verbatim by Phase 4). Do not reorder/rename.
CANDIDATE_COLUMNS = [
    "public_id", "candidate_id", "detector_of_origin", "split", "fold",
    "coordX", "coordY", "coordZ", "x_length", "y_length", "z_length",   # official native (scoring space)
    "score_max", "score_mean", "score_std", "score_min", "slice_count", "z_span", "fill_ratio",
    # [P3U2 3.D] tube-geometry block (TUBE_GEOM_COLUMNS) — a NEW ablatable Phase-4 block; soft cues, not gates
    "centroid_jitter", "area_cv", "area_peak_pos", "area_monotonicity",
    "rank", "rank_norm",
    "label", "iou_gt",                                                   # pos/neg/ignore + max IoU to GT
    "cen_d0", "cen_d1", "cen_d2", "ext_d0", "ext_d1", "ext_d2",         # iso-voxel centre + extents (Phase-4 crop)
    "preprocess_hash",                                                   # cache handle (join to the iso cache)
]


def validate_candidate_record(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the frozen columns are present; return ``df[CANDIDATE_COLUMNS]``."""
    missing = [c for c in CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"candidate record missing columns: {missing}")
    return df[CANDIDATE_COLUMNS]


def write_candidate_record(df: pd.DataFrame, path) -> str:
    """Validate then write Parquet at ``path`` (+ a ``.csv`` mirror).

    Falls back to CSV-only if no Parquet engine is installed. Returns the format
    written (``"parquet+csv"`` or ``"csv-only"``). Mirrors ``detect/schema.write_detections``.
    """
    df = validate_candidate_record(df)
    path = Path(path)
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return "parquet+csv"
    except Exception as e:  # pyarrow/fastparquet absent
        print(f"note: Parquet not written ({type(e).__name__}: {e}); CSV mirror only")
        return "csv-only"


def read_candidate_record(path) -> pd.DataFrame:
    """Read a candidate record (Parquet if present, else the CSV mirror) + validate."""
    path = Path(path)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        try:
            return validate_candidate_record(pd.read_parquet(parquet))
        except Exception:
            pass
    return validate_candidate_record(pd.read_csv(path.with_suffix(".csv")))


def to_official_pred_csv(record_df: pd.DataFrame, prob_col: str = "score_max",
                         path: Optional[str] = None) -> pd.DataFrame:
    """Select ``PRED_COLUMNS`` with ``probability = record[prob_col]``, order preserved.

    Candidate ``k`` in the record <-> row ``k`` in the CSV. Asserts the probability
    lies in ``[0, 1)`` (detector scores are ``< 1``; the det_score FROC sweep requires
    it). When ``path`` is given, writes via ``eval/froc.write_pred_csv`` (which
    re-validates). Returns the prediction DataFrame regardless.
    """
    validate_candidate_record(record_df)
    prob = pd.to_numeric(record_df[prob_col], errors="raise")
    if not ((prob >= 0.0) & (prob < 1.0)).all():
        raise ValueError(f"{prob_col} must lie in [0, 1) for the det_score FROC sweep")

    pred = pd.DataFrame({
        "public_id": record_df["public_id"].to_numpy(),
        "coordX": record_df["coordX"].to_numpy(),
        "coordY": record_df["coordY"].to_numpy(),
        "coordZ": record_df["coordZ"].to_numpy(),
        "x_length": record_df["x_length"].to_numpy(),
        "y_length": record_df["y_length"].to_numpy(),
        "z_length": record_df["z_length"].to_numpy(),
        "probability": prob.to_numpy(),
    }, columns=PRED_COLUMNS)

    if path is not None:
        write_pred_csv(pred, path)
    return pred
