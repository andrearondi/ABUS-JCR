"""The FROZEN common per-slice detection schema (Phase 2 -> Phase 3/6).

Every detector (RetinaNet here; YOLO / Faster R-CNN via adapters in Phase 6)
emits detections into this exact table so Phase 3's linker needs no reframe.

**Coordinate contract (pinned, all detectors obey it):** boxes live in the
**iso-voxel slice frame**; ``x = d1`` (lateral / column), ``y = d0`` (depth /
row); ``slice_z`` indexes ``SLICE_AXIS = d2`` of the iso cache; boxes are
**half-open floats** (``x1 < x2``, ``y1 < y2``; ``x2``/``y2`` exclusive,
torchvision-native); ``score in [0, 1]``. This is the same space
``slice_dataset.get_stack`` / ``slice_labels`` use, so linking (Phase 3) stacks
boxes across ``slice_z``, forms the iso storage box, then maps to native via
``geometry.iso_storage_to_native_storage`` for official scoring. (The GT
``slice_boxes`` inclusive->half-open conversion is ``x1=c0, y1=r0, x2=c1+1,
y2=r1+1``.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# FROZEN column order — do not reorder or rename (Phase 3 depends on it).
DETECTION_COLUMNS = ["volume_id", "slice_z", "x1", "y1", "x2", "y2", "score"]

# int columns vs float columns for a stable, round-trippable dtype contract.
_INT_COLUMNS = ["volume_id", "slice_z"]
_FLOAT_COLUMNS = ["x1", "y1", "x2", "y2", "score"]


def empty_detections() -> pd.DataFrame:
    """A valid zero-row detection frame with the frozen columns and dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype="int64") for c in _INT_COLUMNS})
    for c in _FLOAT_COLUMNS:
        df[c] = pd.Series(dtype="float64")
    return df[DETECTION_COLUMNS]


def validate_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the frozen schema + coordinate contract; return ``df`` unchanged.

    Checks: all columns present; ``volume_id``/``slice_z`` integral; coordinate
    and score columns numeric; ``x1 < x2`` and ``y1 < y2`` (half-open, positive
    extent) on every row; ``0 <= score <= 1``. Raises ``ValueError`` otherwise.
    """
    missing = [c for c in DETECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"detection frame missing columns: {missing}")

    for c in _INT_COLUMNS:
        if not pd.api.types.is_integer_dtype(df[c]):
            raise ValueError(f"column {c!r} must be integer dtype, got {df[c].dtype}")
    for c in _FLOAT_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c!r} must be numeric dtype, got {df[c].dtype}")

    if len(df) == 0:
        return df

    x1 = df["x1"].to_numpy(dtype=float); x2 = df["x2"].to_numpy(dtype=float)
    y1 = df["y1"].to_numpy(dtype=float); y2 = df["y2"].to_numpy(dtype=float)
    s = df["score"].to_numpy(dtype=float)
    if not np.all(x2 > x1):
        raise ValueError("half-open contract violated: some x2 <= x1")
    if not np.all(y2 > y1):
        raise ValueError("half-open contract violated: some y2 <= y1")
    if not (np.all(s >= 0.0) and np.all(s <= 1.0)):
        raise ValueError("score out of range: require 0 <= score <= 1")
    return df


def write_detections(df: pd.DataFrame, path) -> str:
    """Validate then write Parquet at ``path`` (+ a ``.csv`` mirror).

    Falls back to CSV-only if no Parquet engine is installed. Returns the format
    written ("parquet+csv" or "csv-only").
    """
    validate_detections(df)
    path = Path(path)
    df = df[DETECTION_COLUMNS]
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return "parquet+csv"
    except Exception as e:  # pyarrow/fastparquet absent
        print(f"note: Parquet not written ({type(e).__name__}: {e}); CSV mirror only")
        return "csv-only"


def read_detections(path) -> pd.DataFrame:
    """Read a detection table (Parquet if present, else the CSV mirror) + validate."""
    path = Path(path)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        try:
            df = pd.read_parquet(parquet)
            return validate_detections(df[DETECTION_COLUMNS])
        except Exception:
            pass
    df = pd.read_csv(path.with_suffix(".csv"))
    return validate_detections(df[DETECTION_COLUMNS])
