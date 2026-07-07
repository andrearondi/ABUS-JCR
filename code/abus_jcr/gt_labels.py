"""Ground-truth adapter and the mask↔box double-check.

The shipped ``bbx_labels.csv`` uses documented column names
(``id, c_x, ...``); the official ``det_score.py`` expects
``public_id, coordX, ...``. This module is the single I/O boundary that
reconciles the two, and it recomputes each box from the mask to confirm the
CSV boxes are mask-derived (Inv. 3 / Inv. 11: tolerance 0 — the box is
*defined* by the mask, so any nonzero residual flags a corrupted case).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .conventions import GT_COLUMNS, GT_RENAME
from .geometry import mask_to_official_box

_OFFICIAL_FIELDS = ("coordX", "coordY", "coordZ", "x_length", "y_length", "z_length")


def load_gt_documented(csv: Path) -> pd.DataFrame:
    """Read the shipped ``bbx_labels.csv`` (documented header
    ``id,c_x,c_y,c_z,len_x,len_y,len_z``)."""
    return pd.read_csv(csv)


def to_official_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Rename documented columns to the official schema.

    Asserts the result is exactly ``GT_COLUMNS``; ``public_id`` stays the
    integer case id (matches DATA/MASK ``case_id``).
    """
    out = df.rename(columns=GT_RENAME)
    assert list(out.columns) == GT_COLUMNS, (
        f"adapted GT columns {list(out.columns)} != official {GT_COLUMNS}"
    )
    return out


def adapt_gt_csv(in_csv: Path, out_csv: Path) -> Path:
    """File-level adapter: produce the official-named GT CSV fed to
    ``evaluate()`` (Train/Val now, Test only at final eval)."""
    out = to_official_gt(load_gt_documented(in_csv))
    out.to_csv(out_csv, index=False)
    return Path(out_csv)


def doublecheck_case(mask: np.ndarray, gt_row: Mapping) -> Dict[str, float]:
    """Per-field absolute difference between the whole-mask recomputed box and
    the official GT row. Match tolerance = 0; any ``> 0`` flags a bad case."""
    recomputed = mask_to_official_box(mask)  # (coordX, coordY, coordZ, x_len, y_len, z_len)
    residual: Dict[str, float] = {}
    for field, value in zip(_OFFICIAL_FIELDS, recomputed):
        residual[field] = abs(float(value) - float(gt_row[field]))
    return residual
