"""The scoring oracle wrapper (Inv. 3, 12).

Everything detection-metric flows through here. The official code
(``_official_det_score.py``) is vendored byte-identically and always run on
real CSV files, exactly as at challenge time, so black-box parity is preserved.
No metric is ever re-derived from a summary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..conventions import GT_COLUMNS, PRED_COLUMNS, KEY_FP
from ._official_det_score import evaluate, iou_3d  # vendored, unmodified

__all__ = [
    "evaluate", "iou_3d",
    "evaluate_froc_paths", "evaluate_froc",
    "cpm", "recall_ceiling", "key_recall",
    "write_pred_csv", "bootstrap_cpm_ci",
]


def evaluate_froc_paths(gt_csv, pred_csv) -> dict:
    """Thin pass-through to the vendored ``evaluate()``."""
    return evaluate(str(gt_csv), str(pred_csv))


def evaluate_froc(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tmpdir: Optional[str] = None) -> dict:
    """Write both frames to temp CSVs with the official schemas and call the
    oracle. The official code always runs on real CSV files."""
    _require_columns(gt_df, GT_COLUMNS, "gt_df")
    _require_columns(pred_df, PRED_COLUMNS, "pred_df")
    with tempfile.TemporaryDirectory(dir=tmpdir) as td:
        gt_path = Path(td) / "gt.csv"
        pred_path = Path(td) / "pred.csv"
        gt_df[GT_COLUMNS].to_csv(gt_path, index=False)
        pred_df[PRED_COLUMNS].to_csv(pred_path, index=False)
        return evaluate_froc_paths(gt_path, pred_path)


def cpm(res: dict) -> float:
    """CPM = the detection-track metric = mean interpolated recall at key FPs."""
    return float(res["detection"]["average_recall"])


def recall_ceiling(res: dict) -> float:
    """max_recall = full-pool recall ceiling (the Inv. 8 ceiling, for free)."""
    return float(res["detection"]["max_recall"])


def key_recall(res: dict) -> dict:
    """Recall at each of the seven key FPs."""
    return dict(zip(KEY_FP, res["detection"]["key_recall"]))


def write_pred_csv(rows, path) -> None:
    """Canonical Phase-3+ prediction writer. Validates schema, dtypes, and
    ``0 <= probability < 1`` before writing."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows), columns=PRED_COLUMNS)
    _require_columns(df, PRED_COLUMNS, "prediction rows")
    prob = pd.to_numeric(df["probability"], errors="raise")
    if not ((prob >= 0.0) & (prob < 1.0)).all():
        raise ValueError("probability must lie in [0, 1) for the det_score FROC sweep")
    for col in ("public_id",):
        pd.to_numeric(df[col], errors="raise")
    df[PRED_COLUMNS].to_csv(path, index=False)


def bootstrap_cpm_ci(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Seeded volume-level bootstrap CI on CPM (Inv. 12).

    Resample the set of GT ``public_id``s **with replacement**; for each draw,
    relabel duplicated volumes with fresh unique ids in **both** frames so the
    oracle's per-``public_id`` grouping treats them as distinct volumes;
    recompute CPM. Report the point estimate (CPM on the unresampled data) and
    the percentile interval.

    Comparisons that follow this use tolerances, never ``== 0.0`` — the
    vendored code's ``EPS = 1e-8`` floors "zero" recall at ~3e-9.
    """
    _require_columns(gt_df, GT_COLUMNS, "gt_df")
    _require_columns(pred_df, PRED_COLUMNS, "pred_df")

    pids = list(pd.unique(gt_df["public_id"]))
    point = cpm(evaluate_froc(gt_df, pred_df))

    gt_by_pid = {pid: gt_df[gt_df["public_id"] == pid] for pid in pids}
    pred_by_pid = {pid: pred_df[pred_df["public_id"] == pid] for pid in pids}

    rng = np.random.default_rng(seed)
    boots = []
    n = len(pids)
    for _ in range(n_boot):
        draw = rng.choice(n, size=n, replace=True)
        gt_parts, pred_parts = [], []
        for new_id, src_idx in enumerate(draw):
            pid = pids[src_idx]
            g = gt_by_pid[pid].copy()
            g["public_id"] = new_id
            gt_parts.append(g)
            p = pred_by_pid[pid]
            if len(p) > 0:
                p = p.copy()
                p["public_id"] = new_id
                pred_parts.append(p)
        gt_boot = pd.concat(gt_parts, ignore_index=True)
        pred_boot = (
            pd.concat(pred_parts, ignore_index=True)
            if pred_parts
            else pd.DataFrame(columns=PRED_COLUMNS)
        )
        boots.append(cpm(evaluate_froc(gt_boot, pred_boot)))

    boots = np.asarray(boots, dtype=float)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return {"point": float(point), "lo": lo, "hi": hi, "boot": boots}


def _require_columns(df: pd.DataFrame, cols: Iterable[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} missing required columns {missing}; has {list(df.columns)}")
