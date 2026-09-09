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
    "write_pred_csv", "bootstrap_cpm_ci", "paired_bootstrap_delta",
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


# ----------------------------------------------------------------- parallel draw workers
# Added 2026-09-09 (Phase 5). A reportable draw costs ~12 s measured — two CSV writes plus
# the vendored oracle re-reading them — so an n=1000 unit is hours and the full test grid is
# days on one core. Draws are independent GIVEN their resample indices, and the indices are
# pre-generated from the SAME seeded RNG in the same order, so evaluating them in a worker
# pool and reassembling in submission order returns arrays identical to the sequential loop
# (pinned by tests/test_parallel_bootstrap.py; every draw's evaluate_froc runs in its own
# TemporaryDirectory, so there is no shared file to collide on). ``workers=1`` — the default
# at every call site — takes the sequential path.
_BOOT_STATE: dict = {}


def _init_boot_worker(state: dict) -> None:
    _BOOT_STATE.clear()
    _BOOT_STATE.update(state)


def _relabelled_gt(draw) -> pd.DataFrame:
    pids, gt_by_pid = _BOOT_STATE["pids"], _BOOT_STATE["gt"]
    parts = []
    for new_id, src_idx in enumerate(draw):
        g = gt_by_pid[pids[src_idx]].copy()
        g["public_id"] = new_id
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _relabelled_pred(draw, key: str) -> pd.DataFrame:
    pids, by_pid = _BOOT_STATE["pids"], _BOOT_STATE[key]
    parts = []
    for new_id, src_idx in enumerate(draw):
        src = by_pid[pids[src_idx]]
        if len(src) > 0:
            p = src.copy()
            p["public_id"] = new_id
            parts.append(p)
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=PRED_COLUMNS))


def _marginal_draw(draw) -> float:
    return cpm(evaluate_froc(_relabelled_gt(draw), _relabelled_pred(draw, "pred")))


def _paired_draw(draw) -> float:
    gt_boot = _relabelled_gt(draw)
    return (cpm(evaluate_froc(gt_boot, _relabelled_pred(draw, "a")))
            - cpm(evaluate_froc(gt_boot, _relabelled_pred(draw, "b"))))


def _run_draws(worker, draws, state: dict, workers: int) -> list:
    if int(workers) <= 1:
        _init_boot_worker(state)
        return [worker(d) for d in draws]
    import multiprocessing as mp

    # fork where available (Linux default; macOS defaults to spawn, which re-imports the
    # world per worker and cannot handle a heredoc __main__ at all — measured 2026-09-09).
    # The workers touch only numpy/pandas + per-call TemporaryDirectories, so fork is safe.
    method = "fork" if "fork" in mp.get_all_start_methods() else None
    with mp.get_context(method).Pool(processes=int(workers), initializer=_init_boot_worker,
                                     initargs=(state,)) as pool:
        return pool.map(worker, draws,
                        chunksize=max(1, len(draws) // (int(workers) * 8)))


def bootstrap_cpm_ci(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
    workers: int = 1,
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
    if int(n_boot) < 1:
        # Falling through would reach np.quantile on an empty array and raise
        # `IndexError: index -1 is out of bounds` from inside numpy, forty frames from the
        # caller. Producing an interval is this function's entire job, so zero draws is a
        # caller bug; `rescore.evaluate.evaluate_variant` is the layer that legitimately
        # means "no CI" and returns NaN bounds for n_boot <= 0.
        raise ValueError(
            f"bootstrap_cpm_ci needs n_boot >= 1, got {n_boot}. For a deliberately "
            f"CI-free reading use evaluate_variant(..., n_boot=0), which returns NaN bounds.")

    pids = list(pd.unique(gt_df["public_id"]))
    point = cpm(evaluate_froc(gt_df, pred_df))

    gt_by_pid = {pid: gt_df[gt_df["public_id"] == pid] for pid in pids}
    pred_by_pid = {pid: pred_df[pred_df["public_id"] == pid] for pid in pids}

    # The RNG serves ONLY these choice calls, so pre-generating the whole draw sequence is
    # byte-identical to drawing inside the loop — and it is what lets the draws parallelise.
    rng = np.random.default_rng(seed)
    n = len(pids)
    draws = [rng.choice(n, size=n, replace=True) for _ in range(int(n_boot))]
    boots = _run_draws(_marginal_draw, draws,
                       {"pids": pids, "gt": gt_by_pid, "pred": pred_by_pid}, workers)

    boots = np.asarray(boots, dtype=float)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return {"point": float(point), "lo": lo, "hi": hi, "boot": boots}


def paired_bootstrap_delta(
    gt_df: pd.DataFrame,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
    workers: int = 1,
) -> dict:
    """PAIRED volume-level bootstrap CI on ``CPM(a) - CPM(b)`` (Inv. 12; Phase 4 §4.9).

    Resample the set of GT ``public_id``s **once per draw** and score **both** conditions on
    that same resample, exactly mirroring :func:`bootstrap_cpm_ci`'s relabelling so the
    oracle treats duplicated volumes as distinct. Because Phase-4/5 conditions re-rank the
    **identical** pool (Inv. 8) — same rows, same boxes, only ``probability`` differs — the
    shared volume-level variance cancels inside each draw and the paired interval is far
    tighter than the difference of two marginal intervals.

    Returns ``{delta_point, lo, hi, frac_positive, n_boot, boot}`` where ``delta_point`` is
    the difference on the UNRESAMPLED data and ``frac_positive`` is the fraction of draws
    favouring ``a`` — the number the report quotes next to every comparison.
    """
    _require_columns(gt_df, GT_COLUMNS, "gt_df")
    _require_columns(pred_a, PRED_COLUMNS, "pred_a")
    _require_columns(pred_b, PRED_COLUMNS, "pred_b")

    pids = list(pd.unique(gt_df["public_id"]))
    delta_point = cpm(evaluate_froc(gt_df, pred_a)) - cpm(evaluate_froc(gt_df, pred_b))

    gt_by_pid = {pid: gt_df[gt_df["public_id"] == pid] for pid in pids}
    a_by_pid = {pid: pred_a[pred_a["public_id"] == pid] for pid in pids}
    b_by_pid = {pid: pred_b[pred_b["public_id"] == pid] for pid in pids}

    # Same pre-generation argument as bootstrap_cpm_ci: the RNG serves only the choice calls.
    rng = np.random.default_rng(seed)
    n = len(pids)
    draws = [rng.choice(n, size=n, replace=True) for _ in range(int(n_boot))]
    boots = _run_draws(_paired_draw, draws,
                       {"pids": pids, "gt": gt_by_pid, "a": a_by_pid, "b": b_by_pid}, workers)

    boots = np.asarray(boots, dtype=float)
    alpha = (1.0 - ci) / 2.0
    return {
        "delta_point": float(delta_point),
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1.0 - alpha)),
        "frac_positive": float((boots > 0).mean()),
        "n_boot": int(n_boot),
        "boot": boots,
    }


def _require_columns(df: pd.DataFrame, cols: Iterable[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} missing required columns {missing}; has {list(df.columns)}")
