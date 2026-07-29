"""[4.9] Evaluation — one seed pool at a time, through the official oracle, with CIs.

Every number flows through ``eval/froc.evaluate_froc`` on a ``PRED_COLUMNS`` CSV built by
``candidates.record.to_official_pred_csv``; GT comes from the shipped ``bbx_labels.csv`` via
``gt_labels.to_official_gt``. No mAP, no truncated FROC, no re-derivation from a summary
(Inv. 3).

**Inv. 14:** the three val seed pools are rescored and scored **separately**, then reported
mean ± std. They are never concatenated into one prediction CSV.

**Inv. 8:** :func:`assert_pool_identity` machine-checks that two rungs' pred CSVs differ
ONLY in ``probability``. :func:`b0_spread_probability` is the ``B0-spread`` control: a
strictly rank-preserving remap of B0's scores onto the full ``[0,1)`` grid, which bounds the
FROC grid-quantisation artefact on the real pool (the synthetic Check-2 measured Δ = −0.0061
— small, but non-zero and in the *unfavourable* direction, so it is measured, not assumed).

Torch-free except :func:`score_pool`, which imports torch lazily only when handed a model.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C
from ..candidates.record import to_official_pred_csv
from ..eval.froc import (bootstrap_cpm_ci, cpm, evaluate_froc, key_recall,
                         paired_bootstrap_delta, recall_ceiling)

__all__ = ["assert_pool_identity", "b0_spread_probability", "evaluate_variant",
           "score_pool", "compare_variants", "seed_summary"]

#: The pred-CSV columns that must be byte-identical across rungs (everything but the score).
_FROZEN_PRED_COLUMNS = ["public_id", "coordX", "coordY", "coordZ",
                        "x_length", "y_length", "z_length"]


def assert_pool_identity(pred_a: pd.DataFrame, pred_b: pd.DataFrame, atol: float = 0.0) -> None:
    """Inv. 8: two rungs' prediction CSVs may differ ONLY in ``probability``.

    Raises ``AssertionError`` naming the first offending column. Called by
    ``scripts/phase4_eval_grid.py`` for every rung against B0.
    """
    assert len(pred_a) == len(pred_b), (
        f"pool identity violated: row count {len(pred_a)} vs {len(pred_b)} — the rungs are "
        f"not re-ranking the same frozen pool")
    for col in _FROZEN_PRED_COLUMNS:
        a = pd.to_numeric(pred_a[col], errors="raise").to_numpy(float)
        b = pd.to_numeric(pred_b[col], errors="raise").to_numpy(float)
        bad = int(np.sum(~np.isclose(a, b, rtol=0.0, atol=atol)))
        assert bad == 0, (f"pool identity violated in column {col}: {bad}/{len(a)} rows differ. "
                          f"Rescoring may only change `probability` (Inv. 8).")


def b0_spread_probability(prob) -> np.ndarray:
    """The ``B0-spread`` control: remap scores monotonically onto the full ``[0, 1)`` grid.

    ``(rank + 0.5) / n`` over the whole seed pool with a stable ``average`` tie rule, so the
    ranking is preserved exactly (ties stay ties) and the values span the interval the
    official ``np.arange(0, 1, 0.005)`` sweep actually visits. In theory a strictly
    increasing map cannot change CPM; in practice the fixed grid quantises a compressed band
    differently, and this control bounds that artefact on the real pool.
    """
    p = np.asarray(prob, dtype=float)
    n = p.size
    if n == 0:
        return p
    r = pd.Series(p).rank(method="average").to_numpy(float) - 1.0
    return np.clip((r + 0.5) / float(n), 0.0, 1.0 - C.RESC_PROB_EPS)


def evaluate_variant(record_df: pd.DataFrame, prob, gt_df: pd.DataFrame, seed_tag: str,
                     n_boot: int = 1000, boot_seed: int = 0,
                     prob_col: str = "_prob_eval") -> Dict:
    """Score ONE seed pool: ``to_official_pred_csv -> evaluate_froc -> CPM/ceiling/CI``.

    Returns the full payload the report needs: ``cpm``, ``ceiling`` (rung-invariant by
    Inv. 8), the seven ``key_recall`` points, the whole ``fp``/``recall`` curve arrays, and
    the volume-level bootstrap CI. Never concatenates seeds (Inv. 14).

    ``n_boot <= 0`` skips the CI and returns NaN bounds. That is ONLY for diagnostics that
    carry no CI requirement — the train-pool overfitting watch (exit check 13) and the
    per-epoch selection table. **Every REPORTED CPM must carry a CI (Inv. 12)**, so the
    grid's val numbers always run with ``n_boot >= 1``.
    """
    rec = record_df.copy()
    rec[prob_col] = np.asarray(prob, dtype=float)
    pred = to_official_pred_csv(rec, prob_col)
    res = evaluate_froc(gt_df, pred)
    if int(n_boot) > 0:
        ci = bootstrap_cpm_ci(gt_df, pred, n_boot=n_boot, seed=boot_seed)
    else:
        ci = {"point": cpm(res), "lo": float("nan"), "hi": float("nan")}
    det = res["detection"]
    return {
        "seed_tag": seed_tag,
        "n_rows": int(len(rec)),
        "n_volumes": int(rec["public_id"].nunique()),
        "cpm": cpm(res),
        "ceiling": recall_ceiling(res),
        "key_recall": {str(k): float(v) for k, v in key_recall(res).items()},
        "fp": [float(x) for x in np.asarray(det["fp"]).ravel()] if "fp" in det else [],
        "recall": [float(x) for x in np.asarray(det["recall"]).ravel()] if "recall" in det else [],
        "ci": {"lo": ci["lo"], "hi": ci["hi"], "point": ci["point"]},
        "pred": pred,
    }


def compare_variants(gt_df: pd.DataFrame, pred_a: pd.DataFrame, pred_b: pd.DataFrame,
                     name_a: str, name_b: str, n_boot: int = 1000, seed: int = 0) -> Dict:
    """One pre-registered comparison, with the PAIRED interval and the pool-identity check."""
    assert_pool_identity(pred_a, pred_b)
    d = paired_bootstrap_delta(gt_df, pred_a, pred_b, n_boot=n_boot, seed=seed)
    return {"comparison": f"{name_a} - {name_b}", "delta": d["delta_point"],
            "lo": d["lo"], "hi": d["hi"], "frac_positive": d["frac_positive"],
            "n_boot": d["n_boot"]}


def seed_summary(per_seed: Sequence[Dict]) -> Dict:
    """Mean ± std over the 3 replicas (Inv. 14: reported, never pooled into one CSV)."""
    v = np.array([float(r["cpm"]) for r in per_seed], dtype=float)
    c = np.array([float(r["ceiling"]) for r in per_seed], dtype=float)
    return {"cpm_mean": float(v.mean()), "cpm_std": float(v.std(ddof=0)),
            "ceiling_mean": float(c.mean()), "ceiling_std": float(c.std(ddof=0)),
            "per_seed_cpm": [float(x) for x in v], "n_seeds": int(v.size)}


def score_pool(model, feats: np.ndarray, coord: np.ndarray, length: np.ndarray,
               index_lists: Sequence[np.ndarray], n_rows: int, device: str = "cpu",
               max_set_size: Optional[int] = None, batch_sets: Optional[int] = None
               ) -> np.ndarray:
    """Run the rescorer over every SET and return a ``probability`` per RECORD ROW.

    Sets are scored independently (attention never crosses sets); the per-set outputs are
    scattered back to record row order, so the result joins the record — and hence the pred
    CSV — by construction. Un-augmented, no TTA (Inv. 13).
    """
    import torch

    from .datasets import collate_sets

    max_set_size = int(C.RESC_MAX_SET_SIZE if max_set_size is None else max_set_size)
    batch_sets = int(C.RESC_SET_BATCH_SETS if batch_sets is None else batch_sets)
    labels = np.zeros(len(feats), dtype=np.float32)          # unused at inference
    out = np.full(int(n_rows), np.nan, dtype=float)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(index_lists), batch_sets):
            chunk = list(index_lists[start:start + batch_sets])
            batch = collate_sets(chunk, feats, labels, coord, length, max_set_size=max_set_size)
            t = {k: torch.as_tensor(v).to(device) for k, v in batch.items()
                 if k in ("feats", "coord", "length", "mask")}
            logits = model(t["feats"].float(), t["coord"].float(), t["length"].float(), t["mask"])
            p = torch.sigmoid(logits).clamp(0.0, 1.0 - C.RESC_PROB_EPS).cpu().numpy()
            for i, idx in enumerate(chunk):
                out[idx] = p[i, :len(idx)]

    if np.isnan(out).any():
        missing = int(np.isnan(out).sum())
        raise RuntimeError(f"{missing} record rows were never scored — the set grouping does "
                           f"not cover the pool (Inv. 8 requires every row to be scored)")
    return out
