"""[4.3] The per-candidate token — five ablatable blocks (§1.2).

| block | dims | contents |
|---|---|---|
| ``appearance``  | 128 | the frozen encoder embedding ``a_i`` |
| ``abs_geom``    | 6   | ``cen_d*/S*`` (S = ``meta["iso_shape"]``), ``log1p(ext_d*)`` |
| ``score_stats`` | 7   | ``SCORE_STAT_COLUMNS`` verbatim, with ``slice_count``/``z_span`` ``log1p``-transformed |
| ``tube_geom``   | 2   | ``TUBE_GEOM_COLUMNS`` = ``centroid_jitter``, ``area_cv`` |
| ``rank``        | 17  | ``rank_norm`` + a 16-D sinusoidal embedding of the integer ``rank`` |

**De-duplication (resolves the double-listing in PHASE_4.md §1.2):** ``z_span`` and
``fill_ratio`` live ONLY in ``score_stats`` — they are frozen ``SCORE_STAT_COLUMNS``
members — so ``abs_geom`` carries centroid + log-size only. Recorded here so the §4.7 block
ablation stays clean.

**The anisotropy feature was REMOVED 2026-08-09** (``abs_geom`` 7 → 6 dims). PHASE_4.md §1.2
specifies it as *"anisotropy = depth-extent / mean-lateral-extent — the Phase-0b signal"*, and
every part of that description turned out to be wrong or dead:

* **Wrong axis.** It was computed as ``ext_d0 / mean(ext_d1, ext_d2)``, and ``d0`` is the
  MEASURED **lateral** axis (``AXIS_CHECK.md``, 129/130 volumes, four independent lines), not
  depth/beam as ``conventions`` still declares. It measured lateral elongation throughout.
* **Uninterpretable units.** In the deployed cache one voxel spans 1.0959 × 0.1460 × 0.4000 mm
  (Inv. 6 amendment, ≈7.5× in-plane aspect error), so a physically CUBIC candidate reads
  **0.195** on this scale, not 1.0. The number has no physical reading.
* **Dead signal.** It is the weakest of all 12 pool features: val Cliff's δ **0.097**, balanced
  accuracy **0.543** against a chance floor of 0.5 (train δ 0.078) — noise, on the promoted
  pool and the archived one alike.
* **Its motivating hypothesis is a closed negative result.** The Phase-0b shadow-geometry claim
  was tested three ways and failed each: ``structure_present = false`` on both pools, the
  per-candidate δ is **−0.097** (small *and* wrong-signed), and ``[I.6b]`` measured elongation
  about the *true* beam axis to find **zero** candidates above 2× beam elongation in **8/8**
  detectors across both splits. There is no ray-shaped population to detect.
* **Near-redundant with its own block.** Measured on log-uniform extents over the pool's range:
  a linear readout from the three ``log1p(ext_d*)`` beside it recovers the raw ratio at only
  ``R² = 0.415``, but ``log(ratio)`` at ``R² = 0.955`` and a single nonlinearity — which the
  token projection applies anyway — at ``R² = 0.945``. So ``[I.6b]``'s "already *linearly*
  recoverable" was imprecise (it holds for a *geometric*-mean ratio, not the arithmetic-mean
  one shipped, and ``log1p ≠ log`` by up to 0.41 at small extents), but ~95 % recoverable is
  the operative fact, and the residual 5 % is an arithmetic-vs-geometric-mean artefact rather
  than a quantity anything predicts.

**Nothing recorded depends on it:** ``anisotropy`` is *not* one of the 31 frozen
``CANDIDATE_COLUMNS`` — it is derived, and it survives in ``probe.pool_diag``,
``probe.fp_structure`` and ``probe.anisotropy``, which are where every recorded number lives.
Removing it changes the **model**, not the diagnostics.

**No replacement was added.** The true-beam-axis version separates TP/FP better (``|δ| ≈ 0.27``,
7/8 detectors) but ``[I.6b]`` ruled explicitly that *"it licenses no new feature"*, and choosing
a feature by its measured val effect size would add a selection surface for no mechanism.

**``detector_of_origin`` is NOT a feature** (Inv. 7): it is the set key and is constant
within a run. **Standardisation** stats are computed on the TRAIN pool only, persisted in
the checkpoint, and applied unchanged to val (and, in Phase 5, to test) — no val statistic
ever enters them.

A disabled block is **omitted**, not zeroed; the input projection is re-sized accordingly.
The feature-matrix core is torch-free (unit-tested on the laptop); only
:class:`TokenProjection` needs torch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C

__all__ = ["BLOCK_DIMS", "rank_sinusoid", "build_feature_matrix",
           "fit_standardiser", "apply_standardiser", "TokenProjection"]

#: Width contributed by each block. ``sum(BLOCK_DIMS.values())`` is the full token width.
BLOCK_DIMS: Dict[str, int] = {
    "appearance": int(C.RESC_D_APP),
    "abs_geom": 6,          # 7 before 2026-08-09; the anisotropy dim was removed (docstring)
    "score_stats": len(C.SCORE_STAT_COLUMNS),
    "tube_geom": len(C.TUBE_GEOM_COLUMNS),
    "rank": 1 + int(C.RESC_RANK_PE_DIM),
}

#: Score-stat columns that enter the token on a log1p scale (heavy-tailed counts).
_LOG1P_SCORE_STATS = ("slice_count", "z_span")


def rank_sinusoid(rank, dim: Optional[int] = None) -> np.ndarray:
    """Vaswani sinusoidal embedding of the INTEGER within-set rank -> ``(n, dim)``.

    Bounded in ``[-1, 1]``, so it needs no standardisation of its own, and distinguishes
    adjacent ranks (rank 1 vs 2 differ) while degrading smoothly for the deep tail.
    """
    dim = int(C.RESC_RANK_PE_DIM if dim is None else dim)
    r = np.asarray(rank, dtype=float).reshape(-1, 1)
    i = np.arange(dim // 2, dtype=float)
    freq = np.power(10000.0, -2.0 * i / float(dim))
    ang = r * freq
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def _block_abs_geom(df: pd.DataFrame, iso_shape_by_pid: Dict[int, Sequence[int]]):
    if iso_shape_by_pid is None:
        raise KeyError("abs_geom needs iso_shape_by_pid (from cache.read_meta(...)['iso_shape'])")
    shp = np.array([[float(s) for s in iso_shape_by_pid[int(p)]] for p in df["public_id"]], dtype=float)
    cen = df[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float)
    ext = df[["ext_d0", "ext_d1", "ext_d2"]].to_numpy(float)
    # NO extent ratio here. The `anisotropy` dim (ext_d0 / mean(ext_d1, ext_d2)) was removed
    # on 2026-08-09: wrong axis, uninterpretable units, weakest of all 12 pool features, its
    # motivating hypothesis a closed negative result, and ~95% recoverable from the three
    # log1p(ext) below it through one nonlinearity. Full evidence in the module docstring.
    cols = ([cen[:, a] / shp[:, a] for a in range(3)]
            + [np.log1p(ext[:, a]) for a in range(3)])
    names = ["cen_d0_norm", "cen_d1_norm", "cen_d2_norm",
             "log1p_ext_d0", "log1p_ext_d1", "log1p_ext_d2"]
    return np.stack(cols, axis=1), names


def _block_score_stats(df: pd.DataFrame):
    cols, names = [], []
    for c in C.SCORE_STAT_COLUMNS:
        v = df[c].to_numpy(float)
        if c in _LOG1P_SCORE_STATS:
            cols.append(np.log1p(v))
            names.append(f"log1p_{c}")
        else:
            cols.append(v)
            names.append(c)
    return np.stack(cols, axis=1), names


def build_feature_matrix(record_df: pd.DataFrame, emb: Optional[np.ndarray] = None,
                         use_blocks: Sequence[str] = C.RESC_TOKEN_BLOCKS,
                         iso_shape_by_pid: Optional[Dict[int, Sequence[int]]] = None
                         ) -> Tuple[np.ndarray, List[str]]:
    """Assemble the ``(n_rows, sum(dims of enabled blocks))`` raw feature matrix.

    Column names are ``"<block>:<field>"`` so a block ablation can be verified column-wise.
    Row order is the record's row order — the same construction-preserved join the crop
    cache and the official pred CSV use.
    """
    use_blocks = tuple(use_blocks)
    unknown = [b for b in use_blocks if b not in BLOCK_DIMS]
    if unknown:
        raise ValueError(f"unknown token block(s) {unknown}; known: {list(BLOCK_DIMS)}")

    parts: List[np.ndarray] = []
    names: List[str] = []
    for block in C.RESC_TOKEN_BLOCKS:                       # canonical order, always
        if block not in use_blocks:
            continue
        if block == "appearance":
            if emb is None:
                raise ValueError("the 'appearance' block needs `emb`; pass the cached a_i "
                                 "or drop the block from use_blocks")
            a = np.asarray(emb, dtype=float)
            if a.shape[0] != len(record_df):
                raise ValueError(f"emb has {a.shape[0]} rows, record has {len(record_df)}")
            parts.append(a)
            names += [f"appearance:a{j}" for j in range(a.shape[1])]
        elif block == "abs_geom":
            x, nm = _block_abs_geom(record_df, iso_shape_by_pid)
            parts.append(x); names += [f"abs_geom:{n}" for n in nm]
        elif block == "score_stats":
            x, nm = _block_score_stats(record_df)
            parts.append(x); names += [f"score_stats:{n}" for n in nm]
        elif block == "tube_geom":
            x = record_df[list(C.TUBE_GEOM_COLUMNS)].to_numpy(float)
            parts.append(x); names += [f"tube_geom:{n}" for n in C.TUBE_GEOM_COLUMNS]
        elif block == "rank":
            pe = rank_sinusoid(record_df["rank"].to_numpy())
            parts.append(np.concatenate([record_df["rank_norm"].to_numpy(float)[:, None], pe], axis=1))
            names += ["rank:rank_norm"] + [f"rank:pe{j}" for j in range(pe.shape[1])]

    if not parts:
        return np.zeros((len(record_df), 0), dtype=float), []
    return np.concatenate(parts, axis=1).astype(np.float64), names


# ----------------------------------------------------------------------------- standardisation
def fit_standardiser(X: np.ndarray, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    """Per-feature mean/std over the **train pool only**. ``std`` is floored at ``eps`` so a
    constant column cannot produce NaN/inf."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return {"mean": mean, "std": std}


def apply_standardiser(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """``(X - mean) / std`` with the persisted TRAIN statistics — never re-fitted on val."""
    mean = np.asarray(stats["mean"], dtype=float)
    std = np.asarray(stats["std"], dtype=float)
    X = np.asarray(X, dtype=float)
    if X.shape[1] != mean.shape[0]:
        raise ValueError(f"standardiser fitted on {mean.shape[0]} features, got {X.shape[1]}; "
                         "the token blocks must match the ones the checkpoint was trained with")
    return (X - mean) / std


# ----------------------------------------------------------------------------- torch projection
try:
    from torch import nn

    _TORCH_OK = True
except ImportError:  # pragma: no cover - laptop path
    _TORCH_OK = False


if _TORCH_OK:

    class TokenProjection(nn.Module):
        """``Linear(d_in -> H) + LayerNorm + GELU`` — §4.3's ``TokenBuilder.project``.

        ``d_in`` is the width of the ENABLED blocks, so a block ablation genuinely changes
        the projection's input size (the block is omitted, not zeroed).
        """

        def __init__(self, d_in: int, d_model: int):
            super().__init__()
            self.d_in = int(d_in)
            self.d_model = int(d_model)
            self.proj = nn.Linear(self.d_in, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.norm(self.proj(x)))

else:  # pragma: no cover - laptop path

    class TokenProjection:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("TokenProjection needs torch; run it in the server env.")
