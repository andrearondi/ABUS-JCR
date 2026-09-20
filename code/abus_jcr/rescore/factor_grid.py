"""The 2x3 factor grid the thesis reports, and the one-switch pairs read on it.

| architecture \\ objective | CE | pooled |
|---|---|---|
| Independent | B1 | B1-P |
| Joint       | B2 | A2-P |
| Joint+Geo   | A1 | FULL-P |

A condition is named by its two coordinates; the code ids on the right are the keys of
:data:`abus_jcr.rescore.variants.VARIANTS` and of every grid JSON. Each question is read on
pairs that differ in exactly ONE switch (``switches`` / ``differing_switches`` make that
checkable, and ``tests/test_factor_grid.py`` pins it):

* jointness — Joint vs Independent, objective held;
* geometry  — Joint+Geo vs Joint, objective held;
* objective — pooled vs CE, architecture held.

The per-volume rungs (A2, FULL) are not grid conditions: they are the stage-one checkpoints the
Joint pooled rungs are fine-tuned from (``_phase4_common``: ``twin = variant[:-2]``), reported
in the appendix together with the multi-switch and reference comparisons of
:data:`APPENDIX_PAIRS`. The pinned ``COMPARISONS*`` tuples of ``variants.py`` are provenance and
stay as they are.

Torch-free.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .variants import VARIANTS

__all__ = ["ARCH", "OBJ", "GRID", "REFERENCES", "DISPLAY", "CONDITIONS", "FACTOR_PAIRS",
           "OVERALL", "APPENDIX_PAIRS", "BOOT_CONDITIONS", "switches", "differing_switches",
           "pair_key", "pair_stats"]

ARCH: Tuple[str, ...] = ("Independent", "Joint", "Joint+Geo")
OBJ: Tuple[str, ...] = ("CE", "pooled")

GRID: Dict[Tuple[str, str], str] = {
    ("Independent", "CE"): "B1", ("Independent", "pooled"): "B1-P",
    ("Joint", "CE"): "B2", ("Joint", "pooled"): "A2-P",
    ("Joint+Geo", "CE"): "A1", ("Joint+Geo", "pooled"): "FULL-P",
}

#: Not conditions: nothing is trained. The rank rule is a reference line, never a test.
REFERENCES: Dict[str, str] = {"Detector": "B0", "Rank rule": "B0-rank"}

#: code id -> display name (the only place the mapping lives on the Python side).
DISPLAY: Dict[str, str] = {**{cid: f"{a}/{o}" for (a, o), cid in GRID.items()},
                           **{cid: name for name, cid in REFERENCES.items()},
                           "A2": "Joint/per-vol", "FULL": "Joint+Geo/per-vol",
                           "B0-spread": "Detector (spread)"}

#: Grid conditions in reading order (row-major).
CONDITIONS: Tuple[str, ...] = tuple(GRID[(a, o)] for a in ARCH for o in OBJ)


def _pair(factor: str, held: str, a: str, b: str) -> Dict[str, str]:
    return {"factor": factor, "held": held, "a": a, "b": b}


#: Every one-switch pair of the grid; ``a`` is the condition the hypothesis favours.
FACTOR_PAIRS: Tuple[Dict[str, str], ...] = (
    _pair("jointness", "CE", "B2", "B1"),
    _pair("jointness", "pooled", "A2-P", "B1-P"),
    _pair("geometry", "CE", "A1", "B2"),
    _pair("geometry", "pooled", "FULL-P", "A2-P"),
    _pair("objective", "Independent", "B1-P", "B1"),
    _pair("objective", "Joint", "A2-P", "B2"),
    _pair("objective", "Joint+Geo", "FULL-P", "A1"),
)

#: The complete module against no rescoring at all (descriptive; moves every switch).
OVERALL: Dict[str, str] = _pair("overall", "-", "FULL-P", "B0")

#: Reported in the appendix only, with the intervals stored by the original evaluation.
APPENDIX_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("A2", "B2"), ("FULL", "A1"), ("A2-P", "A2"), ("FULL-P", "FULL"),      # the stage-one rungs
    ("FULL", "B2"), ("FULL-P", "B2"), ("FULL-P", "B1-P"),                  # more than one switch
    ("FULL", "B0-rank"), ("FULL-P", "B0-rank"), ("B2", "B0-rank"), ("B1", "B0"),
)

#: What the bootstrap matrix has to hold to serve FACTOR_PAIRS + OVERALL. Ordered so the two
#: pooled architecture pairs (the ones no stored interval covers) complete first.
BOOT_CONDITIONS: Tuple[str, ...] = ("A2-P", "B1-P", "FULL-P", "B2", "B1", "A1", "B0")


def switches(code_id: str) -> Dict[str, object]:
    """The three switches of a trained rung, read from ``VARIANTS`` (never from its name)."""
    v = VARIANTS[code_id]
    if float(v["w_rank"]) == 0.0:
        objective = "CE"
    else:
        objective = {"froc": "pooled", "smooth_ap": "per-vol"}[v.get("rank_loss", "smooth_ap")]
    return {"jointness": v["module"] == "set", "geometry": bool(v["geometry"]),
            "objective": objective}


def differing_switches(a: str, b: str) -> List[str]:
    sa, sb = switches(a), switches(b)
    return [k for k in sa if sa[k] != sb[k]]


def pair_key(a: str, b: str) -> str:
    """The grid JSONs' comparison key (``f"{a}-{b}"``)."""
    return f"{a}-{b}"


def pair_stats(boot_a: np.ndarray, boot_b: np.ndarray, ci: float = 0.95,
               lo_a: Optional[np.ndarray] = None, hi_a: Optional[np.ndarray] = None,
               lo_b: Optional[np.ndarray] = None, hi_b: Optional[np.ndarray] = None) -> Dict:
    """Paired-bootstrap read-off from two per-draw CPM vectors that share their draws.

    ``bootstrap_cpm_ci`` and ``paired_bootstrap_delta`` resample with the same seeded draw
    sequence, and the paired statistic is ``cpm(a) - cpm(b)`` on one resample, so
    ``boot_a - boot_b`` IS ``paired_bootstrap_delta(...)["boot"]`` (pinned by
    ``tests/test_phase5_factor_comparisons.py``). Same percentile interval, same
    ``frac_positive``. With the per-draw tie bounds, the envelope is the widest interval any
    tie ordering of the evaluator's non-stable sort could produce.
    """
    d = np.asarray(boot_a, dtype=float) - np.asarray(boot_b, dtype=float)
    alpha = (1.0 - ci) / 2.0
    out = {"lo": float(np.quantile(d, alpha)), "hi": float(np.quantile(d, 1.0 - alpha)),
           "frac_positive": float((d > 0).mean()), "n_boot": int(d.size)}
    if lo_a is not None:
        d_min = np.asarray(lo_a, dtype=float) - np.asarray(hi_b, dtype=float)
        d_max = np.asarray(hi_a, dtype=float) - np.asarray(lo_b, dtype=float)
        out["envelope"] = {
            "lo": [float(np.quantile(d_min, alpha)), float(np.quantile(d_max, alpha))],
            "hi": [float(np.quantile(d_min, 1.0 - alpha)), float(np.quantile(d_max, 1.0 - alpha))],
            "frac_positive": [float((d_min > 0).mean()), float((d_max > 0).mean())]}
    return out
