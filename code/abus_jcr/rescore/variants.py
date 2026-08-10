"""[4.7] The ablation ladder, the trial budget, and the §4.6 fairness contract.

| id | attention | geometry bias | ``w_rank`` | λ | isolates |
|---|---|---|---|---|---|
| **B0**   | —                        | — | — | — | the Phase-3 floor: pool ranked by ``score_max`` (MEASURED at [4.7], never a constant — 0.6327 ± 0.0526 on the promoted pool, [F.8]) |
| **B1**   | none (per-candidate MLP) | — | 0 | 1 | does *any* rescoring help |
| **B2**   | SAB, appearance-only     | ✗ | 0 | 1 | **jointness** (B2 vs B1) |
| **A1**   | SAB                      | ✓ | 0 | 1 | **geometry** (A1 vs B2) |
| **A2**   | SAB, appearance-only     | ✗ | 1 | swept | **loss** (A2 vs B2) |
| **FULL** | SAB                      | ✓ | 1 | swept | **the contribution** (FULL vs B2) |

"Appearance-only attention" means no pairwise ``g(m,n)`` bias; the *tokens* still carry
absolute geometry in every rung — that is what makes A1-vs-B2 isolate the *pairwise* term.
Every ranking rung carries the BCE term at a swept λ, so A2/FULL vs B2 compares
``ranking + BCE`` against ``BCE alone`` and the calibration confound is removed.

**Recorded spec reconciliation (A1's trial budget).** §4.7's table cell reads as a single
A1 trial at B2's α/lr, while §4.6's fairness contract requires **exactly 4** trials per rung
and exit check 5 machine-asserts it. The contract wins: A1 receives the same
``RESC_CE_SEARCH``. Because B2's selected ``(α, lr)`` is by construction one of those four,
the clean "A1 at B2's α/lr" isolation is still available and is what
``scripts/phase4_eval_grid.py`` uses for the PRIMARY A1−B2 delta; A1's own best-of-4 is
reported alongside. Flagged in RESULTS_PHASE_4 for the user to veto.

Torch-free.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .. import conventions as C

__all__ = ["LADDER", "VARIANTS", "SUB_ABLATION_BLOCKS", "trials_for", "b1_param_count",
           "match_b1_capacity", "fairness_table", "assert_fairness", "select_epoch_by_val_cpm"]

#: The six pre-registered rungs. B0 is the recorded ``score_max`` ranking, not a model.
LADDER = ("B0", "B1", "B2", "A1", "A2", "FULL")

#: The trained rungs. ``search`` picks which 4-trial grid the rung sweeps.
VARIANTS: Dict[str, Dict] = {
    "B1":   {"module": "mlp", "geometry": False, "w_rank": 0.0, "search": "ce",
             "isolates": "does any rescoring help (vs B0)"},
    "B2":   {"module": "set", "geometry": False, "w_rank": 0.0, "search": "ce",
             "isolates": "jointness (vs B1)"},
    "A1":   {"module": "set", "geometry": True,  "w_rank": 0.0, "search": "ce",
             "isolates": "pairwise geometry (vs B2)"},
    "A2":   {"module": "set", "geometry": False, "w_rank": 1.0, "search": "lambda",
             "isolates": "loss (vs B2)"},
    "FULL": {"module": "set", "geometry": True,  "w_rank": 1.0, "search": "lambda",
             "isolates": "the contribution (vs B2)"},
}

#: Blocks toggled on/off on FULL at its selected λ, 3 seeds each (§4.7 sub-ablations).
SUB_ABLATION_BLOCKS = ("rank", "score_stats", "tube_geom")

#: The pre-registered comparison list (primary first).
COMPARISONS = (("FULL", "B2"), ("A1", "B2"), ("A2", "B2"), ("B2", "B1"), ("B1", "B0"),
               ("FULL", "A1"))


def trials_for(variant: str, b2_choice: Optional[Dict] = None) -> List[Dict]:
    """The **exactly 4** hyperparameter trials this rung is allowed (§4.6).

    CE rungs (B1/B2/A1) sweep ``RESC_CE_SEARCH`` (α × lr) at ``w_rank = 0``, ``λ = 1``.
    Ranking rungs (A2/FULL) sweep ``RESC_LAMBDA_SEARCH`` at ``w_rank = 1`` with α/lr fixed
    to B2's selected values, so the loss term is the only thing that moves.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    if spec["search"] == "ce":
        return [{"alpha": float(t["alpha"]), "lr": float(t["lr"]),
                 "w_rank": 0.0, "lam": 1.0} for t in C.RESC_CE_SEARCH]
    if b2_choice is None:
        raise ValueError(f"{variant} sweeps lambda at B2's alpha/lr; pass b2_choice="
                         "{'alpha': ..., 'lr': ...} from [4.5]/[4.6]")
    return [{"alpha": float(b2_choice["alpha"]), "lr": float(b2_choice["lr"]),
             "w_rank": 1.0, "lam": float(lam)} for lam in C.RESC_LAMBDA_SEARCH]


# ----------------------------------------------------------------------------- fairness
def b1_param_count(d_in: int, d_model: int, hidden: int, depth: int = 2) -> int:
    """Analytic parameter count of :class:`setmodel.B1Rescorer` (torch-free, so the match
    can be computed and unit-tested without a GPU).

    ``TokenProjection`` = ``Linear(d_in -> d_model)`` + ``LayerNorm(d_model)``;
    then ``depth`` hidden layers and a 1-unit head.
    """
    d_in, d_model, hidden, depth = int(d_in), int(d_model), int(hidden), int(depth)
    total = d_in * d_model + d_model + 2 * d_model
    prev = d_model
    for _ in range(depth):
        total += prev * hidden + hidden
        prev = hidden
    return int(total + prev + 1)


def match_b1_capacity(d_in: int, d_model: int, target_params: int, depth: int = 2,
                      max_hidden: int = 8192) -> int:
    """Smallest-error ``hidden`` so ``b1_param_count`` lands closest to ``target_params``.

    ``target_params`` is the measured parameter count of the SELECTED set module — the
    fairness contract's reference. Raises if no width gets within ±10 %, because silently
    shipping a handicapped baseline is exactly the drift the checklist forbids.
    """
    widths = np.arange(1, int(max_hidden) + 1)
    counts = np.array([b1_param_count(d_in, d_model, int(h), depth) for h in widths])
    best = int(widths[int(np.argmin(np.abs(counts - int(target_params))))])
    got = b1_param_count(d_in, d_model, best, depth)
    rel = abs(got - target_params) / max(1.0, float(target_params))
    if rel > 0.10:
        raise ValueError(f"no B1 hidden width within +/-10% of {target_params} "
                         f"(best {best} -> {got}, off by {rel:.1%}); widen max_hidden or "
                         f"revisit the capacity grid")
    return best


def fairness_table(params: Dict[str, int], epochs: Dict[str, int], trials: Dict[str, int],
                   reference: str = "B2") -> Dict:
    """Assemble the ``fairness.json`` payload (§4.6, exit check 5).

    ``params[reference]`` is the shared set-module capacity. A1/FULL legitimately carry the
    extra ``GeometryBias`` projection — that IS the treatment — so the ±10 % tolerance is
    applied to the **baseline** B1 only, which is what the do-not-drift rule protects.
    """
    ref = int(params[reference])
    return {
        "reference": reference,
        "reference_params": ref,
        "params": {k: int(v) for k, v in params.items()},
        "epochs": {k: int(v) for k, v in epochs.items()},
        "trials": {k: int(v) for k, v in trials.items()},
        "b1_rel_error": abs(int(params["B1"]) - ref) / max(1.0, float(ref)),
        "tolerance": 0.10,
        "expected_trials": 4,
    }


def assert_fairness(table: Dict, tol: float = 0.10) -> None:
    """Exit check 5: identical epochs, identical trial budget, B1 within ±``tol`` of the
    reference set module. Raises ``AssertionError`` naming the offender."""
    rel = float(table["b1_rel_error"])
    assert rel <= tol, (f"B1 is handicapped: params {table['params']['B1']} vs reference "
                        f"{table['reference_params']} ({rel:.1%} off, tolerance {tol:.0%})")
    ep = table["epochs"]
    assert len(set(ep.values())) == 1, f"unequal epoch budgets across rungs: {ep}"
    tr = table["trials"]
    assert len(set(tr.values())) == 1, f"unequal trial budgets across rungs: {tr}"
    exp = int(table.get("expected_trials", 4))
    assert set(tr.values()) == {exp}, f"each rung must get exactly {exp} trials, got {tr}"


# ----------------------------------------------------------------------------- selection
def select_epoch_by_val_cpm(epochs: Iterable, val_cpm: Iterable,
                            tol: Optional[float] = None) -> int:
    """The deployed epoch: **earliest** epoch whose val CPM is within ``tol`` of the max.

    Never val loss (``RESC_SELECT_METRIC == "val_cpm"``, exit check 8). The tolerance
    exists because CPM moves in ~1/30 steps on 30 val lesions, so a bare argmax selects
    noise — the Inv.-2/A1 lesson. The Inv.-2 ceiling tie-break does NOT apply here: by
    Inv. 8 the recall ceiling is identical for every epoch and every rung.
    """
    tol = C.RESC_SELECT_CPM_TOL if tol is None else float(tol)
    ep = np.asarray(list(epochs))
    cp = np.asarray(list(val_cpm), dtype=float)
    if ep.shape != cp.shape or ep.size == 0:
        raise ValueError(f"epochs/val_cpm must be equal-length and non-empty, got {ep.shape} vs {cp.shape}")
    best = float(np.nanmax(cp))
    tied = ep[cp >= best - tol]
    return int(np.min(tied))
