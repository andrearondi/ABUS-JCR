"""Post-hoc checkpoint selection on the true linked 3D val CPM (Inv. 2 A1, P3-UPDATE D3).

The training loop (``train.py``) saves EVERY epoch and selects nothing. This module holds
the torch-free selection rule so it is unit-tested on the laptop: given each converged
epoch's linked val CPM, pick the deployed checkpoint. The heavy detect->link->oracle CPM
computation lives in ``scripts/phase2_select_checkpoint.py``; the *decision* lives here.

Selection rule (A1, revised 2026-07-19). The pre-P3-UPDATE ``min_epoch=15`` floor assumed
the old SGD-lr-0.01 dynamics where early epochs were high-LR noise. The D4 recipe (AdamW 1e-4
+ frozen BN) inverts that: it converges by ~epoch 6 and then OVERFITS (val-loss U-shape), so
the annealed *late* epochs are the overfit tail. The revised rule:

  1. Only epochs ``>= min_epoch`` (a small floor, e.g. 3, that skips the pre-convergence
     epochs 0-2 — NOT the "annealed half") are selection-eligible.
  2. On a 30-lesion val set, CPM differences of a few 0.001 are noise (CPM moves in ~1/30
     steps). So among epochs whose CPM is within ``cpm_tol`` of the max, tie-break on the
     **highest recall ceiling** (the Inv.-8 hard cap — the rescorer can re-rank the pool but
     can never recover a lesion that isn't in it), then the **earliest** epoch (least overfit).

This reproduces the seed0 evidence, where the bare CPM-argmax picked epoch 10 (CPM 0.4798,
ceiling 0.700) over epoch 6 (0.4779, ceiling 0.833) by a 0.002 noise margin while epoch 6 was
the peak of val-loss/val-AP/cpm-proxy AND held 4 more lesions of ceiling.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def select_epoch(epoch_cpms: Dict[int, float], min_epoch: int,
                 epoch_ceilings: Optional[Dict[int, float]] = None,
                 cpm_tol: float = 0.0) -> int:
    """Select the deployed epoch among ``epoch >= min_epoch`` (A1 revised).

    ``epoch_cpms`` maps epoch -> linked val CPM. If ``epoch_ceilings`` is given and
    ``cpm_tol > 0``: among epochs whose CPM is within ``cpm_tol`` of the max, pick the
    **highest recall ceiling**, tie-broken toward the **earliest** epoch. Otherwise: max CPM,
    tie-broken toward the later epoch (legacy behaviour). Raises if no epoch qualifies. NaN
    CPMs are ignored; all-NaN -> the latest qualifying epoch (defined fallback).
    """
    cands = {e: c for e, c in epoch_cpms.items() if int(e) >= int(min_epoch)}
    if not cands:
        raise ValueError(
            f"no saved epoch >= min_epoch={min_epoch} (have epochs {sorted(epoch_cpms)}); "
            "train at least min_epoch+1 epochs")
    finite = {e: c for e, c in cands.items() if c is not None and not math.isnan(float(c))}
    if not finite:
        return max(cands)  # all-NaN fallback: the latest epoch

    max_cpm = max(finite.values())
    if epoch_ceilings is not None and cpm_tol > 0.0:
        def _ceil(e):
            v = epoch_ceilings.get(e)
            return float(v) if (v is not None and not math.isnan(float(v))) else float("-inf")
        band = [e for e, c in finite.items() if float(c) >= max_cpm - float(cpm_tol)]
        # highest ceiling, then earliest epoch (least overfit)
        best_e = min(band, key=lambda e: (-_ceil(e), int(e)))
        return int(best_e)

    # legacy: max CPM, tie-broken toward the later epoch
    best_e, _ = max(finite.items(), key=lambda kv: (float(kv[1]), int(kv[0])))
    return int(best_e)


def select_epoch_guarded(epoch_cpms: Dict[int, float], epoch_ceilings: Dict[int, float],
                         min_epoch: int, coverage_floor: float, cpm_guard: float,
                         epoch_pools: Optional[Dict[int, float]] = None,
                         pool_budget: Optional[float] = None, cpm_tol: float = 0.0) -> int:
    """[CONTINGENCY policy — NOT the deployed rule] coverage-floor selection with a CPM guard.

    Rationale (P3U3, 2026-07-29). For the **fold** detectors the deployed CPM-first rule optimises a
    number that is never reported (fold CPM appears in no result) while ignoring **coverage** — the
    fraction of training sets that contain a positive — which directly gates the rescorer's training
    signal (Phase-4 §2: a set with no positive contributes NOTHING to the listwise ranking loss). This
    rule instead requires a coverage floor, then maximises CPM among the qualifying epochs:

        ceiling >= coverage_floor  AND  cpm >= max_cpm - cpm_guard  AND  pool <= pool_budget
        -> argmax CPM (ties -> earliest epoch);  otherwise -> :func:`select_epoch` (deployed rule).

    The ``cpm_guard`` blocks *degenerate* early checkpoints that clear the coverage floor only because
    they are barely converged (observed: fold4's epoch 5 at CPM 0.14 vs its 0.41 max). The fallback
    means this rule can never do worse than the deployed pick on a run it cannot improve.

    Simulated on the recorded per-epoch tables (P3U2.SIM, floor 0.80 / guard 0.15): fold coverage
    0.697 -> 0.807 at a fold-CPM cost of 0.042, with **all three seed detectors unchanged** — i.e. the
    evaluation pool, the operating point and B0' are untouched. Adopting it as the DEPLOYED rule would
    require an Inv. 2 / A1 amendment; using it to build a side-by-side contingency training pool does not.
    """
    cands = {e: c for e, c in epoch_cpms.items() if int(e) >= int(min_epoch)}
    finite = {e: float(c) for e, c in cands.items()
              if c is not None and not math.isnan(float(c))}
    if finite:
        max_cpm = max(finite.values())

        def _ok(e: int) -> bool:
            ceil = epoch_ceilings.get(e)
            if ceil is None or math.isnan(float(ceil)) or float(ceil) < float(coverage_floor):
                return False
            if finite[e] < max_cpm - float(cpm_guard):
                return False
            if pool_budget is not None and epoch_pools is not None:
                p = epoch_pools.get(e)
                if p is not None and not math.isnan(float(p)) and float(p) > float(pool_budget):
                    return False
            return True

        ok = [e for e in finite if _ok(e)]
        if ok:                                   # highest CPM among the qualifying epochs, earliest on ties
            return int(min(ok, key=lambda e: (-finite[e], int(e))))
    return select_epoch(epoch_cpms, min_epoch, epoch_ceilings, cpm_tol)   # documented fallback


def selection_stability(epoch_cpms: Dict[int, float], min_epoch: int, top_k: int = 3) -> Tuple[float, list]:
    """Spread of the top-``top_k`` qualifying CPMs — a low-resolution flag for the report.

    Returns ``(spread, top_rows)`` where ``spread = max - min`` over the top-k CPMs and
    ``top_rows`` is the list of ``(epoch, cpm)`` sorted by CPM desc. A wide spread across
    near-tied epochs means the 30-lesion val CPM barely resolves them — worth flagging.
    """
    import math

    cands = [(int(e), float(c)) for e, c in epoch_cpms.items()
             if int(e) >= int(min_epoch) and c is not None and not math.isnan(float(c))]
    cands.sort(key=lambda ec: (ec[1], ec[0]), reverse=True)
    top = cands[:top_k]
    if not top:
        return float("nan"), []
    spread = top[0][1] - top[-1][1]
    return float(spread), top
