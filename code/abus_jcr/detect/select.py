"""Post-hoc checkpoint selection on the true linked 3D val CPM (Inv. 2 A1, P3-UPDATE D3).

The training loop (``train.py``) saves EVERY epoch and selects nothing. This module holds
the torch-free selection rule so it is unit-tested on the laptop: given each converged
epoch's linked val CPM, pick the deployed checkpoint. The heavy detect->link->oracle CPM
computation lives in ``scripts/phase2_select_checkpoint.py``; the *decision* lives here.

Selection rule (A1): among epochs ``>= min_epoch`` (the annealed half of the fixed
schedule — so a lucky high-LR early epoch can never be picked), take the max linked CPM,
tie-broken toward the LATER (more-annealed) epoch. This removes the two mechanisms that
made the pre-P3-UPDATE per-epoch AP/loss selection pick noise on the 30-lesion val set.
"""

from __future__ import annotations

from typing import Dict, Tuple


def select_epoch(epoch_cpms: Dict[int, float], min_epoch: int) -> int:
    """Return the epoch with the max CPM among ``epoch >= min_epoch``; ties -> later epoch.

    ``epoch_cpms`` maps epoch index -> linked val CPM. Raises if no epoch qualifies (the
    caller must have trained at least ``min_epoch + 1`` epochs). NaN CPMs are ignored; if
    every candidate CPM is NaN, the latest qualifying epoch is returned (a defined fallback).
    """
    import math

    cands = {e: c for e, c in epoch_cpms.items() if int(e) >= int(min_epoch)}
    if not cands:
        raise ValueError(
            f"no saved epoch >= min_epoch={min_epoch} (have epochs {sorted(epoch_cpms)}); "
            "train at least min_epoch+1 epochs")
    finite = {e: c for e, c in cands.items() if c is not None and not math.isnan(float(c))}
    if not finite:
        return max(cands)  # all-NaN fallback: the most-annealed epoch
    # max CPM, tie-broken toward the later epoch: sort by (cpm, epoch) descending.
    best_e, _ = max(finite.items(), key=lambda kv: (float(kv[1]), int(kv[0])))
    return int(best_e)


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
