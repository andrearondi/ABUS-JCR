"""[4.2b] The numpy half of the study's chain: record -> collate_sets -> weights -> loss.

``train_set_variant``/``score_pool`` need torch, which the laptop env does not have
(``test_objective_study_e2e`` covers them and runs on the server). Everything that decides
*what the loss sees* is array-agnostic, so it is pinned here — and this is where a defect
would be silent rather than loud: a row-index misalignment between ``collate_sets``' padded
layout and the per-record weight vector would quietly weight the wrong candidates.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.rescore.datasets import batch_row_weights, collate_sets, group_sets
from abus_jcr.rescore.losses import encode_labels, rescorer_loss
from abus_jcr.rescore.objective import record_lesion_weights, record_targets


def _rec():
    """Set A: 4 duplicate hits on one lesion + 1 FP.  Set B: 1 hit + 1 FP.

    To the oracle both sets are worth one lesion. To a plain mean, A's lesion is worth 4x B's.
    """
    lab = ["pos"] * 4 + ["neg"] + ["pos", "neg"]
    iou = [0.55, 0.61, 0.48, 0.52, 0.02] + [0.58, 0.20]
    return pd.DataFrame({
        "public_id": [100] * 5 + [101] * 2,
        "detector_of_origin": ["full_seed0"] * 7,
        "label": lab, "iou_gt": iou,
    })


def _batch(rec, weights=None):
    idx = list(group_sets(rec).values())
    n = len(rec)
    feats = np.zeros((n, 3), dtype=np.float32)
    coord = np.zeros((n, 3)); length = np.ones((n, 3))
    b = collate_sets(idx, feats, encode_labels(rec["label"].to_numpy()), coord, length)
    return b, batch_row_weights(b["rows"], weights)


def test_each_set_contributes_exactly_one_lesions_worth_of_positive_weight():
    rec = _rec()
    b, w = _batch(rec, record_lesion_weights(rec))
    is_pos = b["labels"] > 0.5
    assert w[is_pos].sum() == pytest.approx(2.0), "two sets => two lesions' worth"
    assert w[is_pos & (b["rows"] < 5)].sum() == pytest.approx(1.0)   # set A's four duplicates


def test_padding_carries_no_weight_even_though_the_sets_are_ragged():
    """Set A has 5 rows, set B has 2, so B is padded to width 5 — those slots must be inert."""
    rec = _rec()
    b, w = _batch(rec, record_lesion_weights(rec))
    assert b["mask"].shape[1] == 5
    assert w[~b["mask"]].sum() == 0.0


def test_weighting_moves_the_loss_toward_the_under_covered_lesion():
    rec = _rec()
    b, w = _batch(rec, record_lesion_weights(rec))
    # a fit that is right on set A's four duplicates and wrong on set B's lone hit
    logits = np.where(b["labels"] > 0.5, 4.0, -4.0)
    logits[1, 0] = -4.0                                   # set B's positive, badly wrong
    flat, _ = rescorer_loss(logits, b["labels"], b["mask"].astype(float),
                            w_rank=0.0, lam=1.0, alpha=0.5)
    tilted, _ = rescorer_loss(logits, b["labels"], b["mask"].astype(float),
                              w_rank=0.0, lam=1.0, alpha=0.5, bce_weights=w)
    assert float(tilted) > float(flat), (
        "down-weighting A's duplicates must raise the cost of B's missed lesion")


def test_the_soft_ramp_target_reaches_the_batch_and_keeps_the_near_miss():
    rec = _rec()
    idx = list(group_sets(rec).values())
    n = len(rec)
    b = collate_sets(idx, np.zeros((n, 3), dtype=np.float32),
                     record_targets(rec, soft=True), np.zeros((n, 3)), np.ones((n, 3)))
    # row 6 is the IoU-0.20 near miss: the hard code drops it, the ramp gives it 0.5
    at = b["labels"][b["rows"] == 6]
    assert at.size == 1 and at[0] == pytest.approx(0.5)
    assert (b["labels"][b["mask"]] >= 0.0).all(), "no real row may carry the ignore code"
    assert (b["labels"][~b["mask"]] < 0.0).all(), "padding must still be excluded"
