"""The wrapped ``evaluate()`` reproduces analytically-known FROC outcomes.

Pins CPM = average_recall, the strict >0.3 hit, the recall ceiling accessor,
the prediction-CSV validator, and the volume-level bootstrap CI.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
from abus_jcr.eval.froc import (
    evaluate_froc,
    cpm,
    recall_ceiling,
    key_recall,
    write_pred_csv,
    bootstrap_cpm_ci,
)


def _gt_frame():
    # three single-lesion volumes, boxes far apart in coordinate space
    rows = [
        (1, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0),
        (2, 150.0, 150.0, 150.0, 10.0, 10.0, 10.0),
        (3, 250.0, 250.0, 250.0, 10.0, 10.0, 10.0),
    ]
    return pd.DataFrame(rows, columns=GT_COLUMNS)


def _pred(pid, box, prob):
    return dict(zip(PRED_COLUMNS, (pid, *box, prob)))


def test_exact_copy_predictions_give_perfect_recall():
    gt = _gt_frame()
    preds = pd.DataFrame(
        [_pred(int(r.public_id),
               (r.coordX, r.coordY, r.coordZ, r.x_length, r.y_length, r.z_length),
               0.9)
         for r in gt.itertuples()],
        columns=PRED_COLUMNS,
    )
    res = evaluate_froc(gt, preds)
    assert recall_ceiling(res) == pytest.approx(1.0, abs=1e-6)
    assert cpm(res) == pytest.approx(1.0, abs=1e-6)


def test_all_miss_predictions_give_zero_recall():
    gt = _gt_frame()
    # predictions nowhere near any GT
    preds = pd.DataFrame(
        [_pred(1, (1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0), 0.9)],
        columns=PRED_COLUMNS,
    )
    res = evaluate_froc(gt, preds)
    assert recall_ceiling(res) < 1e-6
    assert cpm(res) < 1e-6


def test_boundary_hit_and_miss_give_two_thirds_ceiling():
    # box centred on GT #1 but shifted along x. A half-width shift gives
    # IoU = 1/3 (> 0.3 hit); a 0.6-width shift gives 0.25 (< 0.3 miss).
    gt = _gt_frame()
    w = 10.0
    hit = _pred(1, (50.0 + 0.5 * w, 50.0, 50.0, w, w, w), 0.9)     # iou 1/3 -> hit
    miss = _pred(2, (150.0 + 0.6 * w, 150.0, 150.0, w, w, w), 0.9)  # iou 0.25 -> miss
    # a second guaranteed hit so ceiling = 2/3 of 3 GTs
    hit3 = _pred(3, (250.0, 250.0, 250.0, w, w, w), 0.9)
    preds = pd.DataFrame([hit, miss, hit3], columns=PRED_COLUMNS)
    res = evaluate_froc(gt, preds)
    assert recall_ceiling(res) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_key_recall_maps_seven_key_fps():
    gt = _gt_frame()
    preds = pd.DataFrame(
        [_pred(int(r.public_id),
               (r.coordX, r.coordY, r.coordZ, r.x_length, r.y_length, r.z_length),
               0.9)
         for r in gt.itertuples()],
        columns=PRED_COLUMNS,
    )
    kr = key_recall(evaluate_froc(gt, preds))
    assert list(kr.keys()) == [0.125, 0.25, 0.5, 1, 2, 4, 8]


def test_write_pred_csv_validates_probability_range(tmp_path):
    good = pd.DataFrame([_pred(1, (0, 0, 0, 1, 1, 1), 0.5)], columns=PRED_COLUMNS)
    out = tmp_path / "pred.csv"
    write_pred_csv(good, out)
    assert out.exists()
    reload = pd.read_csv(out)
    assert list(reload.columns) == PRED_COLUMNS

    bad = pd.DataFrame([_pred(1, (0, 0, 0, 1, 1, 1), 1.0)], columns=PRED_COLUMNS)
    with pytest.raises((ValueError, AssertionError)):
        write_pred_csv(bad, tmp_path / "bad.csv")


def test_bootstrap_cpm_ci_is_seeded_and_brackets_point():
    gt = _gt_frame()
    preds = pd.DataFrame(
        [_pred(int(r.public_id),
               (r.coordX, r.coordY, r.coordZ, r.x_length, r.y_length, r.z_length),
               0.9)
         for r in gt.itertuples()],
        columns=PRED_COLUMNS,
    )
    # small n_boot keeps the suite fast; determinism/bracketing are independent
    # of the draw count (the vendored evaluate() runs a 200-threshold sweep per
    # draw, so boots dominate wall-clock).
    ci_a = bootstrap_cpm_ci(gt, preds, n_boot=25, seed=0)
    ci_b = bootstrap_cpm_ci(gt, preds, n_boot=25, seed=0)
    assert ci_a["point"] == ci_b["point"]  # deterministic under fixed seed
    assert ci_a["lo"] == ci_b["lo"] and ci_a["hi"] == ci_b["hi"]
    assert ci_a["lo"] <= ci_a["point"] <= ci_a["hi"]
    # perfect predictions -> CPM 1 everywhere -> tight interval at 1
    assert ci_a["point"] == pytest.approx(1.0, abs=1e-6)
