"""Parallel bootstrap draws ≡ sequential, bit for bit (eval/froc, 2026-09-09).

The parallelisation contract: draw indices are pre-generated from the same seeded RNG in the
same order, workers evaluate independent draws, results reassemble in submission order. If any
of that drifts — a per-worker RNG, an unordered map, shared temp files — these equalities
break. ``np.array_equal`` on the full boot arrays, not approx: the claim is identity, not
closeness.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.eval.froc import bootstrap_cpm_ci, paired_bootstrap_delta

N_BOOT = 8      # identity is identity; the oracle costs ~1 s/call, so keep the draw count low


def _gt(pids):
    n = len(pids)
    return pd.DataFrame({
        "public_id": pids,
        "coordX": np.arange(n) * 60.0, "coordY": [0.0] * n, "coordZ": [0.0] * n,
        "x_length": [10.0] * n, "y_length": [10.0] * n, "z_length": [10.0] * n})


def _pred(gt, probs, jitter=0.0):
    p = gt.copy()
    p["coordX"] = p["coordX"] + jitter          # jitter>3.4 -> IoU<0.3 -> misses
    p["probability"] = probs
    return p[["public_id", "coordX", "coordY", "coordZ",
              "x_length", "y_length", "z_length", "probability"]]


@pytest.fixture()
def frames():
    gt = _gt([130, 131, 132, 133, 134, 135])
    a = _pred(gt, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    b = _pred(gt, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], jitter=2.0)
    return gt, a, b


def test_marginal_ci_identical_across_worker_counts(frames):
    gt, a, _ = frames
    seq = bootstrap_cpm_ci(gt, a, n_boot=N_BOOT, seed=0, workers=1)
    par = bootstrap_cpm_ci(gt, a, n_boot=N_BOOT, seed=0, workers=2)
    assert np.array_equal(seq["boot"], par["boot"])
    assert (seq["point"], seq["lo"], seq["hi"]) == (par["point"], par["lo"], par["hi"])


def test_paired_delta_identical_across_worker_counts(frames):
    gt, a, b = frames
    seq = paired_bootstrap_delta(gt, a, b, n_boot=N_BOOT, seed=0, workers=1)
    par = paired_bootstrap_delta(gt, a, b, n_boot=N_BOOT, seed=0, workers=3)
    assert np.array_equal(seq["boot"], par["boot"])
    assert seq["delta_point"] == par["delta_point"]
    assert seq["frac_positive"] == par["frac_positive"]
    assert (seq["lo"], seq["hi"]) == (par["lo"], par["hi"])


def test_default_stays_sequential_and_reproducible(frames):
    # the workers argument must not have changed the default path: same seed -> same interval
    gt, a, _ = frames
    r1 = bootstrap_cpm_ci(gt, a, n_boot=N_BOOT, seed=0)
    r2 = bootstrap_cpm_ci(gt, a, n_boot=N_BOOT, seed=0)
    assert np.array_equal(r1["boot"], r2["boot"])
