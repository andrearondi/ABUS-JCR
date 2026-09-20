"""The bootstrap matrix IS the paired bootstrap (scripts/phase5_factor_comparisons, 2026-09-20).

The script's claim: one per-condition bootstrap vector per rung serves every pair, because the
marginal and the paired bootstrap share their seeded draws. Identity, not closeness — so
``np.array_equal`` against the two functions the reportable pass used.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase5_factor_comparisons as P  # noqa: E402

from abus_jcr.eval.froc import bootstrap_cpm_ci, paired_bootstrap_delta  # noqa: E402
from abus_jcr.rescore.factor_grid import pair_stats  # noqa: E402

N_BOOT = 6
META = {"rung": "X", "seed": 0, "n_boot": N_BOOT, "boot_seed": 0}


def _gt(pids):
    n = len(pids)
    return pd.DataFrame({
        "public_id": pids,
        "coordX": np.arange(n) * 60.0, "coordY": [0.0] * n, "coordZ": [0.0] * n,
        "x_length": [10.0] * n, "y_length": [10.0] * n, "z_length": [10.0] * n})


def _pred(gt, probs, jitter=0.0):
    p = gt.copy()
    p["coordX"] = p["coordX"] + jitter
    p["probability"] = probs
    return p[["public_id", "coordX", "coordY", "coordZ",
              "x_length", "y_length", "z_length", "probability"]]


@pytest.fixture()
def frames():
    gt = _gt([130, 131, 132, 133, 134, 135])
    a = _pred(gt, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    b = pd.concat([_pred(gt, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], jitter=20.0),
                   _pred(gt, [0.3, 0.3, 0.3, 0.3, 0.3, 0.95])], ignore_index=True)
    return gt, a, b


def test_matrix_column_is_the_marginal_bootstrap(frames, tmp_path):
    gt, a, _ = frames
    arr = P.boot_unit(gt, a, tmp_path / "a.npy", META, N_BOOT, workers=1)
    assert np.array_equal(arr[:, 0], bootstrap_cpm_ci(gt, a, n_boot=N_BOOT, seed=0)["boot"])
    assert np.all(arr[:, 1] <= arr[:, 0] + 1e-12) and np.all(arr[:, 0] <= arr[:, 2] + 1e-12)


def test_matrix_difference_is_the_paired_bootstrap(frames, tmp_path):
    gt, a, b = frames
    ma = P.boot_unit(gt, a, tmp_path / "a.npy", META, N_BOOT, workers=1)
    mb = P.boot_unit(gt, b, tmp_path / "b.npy", META, N_BOOT, workers=2)
    ref = paired_bootstrap_delta(gt, a, b, n_boot=N_BOOT, seed=0)
    assert np.array_equal(ma[:, 0] - mb[:, 0], ref["boot"])
    ps = pair_stats(ma[:, 0], mb[:, 0])
    assert (ps["lo"], ps["hi"], ps["frac_positive"]) == (ref["lo"], ref["hi"],
                                                         ref["frac_positive"])


def test_unit_resumes_from_a_partial_and_reloads_when_complete(frames, tmp_path, monkeypatch):
    gt, a, _ = frames
    monkeypatch.setattr(P, "CHUNK", 2)
    out = tmp_path / "a.npy"
    part = P.boot_unit(gt, a, out, META, N_BOOT, workers=1, limit=2)
    assert int(np.isnan(part[:, 0]).sum()) == N_BOOT - 2 and not out.exists()

    calls = []
    real = P.F._run_draws
    monkeypatch.setattr(P.F, "_run_draws",
                        lambda w, d, s, n: calls.append(len(d)) or real(w, d, s, n))
    full = P.boot_unit(gt, a, out, META, N_BOOT, workers=1)
    assert sum(calls) == N_BOOT - 2 and out.exists()
    assert np.array_equal(full[:, 0], bootstrap_cpm_ci(gt, a, n_boot=N_BOOT, seed=0)["boot"])

    calls.clear()
    assert np.array_equal(P.boot_unit(gt, a, out, META, N_BOOT, workers=1), full)
    assert calls == []                                       # a complete unit is only loaded
    P.boot_unit(gt, a, out, {**META, "n_boot": N_BOOT, "rung": "Y"}, N_BOOT, workers=1)
    assert sum(calls) == N_BOOT                              # foreign meta -> recomputed
    assert json.loads(out.with_suffix(".json").read_text())["rung"] == "Y"
