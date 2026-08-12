"""[4.2b] ``set_batches`` must accept an explicit target array.

The soft ramp target is a float in [0, 1]; the record's ``label`` column is a string
(``pos``/``neg``/``ignore``), so it cannot carry one. Rather than fork the batcher, the study
injects targets and everything else — schedule, set batching, shuffling — stays the [4.6] path.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _phase4_common import set_batches  # noqa: E402


def _rec(n=6):
    return pd.DataFrame({
        "public_id": [0] * n,
        "detector_of_origin": ["full_seed0"] * n,
        "label": ["pos", "neg", "ignore"] * (n // 3),
        "coordX": np.arange(n, dtype=float), "coordY": np.arange(n, dtype=float),
        "coordZ": np.arange(n, dtype=float),
        "x_length": np.ones(n), "y_length": np.ones(n), "z_length": np.ones(n),
    })


def test_explicit_targets_reach_the_batch():
    rec = _rec()
    soft = np.linspace(0.0, 1.0, len(rec))
    batch = next(iter(set_batches(rec, np.zeros((len(rec), 2)), seed=0,
                                  labels=soft, shuffle=False)(0)))
    assert batch["labels"][0, :len(rec)] == __import__("pytest").approx(soft, abs=1e-6)


def test_default_still_encodes_the_record_label_column():
    rec = _rec()
    batch = next(iter(set_batches(rec, np.zeros((len(rec), 2)), seed=0, shuffle=False)(0)))
    got = batch["labels"][0, :len(rec)]
    assert list(got[:3]) == [1.0, 0.0, -1.0]
