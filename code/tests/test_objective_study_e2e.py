"""[4.2b] End-to-end wiring of one objective-study cell, under torch.

The study reuses the deployed path (``set_batches`` -> ``train_set_variant`` -> ``score_pool``
-> ``evaluate_variant``) with three new kwargs threaded through it. This runs that whole chain
on a synthetic pool, because the real candidate pools live on the server and a wiring break
would otherwise only surface after a server round trip.

Marked ``server``-free: it needs torch, which the laptop env has.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.candidates.record import CANDIDATE_COLUMNS
from abus_jcr.rescore.evaluate import evaluate_variant, score_pool
from abus_jcr.rescore.objective import objective_grid, record_lesion_weights, record_targets
from abus_jcr.rescore.tokens import build_feature_matrix

torch = pytest.importorskip("torch")

from abus_jcr.rescore.setmodel import B1Rescorer          # noqa: E402  (needs torch)
from abus_jcr.rescore.train import train_set_variant      # noqa: E402


def _record(n_per_vol=9, vols=(100, 101, 102), seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_vol * len(vols)
    lab, iou = [], []
    for _ in vols:                       # one TP, one ignore-band near miss, rest negatives
        lab += ["pos"] + ["ignore"] + ["neg"] * (n_per_vol - 2)
        iou += [0.55, 0.20] + list(rng.uniform(0.0, 0.05, n_per_vol - 2))
    return pd.DataFrame({
        "public_id": np.repeat(list(vols), n_per_vol),
        "candidate_id": [f"full_seed0:{i}" for i in range(n)],
        "detector_of_origin": ["full_seed0"] * n, "split": ["val"] * n, "fold": [-1] * n,
        "coordX": rng.uniform(50, 400, n), "coordY": rng.uniform(50, 400, n),
        "coordZ": rng.uniform(20, 150, n),
        "x_length": rng.uniform(10, 60, n), "y_length": rng.uniform(10, 60, n),
        "z_length": rng.uniform(5, 40, n),
        "score_max": rng.uniform(0.08, 0.6, n), "score_mean": rng.uniform(0.02, 0.4, n),
        "score_std": rng.uniform(0, 0.1, n), "score_min": rng.uniform(0, 0.1, n),
        "slice_count": rng.integers(8, 60, n).astype(float),
        "z_span": rng.integers(8, 90, n).astype(float), "fill_ratio": rng.uniform(0.3, 1.0, n),
        "centroid_jitter": rng.uniform(0, 1, n), "area_cv": rng.uniform(0, 1, n),
        "rank": np.tile(np.arange(1, n_per_vol + 1), len(vols)),
        "rank_norm": np.tile(np.linspace(0, 1, n_per_vol), len(vols)),
        "label": lab, "iou_gt": iou,
        "cen_d0": rng.uniform(10, 140, n), "cen_d1": rng.uniform(10, 300, n),
        "cen_d2": rng.uniform(10, 380, n), "ext_d0": rng.uniform(5, 60, n),
        "ext_d1": rng.uniform(5, 180, n), "ext_d2": rng.uniform(5, 80, n),
        "preprocess_hash": ["abc"] * n,
    })[CANDIDATE_COLUMNS]


def _gt(rec):
    """One GT per volume, placed on each volume's `pos` row so the oracle finds a hit."""
    rows = rec[rec["label"] == "pos"]
    return pd.DataFrame({
        "public_id": rows["public_id"].to_numpy(),
        "coordX": rows["coordX"].to_numpy(), "coordY": rows["coordY"].to_numpy(),
        "coordZ": rows["coordZ"].to_numpy(), "x_length": rows["x_length"].to_numpy(),
        "y_length": rows["y_length"].to_numpy(), "z_length": rows["z_length"].to_numpy(),
    })


def _train_one(rec, gt, cell, tmp_path, epochs=2):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from _phase4_common import boxes_of, set_batches, set_index_lists

    iso = {int(p): (158, 341, 420) for p in rec["public_id"].unique()}
    Z, _ = build_feature_matrix(rec, emb=None,
                                use_blocks=("abs_geom", "score_stats", "tube_geom", "rank"),
                                iso_shape_by_pid=iso)
    Z = np.ascontiguousarray(Z, dtype=np.float32)
    coord, length = boxes_of(rec)
    sets = set_index_lists(rec)

    payload = train_set_variant(
        lambda: B1Rescorer(d_in=Z.shape[1], d_model=32, hidden=32, depth=2),
        set_batches(rec, Z, seed=0, labels=record_targets(rec, cell["soft"])),
        lambda ep, m: {"val_cpm": 0.5, "val_ceiling": 1.0},
        tmp_path / cell["name"], seed=0, w_rank=0.0, lam=1.0, alpha=float(cell["alpha"]),
        epochs=epochs, device="cpu", gamma=float(cell["gamma"]),
        soft_targets=bool(cell["soft"]),
        row_weights=(record_lesion_weights(rec, record_targets(rec, cell["soft"]))
                     if cell["per_lesion"] else None))
    model = payload.pop("model")           # the trainer builds it now — seeded first
    prob = score_pool(model, Z, coord, length, sets, n_rows=len(rec), device="cpu")
    return payload, prob


def test_every_grid_cell_trains_and_produces_a_scorable_probability(tmp_path):
    rec, = (_record(),)
    gt = _gt(rec)
    for cell in objective_grid():
        payload, prob = _train_one(rec, gt, cell, tmp_path)
        assert np.isfinite(prob).all(), f"{cell['name']} produced non-finite probabilities"
        assert ((prob >= 0.0) & (prob < 1.0)).all(), \
            f"{cell['name']} left the det_score [0,1) contract"
        assert len(payload["epochs"]) == 2
        res = evaluate_variant(rec, prob, gt, seed_tag=cell["name"], n_boot=0)
        assert 0.0 <= res["cpm"] <= 1.0


def _cell(name):
    """Select by NAME, never via `is_deployed` — that flag tracks `conventions` and moved when
    [4.2c] promoted an objective, which would silently make a one-factor contrast compare a
    cell against itself."""
    return next(c for c in objective_grid() if c["name"] == name)


def test_the_soft_cell_actually_trains_on_the_ignore_band(tmp_path):
    """The whole point of the ramp: an IoU-0.20 row must move the weights.

    Matched on every other factor, so the ONLY difference is the target. If the ignore band
    were still masked out the two runs would be identical.
    """
    rec = _record()
    gt = _gt(rec)
    _, p_hard = _train_one(rec, gt, _cell("g2_a0.25_hard_cand"), tmp_path, epochs=3)
    _, p_soft = _train_one(rec, gt, _cell("g2_a0.25_soft_cand"), tmp_path, epochs=3)
    assert not np.allclose(p_hard, p_soft)


def test_per_lesion_weighting_changes_the_fit(tmp_path):
    """Matched on gamma, alpha and target — only the weighting differs."""
    rec = _record()
    gt = _gt(rec)
    _, p_base = _train_one(rec, gt, _cell("g2_a0.25_hard_cand"), tmp_path, epochs=3)
    _, p_w = _train_one(rec, gt, _cell("g2_a0.25_hard_lesion"), tmp_path, epochs=3)
    assert not np.allclose(p_base, p_w)


def test_the_promoted_objective_is_reachable_by_name(tmp_path):
    """[4.2c] promoted g0_a0.25_hard_lesion; the ladder trains under it via `conventions`."""
    from abus_jcr import conventions as C

    promoted = _cell("g0_a0.25_hard_lesion")
    assert promoted["is_deployed"] is True
    assert C.RESC_FOCAL_GAMMA == 0.0 and C.RESC_PER_LESION_WEIGHTS is True
    rec = _record()
    _, prob = _train_one(rec, _gt(rec), promoted, tmp_path, epochs=2)
    assert np.isfinite(prob).all() and ((prob >= 0.0) & (prob < 1.0)).all()
