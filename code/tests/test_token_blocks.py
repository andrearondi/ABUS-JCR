"""[4.3] The per-candidate token — block-ablatable, leak-free, detector-blind.

Three contracts this pins:
1. A disabled block is **omitted**, not zeroed (so the input projection re-sizes and the
   sub-ablations in §4.7 measure a real capacity change, not a masked one).
2. ``z_span`` and ``fill_ratio`` live ONLY in ``score_stats`` — the de-duplication that
   resolves the double-listing in PHASE_4.md §1.2, recorded so the block ablation is clean.
3. ``detector_of_origin`` never reaches the model (Inv. 7) and the standardisation stats
   come from the TRAIN pool only (Inv. 9/10) — no val statistic ever enters them.

Torch-free: the feature matrix is pure numpy/pandas.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.tokens import (
    BLOCK_DIMS,
    apply_standardiser,
    build_feature_matrix,
    fit_standardiser,
    rank_sinusoid,
)

ISO_SHAPE = {7: (158, 341, 420), 9: (158, 304, 392)}


def _record(n=6, det="fold0"):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "public_id": [7] * (n // 2) + [9] * (n - n // 2),
        "candidate_id": [f"{det}:{i}" for i in range(n)],
        "detector_of_origin": [det] * n,
        "score_max": rng.uniform(0.05, 0.6, n), "score_mean": rng.uniform(0.02, 0.4, n),
        "score_std": rng.uniform(0.0, 0.1, n), "score_min": rng.uniform(0.0, 0.1, n),
        "slice_count": rng.integers(8, 60, n).astype(float),
        "z_span": rng.integers(8, 90, n).astype(float),
        "fill_ratio": rng.uniform(0.3, 1.0, n),
        "centroid_jitter": rng.uniform(0, 1, n), "area_cv": rng.uniform(0, 1, n),
        "rank": np.arange(1, n + 1), "rank_norm": np.linspace(0, 1, n),
        "cen_d0": rng.uniform(10, 140, n), "cen_d1": rng.uniform(10, 300, n),
        "cen_d2": rng.uniform(10, 380, n),
        "ext_d0": rng.uniform(5, 60, n), "ext_d1": rng.uniform(5, 180, n),
        "ext_d2": rng.uniform(5, 80, n),
    })


def _emb(n, d=None):
    return np.random.default_rng(1).normal(size=(n, d or C.RESC_D_APP))


# --------------------------------------------------------------------------- block dims
def test_all_blocks_together_give_the_declared_width():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    assert X.shape == (len(rec), sum(BLOCK_DIMS.values()))
    assert len(names) == X.shape[1]


def test_declared_block_dims_match_the_spec_table():
    assert BLOCK_DIMS == {"appearance": C.RESC_D_APP, "abs_geom": 7,
                          "score_stats": 7, "tube_geom": 2, "rank": 17}


@pytest.mark.parametrize("dropped", ["appearance", "abs_geom", "score_stats", "tube_geom", "rank"])
def test_disabling_a_block_removes_exactly_its_dims(dropped):
    rec = _record()
    emb = _emb(len(rec))
    keep = tuple(b for b in C.RESC_TOKEN_BLOCKS if b != dropped)
    X_full, _ = build_feature_matrix(rec, emb=emb, iso_shape_by_pid=ISO_SHAPE)
    X_drop, _ = build_feature_matrix(rec, emb=emb, use_blocks=keep, iso_shape_by_pid=ISO_SHAPE)
    assert X_drop.shape[1] == X_full.shape[1] - BLOCK_DIMS[dropped]


@pytest.mark.parametrize("dropped", ["abs_geom", "score_stats", "tube_geom", "rank"])
def test_the_surviving_blocks_are_bit_identical_after_a_drop(dropped):
    """A disabled block is OMITTED, not zeroed — the remaining columns must not move."""
    rec = _record()
    emb = _emb(len(rec))
    keep = tuple(b for b in C.RESC_TOKEN_BLOCKS if b != dropped)
    X_full, n_full = build_feature_matrix(rec, emb=emb, iso_shape_by_pid=ISO_SHAPE)
    X_drop, n_drop = build_feature_matrix(rec, emb=emb, use_blocks=keep, iso_shape_by_pid=ISO_SHAPE)
    for j, name in enumerate(n_drop):
        np.testing.assert_allclose(X_drop[:, j], X_full[:, n_full.index(name)], atol=0)


def test_appearance_block_may_be_omitted_without_an_embedding():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None,
                                    use_blocks=("abs_geom", "score_stats", "tube_geom", "rank"),
                                    iso_shape_by_pid=ISO_SHAPE)
    assert X.shape[1] == 7 + 7 + 2 + 17
    assert not any(n.startswith("app") for n in names)


def test_appearance_block_without_an_embedding_is_an_error():
    with pytest.raises(ValueError, match="appearance"):
        build_feature_matrix(_record(), emb=None, iso_shape_by_pid=ISO_SHAPE)


# --------------------------------------------------------------------------- de-duplication
def test_zspan_and_fill_ratio_live_only_in_score_stats():
    rec = _record()
    _, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    for col in ("z_span", "fill_ratio"):
        hits = [n for n in names if col in n]
        assert len(hits) == 1 and hits[0].startswith("score_stats"), hits


def test_abs_geom_carries_centroid_logsize_and_anisotropy_only():
    rec = _record()
    _, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    ag = [n.split(":", 1)[1] for n in names if n.startswith("abs_geom")]
    assert ag == ["cen_d0_norm", "cen_d1_norm", "cen_d2_norm",
                  "log1p_ext_d0", "log1p_ext_d1", "log1p_ext_d2", "anisotropy"]


def test_score_stats_block_is_the_frozen_column_list():
    rec = _record()
    _, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    ss = [n.split(":", 1)[1] for n in names if n.startswith("score_stats")]
    assert [s.replace("log1p_", "") for s in ss] == list(C.SCORE_STAT_COLUMNS)


def test_slice_count_and_zspan_are_log1p_transformed():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, use_blocks=("score_stats",),
                                    iso_shape_by_pid=ISO_SHAPE)
    j = names.index("score_stats:log1p_slice_count")
    np.testing.assert_allclose(X[:, j], np.log1p(rec["slice_count"].to_numpy(float)))


def test_tube_geom_block_is_the_frozen_two_column_list():
    rec = _record()
    _, names = build_feature_matrix(rec, emb=None, use_blocks=("tube_geom",),
                                    iso_shape_by_pid=ISO_SHAPE)
    assert [n.split(":", 1)[1] for n in names] == list(C.TUBE_GEOM_COLUMNS)


# --------------------------------------------------------------------------- abs geom values
def test_centroids_are_normalised_by_the_volume_iso_shape():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, use_blocks=("abs_geom",),
                                    iso_shape_by_pid=ISO_SHAPE)
    j = names.index("abs_geom:cen_d1_norm")
    expected = rec["cen_d1"].to_numpy(float) / np.array(
        [ISO_SHAPE[int(p)][1] for p in rec["public_id"]], dtype=float)
    np.testing.assert_allclose(X[:, j], expected)


def test_anisotropy_is_depth_over_mean_lateral_matching_the_phase0b_probe():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, use_blocks=("abs_geom",),
                                    iso_shape_by_pid=ISO_SHAPE)
    j = names.index("abs_geom:anisotropy")
    lat = (rec["ext_d1"].to_numpy(float) + rec["ext_d2"].to_numpy(float)) / 2.0
    np.testing.assert_allclose(X[:, j], rec["ext_d0"].to_numpy(float) / lat)


def test_missing_iso_shape_is_a_loud_error_not_a_silent_default():
    with pytest.raises(KeyError):
        build_feature_matrix(_record(), emb=None, use_blocks=("abs_geom",),
                             iso_shape_by_pid={7: (158, 341, 420)})


# --------------------------------------------------------------------------- rank block
def test_rank_block_is_rank_norm_plus_a_sinusoidal_embedding():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, use_blocks=("rank",), iso_shape_by_pid=ISO_SHAPE)
    assert X.shape[1] == 1 + C.RESC_RANK_PE_DIM
    assert names[0] == "rank:rank_norm"
    np.testing.assert_allclose(X[:, 0], rec["rank_norm"].to_numpy(float))
    np.testing.assert_allclose(X[:, 1:], rank_sinusoid(rec["rank"].to_numpy()))


def test_rank_sinusoid_is_bounded_and_distinguishes_ranks():
    pe = rank_sinusoid(np.arange(1, 30))
    assert pe.shape == (29, C.RESC_RANK_PE_DIM)
    assert np.abs(pe).max() <= 1.0
    assert not np.allclose(pe[0], pe[1])


# --------------------------------------------------------------------------- Inv. 7
def test_detector_of_origin_never_becomes_a_feature():
    rec = _record()
    X_a, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    rec_b = rec.copy()
    rec_b["detector_of_origin"] = "full_seed2"          # a different detector entirely
    X_b, _ = build_feature_matrix(rec_b, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    np.testing.assert_array_equal(X_a, X_b)
    assert not any("detector" in n for n in names)


# --------------------------------------------------------------------------- standardisation
def test_standardiser_is_fitted_on_train_and_applied_unchanged_to_val():
    train = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    val = np.array([[100.0, 1000.0]])
    stats = fit_standardiser(train)
    np.testing.assert_allclose(stats["mean"], [2.0, 20.0])
    z_train = apply_standardiser(train, stats)
    np.testing.assert_allclose(z_train.mean(axis=0), [0.0, 0.0], atol=1e-12)
    z_val = apply_standardiser(val, stats)
    # val is standardised by TRAIN statistics -> it is NOT centred; no val stat leaked in
    assert abs(z_val[0, 0]) > 10.0


def test_standardiser_survives_a_constant_column():
    stats = fit_standardiser(np.array([[1.0, 5.0], [1.0, 7.0]]))
    out = apply_standardiser(np.array([[1.0, 5.0]]), stats)
    assert np.isfinite(out).all()


def test_standardiser_round_trips_through_json():
    import json
    stats = fit_standardiser(np.array([[0.0, 10.0], [2.0, 20.0]]))
    back = json.loads(json.dumps({k: list(map(float, v)) for k, v in stats.items()}))
    np.testing.assert_allclose(back["mean"], stats["mean"])
