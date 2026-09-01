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
#: Every KNOWN block, appearance included. BRANCH B (2026-09-01, [4.2d.3]) removed appearance
#: from the DEFAULT set (C.RESC_TOKEN_BLOCKS) but it stays a requestable block — the [4.2d.3]
#: study and any re-audit build it explicitly — so the tests below exercise both.
ALL_BLOCKS = tuple(BLOCK_DIMS)


def test_all_blocks_together_give_the_declared_width():
    rec = _record()
    X, names = build_feature_matrix(rec, emb=_emb(len(rec)), use_blocks=ALL_BLOCKS,
                                    iso_shape_by_pid=ISO_SHAPE)
    assert X.shape == (len(rec), sum(BLOCK_DIMS.values()))
    assert len(names) == X.shape[1]


def test_default_blocks_give_the_branch_b_width():
    """The deployed default (no appearance) is 6 + 7 + 2 + 17 = 32."""
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, iso_shape_by_pid=ISO_SHAPE)
    assert X.shape == (len(rec), sum(BLOCK_DIMS[b] for b in C.RESC_TOKEN_BLOCKS)) == (len(rec), 32)
    assert not any(n.startswith("app") for n in names)


def test_declared_block_dims_match_the_spec_table():
    # abs_geom is 6, not 7: the anisotropy dim was removed 2026-08-09 (see
    # test_abs_geom_carries_no_extent_ratio and the tokens module docstring).
    assert BLOCK_DIMS == {"appearance": C.RESC_D_APP, "abs_geom": 6,
                          "score_stats": 7, "tube_geom": 2, "rank": 17}


@pytest.mark.parametrize("dropped", ["appearance", "abs_geom", "score_stats", "tube_geom", "rank"])
def test_disabling_a_block_removes_exactly_its_dims(dropped):
    rec = _record()
    emb = _emb(len(rec))
    keep = tuple(b for b in ALL_BLOCKS if b != dropped)
    X_full, _ = build_feature_matrix(rec, emb=emb, use_blocks=ALL_BLOCKS, iso_shape_by_pid=ISO_SHAPE)
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
    assert X.shape[1] == 6 + 7 + 2 + 17      # abs_geom is 6 since the anisotropy dim went
    assert not any(n.startswith("app") for n in names)


def test_appearance_block_without_an_embedding_is_an_error():
    """Requesting appearance explicitly with no embedding must still raise (the default no
    longer includes it, so the default + emb=None is legal — pinned above)."""
    with pytest.raises(ValueError, match="appearance"):
        build_feature_matrix(_record(), emb=None, use_blocks=ALL_BLOCKS,
                             iso_shape_by_pid=ISO_SHAPE)


# --------------------------------------------------------------------------- de-duplication
def test_zspan_and_fill_ratio_live_only_in_score_stats():
    rec = _record()
    _, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    for col in ("z_span", "fill_ratio"):
        hits = [n for n in names if col in n]
        assert len(hits) == 1 and hits[0].startswith("score_stats"), hits


def test_abs_geom_carries_centroid_and_logsize_only():
    rec = _record()
    _, names = build_feature_matrix(rec, emb=_emb(len(rec)), iso_shape_by_pid=ISO_SHAPE)
    ag = [n.split(":", 1)[1] for n in names if n.startswith("abs_geom")]
    assert ag == ["cen_d0_norm", "cen_d1_norm", "cen_d2_norm",
                  "log1p_ext_d0", "log1p_ext_d1", "log1p_ext_d2"]


def test_abs_geom_carries_no_extent_ratio():
    """The ``anisotropy`` dim was REMOVED on 2026-08-09 (``abs_geom`` 7 -> 6).

    It was computed as ``ext_d0 / mean(ext_d1, ext_d2)`` and PHASE_4.md §1.2 described it as
    "depth-extent / mean-lateral-extent — the Phase-0b signal". Every part of that was wrong
    or dead: ``d0`` is the MEASURED LATERAL axis (AXIS_CHECK.md, 129/130 volumes); a
    physically cubic candidate reads 0.195 on the deployed cache's distorted scale, not 1.0;
    it is the weakest of all 12 pool features (val delta 0.097, balacc 0.543 vs a 0.5 floor);
    its motivating shadow-geometry hypothesis is a closed negative result (zero candidates
    above 2x beam elongation in 8/8 detectors, [I.6b]); and it is ~95% recoverable from the
    three ``log1p(ext_d*)`` beside it through one nonlinearity.

    This pins the removal so it cannot drift back in under any name — the ratio, not the
    label, is what was dropped."""
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, use_blocks=("abs_geom",),
                                    iso_shape_by_pid=ISO_SHAPE)
    assert X.shape[1] == 6 and len(names) == 6
    assert not any(k in n for n in names for k in ("aniso", "elong", "ratio", "depth"))
    # and no column equals the ratio numerically, whatever it might be called
    other = (rec["ext_d1"].to_numpy(float) + rec["ext_d2"].to_numpy(float)) / 2.0
    ratio = rec["ext_d0"].to_numpy(float) / other
    assert not any(np.allclose(X[:, j], ratio) for j in range(X.shape[1]))


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


def test_extents_enter_on_a_log1p_scale():
    """The three log-sizes are what remains of abs_geom's size information after the
    anisotropy dim was removed; any extent ratio the model wants is reconstructible from
    them (measured R^2 = 0.945 through one nonlinearity)."""
    rec = _record()
    X, names = build_feature_matrix(rec, emb=None, use_blocks=("abs_geom",),
                                    iso_shape_by_pid=ISO_SHAPE)
    for a in range(3):
        j = names.index(f"abs_geom:log1p_ext_d{a}")
        np.testing.assert_allclose(X[:, j], np.log1p(rec[f"ext_d{a}"].to_numpy(float)))


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
