"""[4.6] Step 3 — the pooled-surrogate WIRING (the loss itself is pinned by
test_froc_surrogate.py; this file pins how the trainer reaches it).

Written 2026-09-02 with the wiring, before any ``-P`` rung trained. The four decisions it
pins were approved by the user the same day:

* β anneal: linear ``RESC_FROC_BETA0 → RESC_FROC_BETA`` over the first
  ``RESC_FROC_BETA_ANNEAL_FRAC`` of the budget, constant after (``train.froc_beta``);
* reference table: the WHOLE train pool scored detached per epoch, each batch receiving the
  complement of its own rows (``train.ref_complement``) — else every batch candidate is
  counted twice, once live and once stale;
* warm start: the CE twin's deployed checkpoint, loaded AFTER seeding (``init_state``);
* the refuse-guard: a ``-P`` rung whose effective rank loss is not ``"froc"`` must not start.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.train import froc_beta, ref_complement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


# --------------------------------------------------------------------------- beta anneal
def test_beta_anneal_endpoints_and_midpoint():
    assert froc_beta(0, 60) == pytest.approx(C.RESC_FROC_BETA0)          # 1.0 wide at start
    assert froc_beta(30, 60) == pytest.approx(C.RESC_FROC_BETA)          # 0.25 at half budget
    assert froc_beta(59, 60) == pytest.approx(C.RESC_FROC_BETA)          # holds to the end
    mid = 0.5 * (C.RESC_FROC_BETA0 + C.RESC_FROC_BETA)
    assert froc_beta(15, 60) == pytest.approx(mid)                       # linear in between


def test_beta_anneal_is_monotone_nonincreasing():
    vals = [froc_beta(e, 60) for e in range(60)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert min(vals) == pytest.approx(C.RESC_FROC_BETA)


# --------------------------------------------------------------------------- complement
def test_ref_complement_excludes_exactly_the_batch_rows():
    rows = np.array([[3, 5, -1], [0, -1, -1]])          # padded collate_sets map
    keep = ref_complement(rows, 8)
    assert keep.tolist() == [False, True, True, False, True, False, True, True]


def test_ref_complement_ignores_padding_only():
    keep = ref_complement(np.full((2, 4), -1), 5)       # all padding -> keep everything
    assert keep.all() and keep.shape == (5,)


# --------------------------------------------------------------------------- refuse-guard
def test_a_pooled_rung_without_froc_refuses_to_start(monkeypatch):
    """THE guard. Uses run_variant_trial's real code path up to the guard, with the variant
    table monkeypatched so no heavy input loading is needed (the guard fires first)."""
    import _phase4_common as PC
    from abus_jcr.rescore import variants as V

    broken = dict(V.VARIANTS["FULL-P"]); broken.pop("rank_loss")
    monkeypatch.setitem(V.VARIANTS, "FULL-P", broken)
    with pytest.raises(SystemExit, match="refusing to train a duplicate"):
        PC.run_variant_trial(args=None, variant="FULL-P", seed=0,
                             trial={"alpha": 0.25, "lr": 1e-3, "w_rank": 1.0, "lam": 1.0},
                             capacity=("L2H128h4", 2, 128, 4), out_dir="/nonexistent",
                             inputs={"d_in": 32, "rec_tr": None, "rec_va": None,
                                     "Ztr": None, "Zva": None, "gt_va": None})


# --------------------------------------------------------------------------- trainer wiring
def _tiny_run(rank_loss=None, ref_provider=None, n_vol=None, init_state=None,
              captured=None, epochs=2, lr=1e-3):
    """Drive train_set_variant with a 1-parameter model and a spy on rescorer_loss."""
    torch = pytest.importorskip("torch")
    import abus_jcr.rescore.train as T

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, feats, coord, length, mask):
            return feats[..., 0] * self.w      # (B, N)

    real = T.rescorer_loss

    def spy(logits, labels, mask, **kw):
        if captured is not None:
            captured.append(dict(kw))
        kw.pop("bce_weights", None)
        return real(logits, labels, mask, bce_weights=None, **kw)

    def batches(ep):
        yield {"feats": np.ones((1, 3, 2), dtype=np.float32),
               "labels": np.array([[1.0, 0.0, -1.0]], dtype=np.float32),
               "coord": np.zeros((1, 3, 3), dtype=np.float32),
               "length": np.ones((1, 3, 3), dtype=np.float32),
               "mask": np.ones((1, 3), dtype=np.float32),
               "rows": np.array([[0, 1, -1]])}

    def evaluate_epoch(ep, model):
        return {"val_cpm": 0.5, "val_ceiling": 1.0,
                "val_ci_lo": float("nan"), "val_ci_hi": float("nan")}

    import unittest.mock as um
    with um.patch.object(T, "rescorer_loss", side_effect=spy):
        payload = T.train_set_variant(Tiny, batches, evaluate_epoch, "/tmp/froc_wiring_test",
                                      seed=0, w_rank=1.0, lam=1.0, alpha=0.25, lr=lr,
                                      epochs=epochs, device="cpu", rank_loss=rank_loss,
                                      n_vol=n_vol, ref_provider=ref_provider,
                                      init_state=init_state)
    return payload


def test_default_path_reaches_the_loss_with_no_pooled_kwargs():
    captured = []
    _tiny_run(captured=captured)
    assert captured, "the spy never fired"
    for kw in captured:
        assert "rank_loss" not in kw and "ref_logits" not in kw and "beta" not in kw


def test_froc_path_reaches_the_loss_with_the_full_pooled_contract():
    captured = []
    pool_logits = np.arange(5, dtype=np.float32)
    pool_labels = np.array([1, 0, 0, -1, 0], dtype=np.float32)
    calls = []

    def provider(ep, model):
        calls.append(ep)
        return pool_logits, pool_labels

    _tiny_run(rank_loss="froc", ref_provider=provider, n_vol=100.0, captured=captured)
    assert calls == [0, 1], "the reference table must be rebuilt once per epoch"
    for kw in captured:
        assert kw["rank_loss"] == "froc" and kw["n_vol"] == 100.0
        # the batch's rows (0, 1) are excluded; padding (-1) excludes nothing
        assert kw["ref_logits"].tolist() == [2.0, 3.0, 4.0]
        assert kw["ref_labels"].tolist() == [0.0, -1.0, 0.0]
    # beta follows the anneal, per epoch
    assert captured[0]["beta"] == pytest.approx(froc_beta(0, 2))
    assert captured[-1]["beta"] == pytest.approx(froc_beta(1, 2))


# --------------------------------------------------------------------------- [4.7] default rungs
def test_eval_grid_default_includes_the_pooled_rungs():
    """Fixed 2026-09-03: the old default was LADDER alone, and absent rungs are skipped rather
    than faked — so a flagless [4.7] run silently dropped every pooled comparison (FULL-P vs B2
    included) and stored no -P curves for the thesis FROC figure."""
    from phase4_eval_grid import DEFAULT_VARIANTS

    assert DEFAULT_VARIANTS == ("B1", "B2", "A1", "A2", "FULL", "B1-P", "A2-P", "FULL-P")


# --------------------------------------------------------------------------- deployed-model load
def test_load_deployed_model_matches_hidden_for_every_mlp_rung(monkeypatch, tmp_path):
    """The 2026-09-03 [MIG-5] crash: `hidden` was computed only for the literal "B1", so B1-P
    (also an mlp rung) reached build_rescorer with hidden=None and raised. Same bug class as
    the Branch-B embeddings loader. Pinned for BOTH mlp rungs and one set rung."""
    torch = pytest.importorskip("torch")

    import _phase4_common as PC
    from abus_jcr.rescore.setmodel import build_rescorer

    cap = ("L2H128h4", 2, 128, 4)

    def fake_report(args, variant, seed):
        # a real deployed report's shape, pointing at a checkpoint we create on the fly
        m = build_rescorer(variant, 32, cap,
                           hidden=PC.b1_hidden_for(args, 32, cap)
                           if variant in ("B1", "B1-P") else None)
        p = tmp_path / f"{variant}.pt"
        torch.save({"model": m.state_dict()}, p)
        return {"capacity": list(cap), "geom_mechanism": None,
                "deployed": {"selected_ckpt": str(p)}}

    monkeypatch.setattr(PC, "load_deployed_report", fake_report)
    args = type("A", (), {"out_root": str(tmp_path)})()
    for variant in ("B1", "B1-P", "A2-P"):
        model, rep = PC.load_deployed_model(args, variant, 0, 32, device="cpu")
        assert model is not None, variant


# --------------------------------------------------------------------------- Branch B loading
def test_token_only_inputs_never_touch_the_embedding_files(monkeypatch, tmp_path):
    """The exact 2026-09-02 session A/B crash: Branch B trains no encoder for seeds 1/2, so
    their embedding files must not merely go unused — they must never be OPENED. Pinned by
    running load_variant_inputs for seed 1 with an out-root holding no embeddings at all."""
    import pandas as pd

    import _phase4_common as PC

    rec = pd.DataFrame({
        "public_id": [1, 1, 2, 2], "detector_of_origin": ["full_seed1"] * 4,
        "label": ["pos", "neg", "pos", "neg"], "rank": [1, 2, 1, 2],
        "score_max": [0.9, 0.5, 0.8, 0.4],
    })
    gt = pd.DataFrame({"public_id": [1, 2]})
    args = type("A", (), {"out_root": str(tmp_path), "emb_suffix": ""})()

    monkeypatch.setattr(PC, "load_record", lambda a, split: rec.copy())
    monkeypatch.setattr(PC, "load_gt", lambda a, split: gt.copy())
    monkeypatch.setattr(PC, "gt_for_pool", lambda g, r: g)
    monkeypatch.setattr(PC, "iso_shape_map", lambda a, r: {1: (8, 8, 8), 2: (8, 8, 8)})
    monkeypatch.setattr(PC, "build_features",
                        lambda r, emb, blocks, shapes, stats=None:
                        (np.zeros((len(r), 32), np.float32), ["f"] * 32,
                         {"mean": [0.0] * 32, "std": [1.0] * 32}))

    out = PC.load_variant_inputs(args, seed=1, use_blocks=tuple(C.RESC_TOKEN_BLOCKS))
    assert out["d_in"] == 32          # and, crucially, no FileNotFoundError above


def test_appearance_blocks_still_demand_the_files(monkeypatch, tmp_path):
    """The converse: an explicit appearance build must still fail loudly on a missing file,
    not silently degrade to token-only."""
    import pandas as pd

    import _phase4_common as PC

    args = type("A", (), {"out_root": str(tmp_path), "emb_suffix": ""})()
    monkeypatch.setattr(PC, "load_record",
                        lambda a, split: pd.DataFrame({"public_id": [1],
                                                       "detector_of_origin": ["full_seed1"]}))
    with pytest.raises(FileNotFoundError):
        PC.load_variant_inputs(args, seed=1,
                               use_blocks=("appearance",) + tuple(C.RESC_TOKEN_BLOCKS))


def test_warm_start_loads_the_given_state_after_seeding():
    torch = pytest.importorskip("torch")
    init = {"w": torch.tensor([7.5])}
    payload = _tiny_run(init_state=init, epochs=1, lr=0.0)   # lr=0: nothing moves
    ckpt = torch.load(payload["selected_ckpt"], map_location="cpu")
    assert float(ckpt["model"]["w"]) == pytest.approx(7.5)
