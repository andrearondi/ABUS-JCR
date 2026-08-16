"""Seed BEFORE constructing the model — the `[4.2c]` reproducibility defect.

Measured: `[4.2b]` and `[4.2c]` scored `g2_a0.25_hard_cand` on seed 0 at **0.5796** and **0.6006**
on identical data (+0.0210). The other three cells reproduced to 4 dp. Cause — the model was built
in the caller, *before* ``train_set_variant`` called ``seed_everything``, so the **first** cell's
weight init inherited whatever global RNG state the reference block happened to leave, and that
block changed between the runs (B0-spread's bootstrap draws 200 -> 0). Every later cell is built
after a ``seed_everything`` plus deterministic training, which is exactly why only the first moved.

0.021 CPM is the same size as the weaker factor contrasts this study is meant to resolve, and the
identical pattern sits in ``run_variant_trial`` — so it would have hit **[4.5], [4.6] and [4.8]**,
not just the study.

The fix removes the ordering rather than pinning it: pass a **factory** and let the trainer seed
first. A plain model is still accepted so nothing breaks, but it cannot be made safe — the init has
already happened — and that is asserted here too.
"""

import pytest

from abus_jcr.rescore.train import resolve_model


class _Recorder:
    def __init__(self):
        self.calls = []

    def seed(self, s):
        self.calls.append(("seed", s))

    def build(self):
        self.calls.append(("build", None))
        return "model"


def test_the_factory_runs_after_the_seed():
    r = _Recorder()
    assert resolve_model(r.build, seed=7, seed_fn=r.seed) == "model"
    assert r.calls == [("seed", 7), ("build", None)]


def test_the_seed_is_the_one_passed():
    r = _Recorder()
    resolve_model(r.build, seed=2, seed_fn=r.seed)
    assert r.calls[0] == ("seed", 2)


def test_two_calls_with_different_prior_state_build_identically():
    """What the defect actually broke: the factory must see the same state every time."""
    seen = []

    def seed_fn(s):
        seen.append(("seed", s))

    def factory():
        seen.append(("build", len(seen)))
        return object()

    resolve_model(factory, seed=0, seed_fn=seed_fn)
    before = seen[-1]
    seen.clear()
    seed_fn("noise from an earlier step")      # stand-in for the reference block's RNG use
    seen.clear()
    resolve_model(factory, seed=0, seed_fn=seed_fn)
    assert seen[-1] == before


def test_a_prebuilt_model_is_still_accepted_but_is_seeded_too():
    """Back-compatible. The init already happened, so this cannot be made reproducible —
    every in-repo caller passes a factory."""
    r = _Recorder()
    sentinel = object()
    assert resolve_model(sentinel, seed=1, seed_fn=r.seed) is sentinel
    assert r.calls == [("seed", 1)], "seeding must still happen for the training loop itself"


def test_default_seed_fn_is_the_project_wide_one():
    """Not a private copy: cudnn determinism and the python/numpy/torch seeds all come from
    detect.train.seed_everything, so the rescorer cannot drift from the detector."""
    pytest.importorskip("torch")                   # the real seed_everything needs torch
    from abus_jcr.detect.train import seed_everything

    calls = []
    orig = seed_everything

    class _Probe:
        def __call__(self, s):
            calls.append(s)
            return orig(s)

    import abus_jcr.detect.train as dt
    dt.seed_everything = _Probe()
    try:
        resolve_model(lambda: "m", seed=3)
    finally:
        dt.seed_everything = orig
    assert calls == [3]
