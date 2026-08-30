"""Script CLI path defaults must follow ``$WORK``, never a hardcoded machine.

**Why this file exists.** On 2026-08-31 `[4.2]` died several minutes into a GPU job with

    subprocess.CalledProcessError: ... '--data-root', '/home/maia-user/Andre2/data' ...

`/home/maia-user` is the *pre-migration* home. `scripts/_common.py` and both phase helpers still
carried it as the default for every root, so any runbook line that omitted a flag pointed at a
machine that no longer exists. `[4.1]` survived only because it never reads ``--data-root``.

A dead path at least fails loudly. The dangerous sibling is a path that exists but names the wrong
*arm* — CLAUDE.md's "never mix profiles inside one experiment" — which is why the resolved roots are
asserted here rather than left to a reviewer's eye.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _reload(modname: str, work):
    """Import a script helper with ``$WORK`` set (or unset) and hand back the fresh module."""
    old = os.environ.get("WORK")
    if work is None:
        os.environ.pop("WORK", None)
    else:
        os.environ["WORK"] = work
    try:
        mod = importlib.import_module(modname)
        return importlib.reload(mod)
    finally:
        if old is None:
            os.environ.pop("WORK", None)
        else:
            os.environ["WORK"] = old


WORK = "/proj/berzbiomedicalimagingkth/users/x_andro/Andre2"


@pytest.mark.parametrize("modname,attr,rel", [
    ("_phase4_common", "DEFAULT_DATA_ROOT",   "data"),
    ("_phase4_common", "DEFAULT_PHASE1_OUT",  "outputs_iso/phase1"),
    ("_phase4_common", "DEFAULT_PHASE3_OUT",  "outputs_iso/phase3"),
    ("_phase4_common", "DEFAULT_PHASE4_OUT",  "outputs_iso/phase4"),
    ("_phase3_common", "DEFAULT_DATA_ROOT",   "data"),
    ("_phase3_common", "DEFAULT_PHASE1_OUT",  "outputs/phase1"),
    ("_phase3_common", "DEFAULT_PHASE2_OUT",  "outputs/phase2"),
    ("_phase3_common", "DEFAULT_PHASE3_OUT",  "outputs/phase3"),
    ("_common",        "DEFAULT_DATA_ROOT",   "data"),
])
def test_default_roots_follow_work(modname, attr, rel):
    mod = _reload(modname, WORK)
    assert getattr(mod, attr) == f"{WORK}/{rel}"


@pytest.mark.parametrize("modname", ["_phase4_common", "_phase3_common", "_common"])
def test_no_default_names_a_decommissioned_machine(modname):
    """The literal that caused the failure must not survive anywhere in a default."""
    mod = _reload(modname, WORK)
    for name in dir(mod):
        if name.startswith("DEFAULT_"):
            assert "maia-user" not in str(getattr(mod, name)), f"{modname}.{name} is stale"


@pytest.mark.parametrize("modname", ["_phase4_common", "_phase3_common", "_common"])
def test_unset_work_yields_a_default_that_names_its_own_fix(modname):
    """No ``$WORK``, no guessing. The placeholder has to say what to do rather than silently
    resolve to somebody's old home directory."""
    mod = _reload(modname, None)
    for name in dir(mod):
        if name.startswith("DEFAULT_"):
            val = str(getattr(mod, name))
            assert "maia-user" not in val
            assert "WORK" in val, f"{modname}.{name} = {val!r} does not name the fix"


def test_phase4_argparse_actually_uses_the_resolved_default():
    """The constant being right is not enough; the parser has to hand it to the script."""
    import argparse
    mod = _reload("_phase4_common", WORK)
    p = argparse.ArgumentParser()
    mod.add_phase4_paths(p)
    args = p.parse_args([])
    assert args.data_root == f"{WORK}/data"
    assert args.phase1_out == f"{WORK}/outputs_iso/phase1"
    assert args.phase3_out == f"{WORK}/outputs_iso/phase3"
    assert args.out_root == f"{WORK}/outputs_iso/phase4"
