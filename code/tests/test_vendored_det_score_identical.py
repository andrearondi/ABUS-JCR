"""The vendored oracle must stay byte-identical to the official code.

This is a *drift guard*: it protects against the vendored copy
``abus_jcr/eval/_official_det_score.py`` silently diverging from its upstream
source ``Final_Evaluation/det_score.py`` (Inv. 3 — we score with the exact
challenge code, never a re-derivation).

The guard is only meaningful where the upstream original coexists with the
copy — i.e. the full local repo / CI. The **server ships only the ``code/``
repo**, so the upstream file is legitimately absent there; in that case the
vendored copy *is* the authority and the test skips (nothing to diff against).
When the original is present, byte-identity is enforced hard.
"""

from pathlib import Path

import pytest

import abus_jcr.eval as _eval_pkg

# Anchor the vendored copy to the actually-imported package, not a guessed root.
_VENDORED = Path(_eval_pkg.__file__).resolve().parent / "_official_det_score.py"


def _find_official() -> Path | None:
    """Locate ``Final_Evaluation/det_score.py`` by walking up from the vendored
    copy. Returns None if it is not shipped (e.g. the server code-only repo)."""
    for parent in _VENDORED.parents:
        candidate = parent / "Final_Evaluation" / "det_score.py"
        if candidate.exists():
            return candidate
    return None


def test_vendored_det_score_is_byte_identical():
    assert _VENDORED.exists(), f"vendored oracle missing: {_VENDORED}"
    official = _find_official()
    if official is None:
        pytest.skip(
            "upstream Final_Evaluation/det_score.py not present (code-only repo); "
            "the vendored copy is the authority here — drift is checked in the full repo/CI"
        )
    assert _VENDORED.read_bytes() == official.read_bytes(), (
        "Vendored det_score.py has drifted from Final_Evaluation/det_score.py. "
        "Re-vendor byte-identically; never edit the copy."
    )
