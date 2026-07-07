"""The vendored oracle must stay byte-identical to the official code.

Any drift from ``Final_Evaluation/det_score.py`` fails loudly here — the whole
point of vendoring is that we score with the *exact* challenge code, never a
re-derivation (Inv. 3).
"""

from pathlib import Path

# tests/ -> code/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _REPO_ROOT / "Final_Evaluation" / "det_score.py"
_VENDORED = _REPO_ROOT / "code" / "abus_jcr" / "eval" / "_official_det_score.py"


def test_vendored_det_score_is_byte_identical():
    assert _OFFICIAL.exists(), f"official oracle missing: {_OFFICIAL}"
    assert _VENDORED.exists(), f"vendored oracle missing: {_VENDORED}"
    official_bytes = _OFFICIAL.read_bytes()
    vendored_bytes = _VENDORED.read_bytes()
    assert vendored_bytes == official_bytes, (
        "Vendored det_score.py has drifted from Final_Evaluation/det_score.py. "
        "Re-vendor byte-identically; never edit the copy."
    )
