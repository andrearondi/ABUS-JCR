"""Data-driven GT double-check: the whole-mask box must reproduce the CSV box
with **0** residual on every available case.

Split root resolution:
- ``ABUS_SPLIT_ROOT`` env var if set (the server runbook points this at Train);
- else the local Validation split from LOCAL_DATA_LAYOUT.md.

If neither exists, the whole module skips (e.g. CI without data). Locally this
runs 30/30 Validation cases; on the server it runs all 100 Train cases.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from abus_jcr.io_nrrd import discover_cases, load_array
from abus_jcr.gt_labels import load_gt_documented, to_official_gt, doublecheck_case

_LOCAL_VALIDATION = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")


def _split_root() -> Path | None:
    env = os.environ.get("ABUS_SPLIT_ROOT")
    if env:
        return Path(env)
    if _LOCAL_VALIDATION.exists():
        return _LOCAL_VALIDATION
    return None


_ROOT = _split_root()
if _ROOT is None or not _ROOT.exists():
    pytest.skip(
        "no ABUS split root available (set ABUS_SPLIT_ROOT or provide local Validation)",
        allow_module_level=True,
    )

_CASES = discover_cases(_ROOT)
_GT = to_official_gt(load_gt_documented(_ROOT / "bbx_labels.csv")).set_index("public_id")


@pytest.mark.parametrize("case_id", sorted(_CASES.keys()))
def test_gt_box_doublecheck_zero_residual(case_id):
    assert case_id in _GT.index, f"case {case_id} has no GT row"
    mask, _ = load_array(_CASES[case_id].mask)
    gt_row = _GT.loc[case_id]
    residual = doublecheck_case(np.asarray(mask), gt_row)
    assert max(residual.values()) == 0, f"case {case_id} residual {residual}"
