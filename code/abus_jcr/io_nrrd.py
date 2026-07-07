"""NRRD reading, case discovery, and spacing injection.

Reader choice: **pynrrd**. It returns the array in documented storage order
``(d0, d1, d2)`` with no silent reversal (unlike ``SimpleITK.GetArrayFromImage``,
which reverses to ``(z, y, x)``) and exposes the raw header, so the
identity-spacing placeholder is visible. Phase 1 may adopt MONAI for resampling
*iff* its loader passes the same storage-order + injected-spacing contract test.

**Spacing:** the NRRD header carries an identity-matrix placeholder for spacing
and MUST be ignored. Physical spacing is the injected constant
``SPACING_STORAGE_MM`` from the official challenge description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import nrrd

from .conventions import SPACING_STORAGE_MM

_CASE_ID_RE = re.compile(r"_(\d+)\.nrrd$")


@dataclass(frozen=True)
class CasePaths:
    case_id: int
    data: Path
    mask: Path


def _parse_case_id(path: Path) -> int:
    m = _CASE_ID_RE.search(path.name)
    if m is None:
        raise ValueError(f"cannot parse integer case_id from {path.name}")
    return int(m.group(1))


def discover_cases(split_root: Path) -> Dict[int, CasePaths]:
    """Pair DATA/MASK files by parsed integer ``case_id``.

    Uses a **recursive** ``rglob`` under ``<split>/DATA/`` so the nested Train
    shard layout (``DATA00_49/`` …) is found; a non-recursive glob silently
    returns nothing there. Matching keys on the parsed integer id, never on
    filename padding width. Raises on any unmatched DATA or MASK.
    """
    split_root = Path(split_root)
    data_dir = split_root / "DATA"
    mask_dir = split_root / "MASK"

    data_by_id: Dict[int, Path] = {}
    for p in data_dir.rglob("DATA_*.nrrd"):
        cid = _parse_case_id(p)
        if cid in data_by_id:
            raise ValueError(f"duplicate DATA case_id {cid}: {p} and {data_by_id[cid]}")
        data_by_id[cid] = p

    mask_by_id: Dict[int, Path] = {}
    for p in mask_dir.rglob("MASK_*.nrrd"):
        cid = _parse_case_id(p)
        if cid in mask_by_id:
            raise ValueError(f"duplicate MASK case_id {cid}: {p} and {mask_by_id[cid]}")
        mask_by_id[cid] = p

    data_ids = set(data_by_id)
    mask_ids = set(mask_by_id)
    if data_ids != mask_ids:
        only_data = sorted(data_ids - mask_ids)
        only_mask = sorted(mask_ids - data_ids)
        raise ValueError(
            f"unmatched DATA/MASK in {split_root}: DATA-only={only_data}, MASK-only={only_mask}"
        )
    if not data_ids:
        raise FileNotFoundError(f"no DATA_*.nrrd found under {data_dir}")

    return {cid: CasePaths(cid, data_by_id[cid], mask_by_id[cid]) for cid in sorted(data_ids)}


def load_array(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a NRRD, returning the array in storage order ``(d0, d1, d2)`` and
    the raw header. Contract: ``array.shape == tuple(header["sizes"])``."""
    array, header = nrrd.read(str(path))
    return array, header


def read_shape(path: Path) -> Tuple[int, int, int]:
    """Header-only shape (no decompress of the voxel data)."""
    header = nrrd.read_header(str(path))
    sizes = tuple(int(s) for s in header["sizes"])
    return sizes  # type: ignore[return-value]


def injected_spacing_storage() -> Tuple[float, float, float]:
    """Physical spacing in storage order (mm). The NRRD header's identity
    placeholder is ignored — always use this constant."""
    return SPACING_STORAGE_MM
