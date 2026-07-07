"""Isotropic resampling — the "cache trick" (Inv. 6).

Every volume+mask is resampled ONCE to ``ISO_SPACING_MM`` isotropic and cached
(see :mod:`abus_jcr.cache`). Detector slices, 2D boxes, linking, and rescorer
crops all live in this single space; the recorded per-volume affine maps back to
native voxel indices for official scoring only.

**Resampler choice:** ``scipy.ndimage.zoom`` (NOT MONAI) because it honours the
Phase-0 storage-order + injected-spacing contract — pynrrd returns storage order
``(d0, d1, d2)`` and spacing is the injected ``SPACING_STORAGE_MM``, not the
NRRD header's identity placeholder. ``grid_mode=True`` + ``mode="grid-constant"``
is edge-aligned so the physical extent is preserved and the output size is
``round(n_in * f)`` per axis.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from . import conventions as C

Shape3 = Tuple[int, int, int]


def zoom_factors(iso_spacing_mm: float = C.ISO_SPACING_MM) -> Tuple[float, float, float]:
    """Per-storage-axis zoom factor ``f[a] = SPACING_STORAGE_MM[a] / iso``.

    At 0.5 mm this is ``(0.146, 0.400, 0.951348)`` — depth is downsampled hardest
    (finest native spacing), the sweep axis barely at all.
    """
    return tuple(C.SPACING_STORAGE_MM[a] / iso_spacing_mm for a in range(3))  # type: ignore[return-value]


def iso_shape(native_shape: Shape3, iso_spacing_mm: float = C.ISO_SPACING_MM) -> Shape3:
    """Isotropic output shape: ``round(n_in * f)`` per axis (edge-aligned)."""
    f = zoom_factors(iso_spacing_mm)
    return tuple(int(round(native_shape[a] * f[a])) for a in range(3))  # type: ignore[return-value]


def resample_case(vol_u8: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Resample one case to isotropic space.

    ``vol_u8`` (uint8, storage order) -> ``vol_iso`` float32 in ``[0, 1]`` via
    linear interpolation of ``vol/255``. ``mask_u8`` -> ``mask_iso`` uint8 via
    nearest-neighbour so it stays ``{0, 1}``. Returns ``(vol_iso, mask_iso,
    meta)`` where ``meta`` records everything needed to invert the transform.
    """
    if vol_u8.shape != mask_u8.shape:
        raise ValueError(f"vol/mask shape mismatch: {vol_u8.shape} vs {mask_u8.shape}")
    native_shape = tuple(int(s) for s in vol_u8.shape)
    f = zoom_factors()
    div = float(C.INTENSITY_NORM["divisor"])

    vol_norm = np.asarray(vol_u8, dtype=np.float32) / np.float32(div)
    vol_iso = ndimage.zoom(
        vol_norm, f, order=C.RESAMPLE["image_order"],
        grid_mode=C.RESAMPLE["grid_mode"], mode=C.RESAMPLE["mode"],
    ).astype(np.float32)
    # linear interpolation can overshoot [0,1] by rounding; clamp to the contract.
    np.clip(vol_iso, 0.0, 1.0, out=vol_iso)

    mask_iso = ndimage.zoom(
        np.asarray(mask_u8), f, order=C.RESAMPLE["mask_order"],
        grid_mode=C.RESAMPLE["grid_mode"], mode=C.RESAMPLE["mode"],
    )
    mask_iso = (mask_iso > 0).astype(np.uint8)

    out_shape = tuple(int(s) for s in vol_iso.shape)
    meta = {
        "native_shape": list(native_shape),
        "iso_shape": list(out_shape),
        "spacing_storage_mm": list(C.SPACING_STORAGE_MM),
        "iso_spacing_mm": C.ISO_SPACING_MM,
        "zoom_factors": list(f),
        # inverse affine per axis (iso_index -> native_index): native = (iso+0.5)/f - 0.5
        "inverse_affine": {"formula": "native = (iso + 0.5) / f - 0.5", "f": list(f)},
    }
    return vol_iso, mask_iso, meta


def _canonical_cfg(iso_spacing_mm: float) -> Dict:
    """The dict the cache hash is taken over. Every entry is cache-invalidating."""
    return {
        "iso_spacing_mm": iso_spacing_mm,
        "intensity_norm": C.INTENSITY_NORM,
        "resample": C.RESAMPLE,
        "spacing_storage_mm": list(C.SPACING_STORAGE_MM),
        "perm": list(C.PERM_STORAGE_TO_ITK),
        "schema_version": 1,
    }


def preprocess_hash(iso_spacing_mm: float = C.ISO_SPACING_MM) -> str:
    """SHA-256 over the canonical preprocessing config. Names the cache dir;
    any change to a cache-invalidating input yields a new hash (never a silent
    stale reuse)."""
    blob = json.dumps(_canonical_cfg(iso_spacing_mm), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
