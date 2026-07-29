"""[4.1] Crop augmentation — **encoder pretraining ONLY** (Inv. 13).

Two ops, both ABUS-physics-safe:

* **horizontal flip along d1** (lateral) with p = ``RESC_ENC_AUG["hflip_d1_p"]`` — the
  breast is approximately left–right symmetric.
* **centre jitter**, uniform in ``±centre_jitter_frac × side`` per axis, applied to ``cen``
  **before sampling** (so it is a genuine re-extraction, not a shift of an existing crop).

Explicitly ABSENT and never to be added: any flip along **d0** (depth/beam — skin is always
at the top and acoustic shadows always extend downward, so a d0 flip produces an
anatomically impossible frame), rotation, scale jitter, mosaic/mixup, copy-paste.

Because jitter changes the sample grid, an augmented crop cannot be produced from the
cached 48³ crop — the pretraining dataset re-extracts from the iso cache. Every other
consumer (embedding caching, set-module training, all evaluation) reads the **un-augmented**
crop cache; there is no TTA anywhere.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .. import conventions as C

__all__ = ["FLIP_AXIS", "augment_params", "jitter_centre", "maybe_flip_d1", "augmented_crop"]

#: The ONLY axis a crop may be mirrored along: d1 = lateral (``IN_PLANE_COL_AXIS``).
FLIP_AXIS = C.IN_PLANE_COL_AXIS


def augment_params() -> Dict[str, float]:
    """The two sanctioned knobs, read from :mod:`conventions` at call time."""
    return {"hflip_d1_p": float(C.RESC_ENC_AUG["hflip_d1_p"]),
            "centre_jitter_frac": float(C.RESC_ENC_AUG["centre_jitter_frac"])}


def jitter_centre(cen: Tuple[float, float, float], side: float, rng: np.random.Generator,
                  frac: Optional[float] = None) -> Tuple[float, float, float]:
    """Uniform ``±frac × side`` offset per axis applied to the ROI centre."""
    frac = augment_params()["centre_jitter_frac"] if frac is None else float(frac)
    if frac <= 0.0:
        return tuple(float(c) for c in cen)  # type: ignore[return-value]
    off = rng.uniform(-frac * float(side), frac * float(side), size=3)
    return tuple(float(cen[a] + off[a]) for a in range(3))  # type: ignore[return-value]


def maybe_flip_d1(crop: np.ndarray, rng: np.random.Generator,
                  p: Optional[float] = None) -> np.ndarray:
    """Mirror the crop along **d1** with probability ``p``. Never touches d0 or d2."""
    p = augment_params()["hflip_d1_p"] if p is None else float(p)
    if p <= 0.0 or rng.random() >= p:
        return crop
    return np.flip(crop, axis=FLIP_AXIS).copy()


def augmented_crop(vol_iso, cen_d0: float, cen_d1: float, cen_d2: float,
                   ext_d0: float, ext_d1: float, ext_d2: float,
                   rng: np.random.Generator, out: Optional[int] = None,
                   side: Optional[float] = None,
                   hflip_p: Optional[float] = None,
                   jitter_frac: Optional[float] = None) -> np.ndarray:
    """Jitter the centre, extract, then maybe flip along d1. Reduces to
    :func:`crops.extract_crop` exactly when both probabilities/fractions are 0."""
    from .crops import extract_crop, roi_side_iso

    side = roi_side_iso(ext_d0, ext_d1, ext_d2) if side is None else float(side)
    cen = jitter_centre((cen_d0, cen_d1, cen_d2), side, rng, frac=jitter_frac)
    crop = extract_crop(vol_iso, cen[0], cen[1], cen[2], ext_d0, ext_d1, ext_d2,
                        out=out, side=side)
    return maybe_flip_d1(crop, rng, p=hflip_p)
