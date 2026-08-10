"""[4.1] Crop augmentation — **encoder pretraining ONLY** (Inv. 13).

Two ops, both ABUS-physics-safe:

* **mirror flip along the LATERAL axis** = ``d0`` with p = ``RESC_ENC_AUG["mirror_lateral_p"]``
  — the breast is approximately left–right symmetric.
* **centre jitter**, uniform in ``±centre_jitter_frac × side`` per axis, applied to ``cen``
  **before sampling** (so it is a genuine re-extraction, not a shift of an existing crop).

Explicitly ABSENT and never to be added: any flip along **d1** (depth/beam — skin is always
at the top and acoustic shadows always extend downward, so a d1 flip produces an
acoustically impossible frame), any flip along d2 (sweep), rotation, scale jitter,
mosaic/mixup, copy-paste.

Because jitter changes the sample grid, an augmented crop cannot be produced from the
cached 48³ crop — the pretraining dataset re-extracts from the iso cache. Every other
consumer (embedding caching, set-module training, all evaluation) reads the **un-augmented**
crop cache; there is no TTA anywhere.

CORRECTION 2026-08-04 (Inv. 13). This module previously set ``FLIP_AXIS = C.IN_PLANE_COL_AXIS``
(= 1), whose *declared* role is "lateral" but whose *measured* role is **depth/beam**
(``results/AXIS_CHECK.md``: 129/130 volumes, four independent lines). It therefore mirrored the
beam axis — skin at the bottom, shadows pointing back at the transducer — which Inv. 13
explicitly forbids. Fixed here to the measured lateral axis, ``d0``.

Safe to change now, and only now: **no rescorer has been trained yet**, so the code and the
models it describes agree.

UPDATE 2026-08-08. The same defect in the *detector* augmentation
(``augment.TRAIN_AUGMENT["flip_stack_axis"]``) is **also corrected now** — it was held at ``1``
while 8 detectors were deployed under it, all 8 have since been retrained on ``d0`` and
PROMOTED (``runbooks/RB_FOLD_FLIP.md``), and Inv. 13 amendment (b) records it. So **both**
behavioural consumers of the inverted axis names are on the measured lateral axis and
**Inv. 13 is no longer violated anywhere.** The measured cost of the defect is in
``results/RESULTS_AUG_FLIP_AB.md`` (3 paired seeds) and ``results/RESULTS_FOLD_FLIP.md``
(5 folds + the promotion decision).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .. import conventions as C

__all__ = ["FLIP_AXIS", "augment_params", "jitter_centre", "maybe_flip_lateral",
           "maybe_flip_d1", "augmented_crop"]

#: The ONLY axis a crop may be mirrored along: **d0 = LATERAL** (measured, ``AXIS_CHECK.md``).
#:
#: Deliberately a literal, NOT ``C.IN_PLANE_ROW_AXIS``. That constant is also 0, but its
#: docstring declares it "depth/beam", so spelling the flip through it would read as a
#: request to mirror the beam axis — the exact naming failure that caused this bug. The
#: declared in-plane roles in ``conventions`` are inverted and are left inverted on purpose
#: (they feed ``preprocess_hash``; see the header comment there).
FLIP_AXIS = 0  # d0 = lateral; d1 = depth/beam (NEVER flip); d2 = sweep (NEVER flip)


def augment_params() -> Dict[str, float]:
    """The two sanctioned knobs, read from :mod:`conventions` at call time.

    ``mirror_lateral_p`` = probability of the **lateral** (``d0``) mirror. RENAMED 2026-08-09
    from ``hflip_d1_p``, which named the one axis this module must never touch; no recorded
    config changed, because no Phase-4 run had produced one.
    """
    return {"mirror_lateral_p": float(C.RESC_ENC_AUG["mirror_lateral_p"]),
            "centre_jitter_frac": float(C.RESC_ENC_AUG["centre_jitter_frac"])}


def jitter_centre(cen: Tuple[float, float, float], side: float, rng: np.random.Generator,
                  frac: Optional[float] = None) -> Tuple[float, float, float]:
    """Uniform ``±frac × side`` offset per axis applied to the ROI centre."""
    frac = augment_params()["centre_jitter_frac"] if frac is None else float(frac)
    if frac <= 0.0:
        return tuple(float(c) for c in cen)  # type: ignore[return-value]
    off = rng.uniform(-frac * float(side), frac * float(side), size=3)
    return tuple(float(cen[a] + off[a]) for a in range(3))  # type: ignore[return-value]


def maybe_flip_lateral(crop: np.ndarray, rng: np.random.Generator,
                       p: Optional[float] = None) -> np.ndarray:
    """Mirror the crop along the **lateral** axis (d0) with probability ``p``.

    Never touches **d1** (depth/beam — Inv. 13's red line) or d2 (sweep).
    """
    p = augment_params()["mirror_lateral_p"] if p is None else float(p)
    if p <= 0.0 or rng.random() >= p:
        return crop
    return np.flip(crop, axis=FLIP_AXIS).copy()


#: Deprecated alias kept so no caller breaks. The name is a historical misnomer: it mirrors
#: the LATERAL axis (d0), not d1. Prefer :func:`maybe_flip_lateral`.
maybe_flip_d1 = maybe_flip_lateral


def augmented_crop(vol_iso, cen_d0: float, cen_d1: float, cen_d2: float,
                   ext_d0: float, ext_d1: float, ext_d2: float,
                   rng: np.random.Generator, out: Optional[int] = None,
                   side: Optional[float] = None,
                   hflip_p: Optional[float] = None,
                   jitter_frac: Optional[float] = None) -> np.ndarray:
    """Jitter the centre, extract, then maybe flip along the lateral axis (d0). Reduces to
    :func:`crops.extract_crop` exactly when both probabilities/fractions are 0."""
    from .crops import extract_crop, roi_side_iso

    side = roi_side_iso(ext_d0, ext_d1, ext_d2) if side is None else float(side)
    cen = jitter_centre((cen_d0, cen_d1, cen_d2), side, rng, frac=jitter_frac)
    crop = extract_crop(vol_iso, cen[0], cen[1], cen[2], ext_d0, ext_d1, ext_d2,
                        out=out, side=side)
    return maybe_flip_lateral(crop, rng, p=hflip_p)
