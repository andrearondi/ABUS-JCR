"""ABUS-physics augmentation policy (Inv. 13). Written now, exercised in Phase 2.

The depth/beam axis is physically privileged: skin is always at the top and
acoustic shadows always extend downward along ``d0``. The augmentation policy
must not manufacture anatomically impossible frames.

**OFF (forbidden):**
- vertical flip — would put skin at the bottom / flip shadows upward (``d0`` axis).
- large rotation — same reason; small in-plane jitter only.
- mosaic / mixup — splice multiple slices into impossible frames. Ultralytics
  enables mosaic by DEFAULT; it MUST be disabled in the Phase-6 YOLO config.

**ON (allowed):**
- horizontal flip — lateral ``d1`` axis, approx. left-right symmetry.
- grayscale-appropriate intensity jitter (brightness/contrast), mild Gaussian
  blur/noise, small in-plane translations.

**Consistency:** every spatial op shares ONE sampled parameter set across the C
channel-slices (same crop/flip/translate for all) or the 2.5D channels
desynchronise. Augmentation is **training-only** — candidate generation and
rescorer feature extraction run without augmentation (no TTA by default).
"""

from __future__ import annotations

import numpy as np

from . import conventions as C

# The frozen policy spec (a plain dict so Phase 2 / Phase 6 configs read it
# directly). Booleans are explicit so tests can assert each forbidden op is off.
TRAIN_AUGMENT = {
    # forbidden by ABUS physics
    "vertical_flip": False,          # d0 = depth: skin top, shadows downward
    "large_rotation": False,
    "mosaic": False,                 # Ultralytics default — must stay off (Phase 6)
    "mixup": False,
    # allowed
    "horizontal_flip": True,         # d1 = lateral, ~L-R symmetry
    "horizontal_flip_p": 0.5,
    "intensity_jitter": True,        # grayscale brightness/contrast
    "brightness_contrast_limit": 0.2,
    "gaussian_blur": True,
    "gaussian_noise": True,
    "small_translation": True,
    "translate_frac": 0.0625,        # small in-plane shift
    "rotation_deg": 5.0,             # small only (large rotation forbidden)
    # inference
    "tta": False,                    # no test-time augmentation by default
    # 2.5D consistency contract
    "spatial_ops_shared_across_channels": True,
}


def hflip_stack(stack: np.ndarray) -> np.ndarray:
    """Horizontal (lateral ``d1``) flip applied identically across all C channels.

    ``stack`` is ``(C, d0, d1)``; the flip is along the last axis (lateral), never
    ``d0`` (depth). Applied to every channel with the same operation so the 2.5D
    channels stay synchronised (Inv. 13).
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"expected (C, d0, d1), got shape {stack.shape}")
    return stack[:, :, ::-1].copy()
