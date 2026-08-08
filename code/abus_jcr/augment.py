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
    "horizontal_flip": True,         # the L-R-symmetry flip; see flip_stack_axis for WHICH axis
    "horizontal_flip_p": 0.5,
    # WHICH in-plane axis of the (C, d0, d1) stack the allowed flip acts on.
    #   0 = d0 = the MEASURED LATERAL axis  <- DEPLOYED DEFAULT since 2026-08-08.
    #   1 = d1 = the MEASURED DEPTH/BEAM axis (Inv. 13 forbids flipping it).
    # The names above ("horizontal" / "vertical") describe IMAGE geometry and were attached to the
    # WRONG physical axis: d1 is depth/beam and d0 is lateral (measured four independent ways on
    # 129/130 volumes -- results/AXIS_CHECK.md, RESULTS_INTENSITY_PROBE.md [I.3]).
    #
    # [CORRECTED 2026-08-08, Inv. 13 amendment (b)] Was 1 until this date, i.e. every detector
    # trained before it had a 50% chance of a DEPTH flip -- skin at the bottom, shadows pointing back
    # at the transducer -- which Inv. 13 explicitly forbids, while the genuinely symmetry-preserving
    # lateral flip was switched off. It was held at 1 while those 8 detectors were deployed, so the
    # code would not describe models that were never trained that way. All 8 have now been retrained
    # on axis 0 and PROMOTED (runbooks/RB_FOLD_FLIP.md), so code and models agree again.
    #
    # Measured cost of the defect (RESULTS_AUG_FLIP_AB.md, 3 paired seeds; RESULTS_FOLD_FLIP.md,
    # 5 folds): correcting it gives linked val CPM 0.5567 -> 0.6327, a 14% smaller val pool, HALF the
    # distinct FP locations, +11/100 TP-bearing training volumes, and a monotone recall-vs-threshold
    # curve where the old one was pathological inside its own deployment region -- at a cost of 1
    # lesion of recall ceiling (78/90 -> 77/90), inside the ~2.4-lesion sampling error of that total.
    # The pre-2026-08-08 arm is the ARCHIVED control; `--flip-stack-axis 1 --run-suffix <sfx>`
    # reproduces it (phase2_train_retinanet.py refuses a non-default axis without a suffix).
    "flip_stack_axis": 0,
    "intensity_jitter": True,        # grayscale brightness/contrast
    "brightness_contrast_limit": 0.2,
    "gaussian_blur": True,
    "gaussian_noise": True,
    "small_translation": True,
    "translate_frac": 0.0625,        # small in-plane shift
    # [P2-UPDATE B4] scale/zoom — the highest-value missing detector aug (size robustness).
    "scale_zoom": True,
    "scale_range": [0.8, 1.2],       # isotropic zoom about the frame centre
    # [P2-UPDATE B4] small in-plane rotation — now WIRED (was a dead flag). Large rotation forbidden.
    "rotation": True,
    "rotation_deg": 5.0,             # small only (large rotation forbidden)
    # [P2-UPDATE P2] shadow-aware lesion copy-paste — DEFAULT OFF (Stage-3 experiment only);
    # when on, lesions are pasted WITH their posterior-shadow column (naive copy-paste stays forbidden).
    "lesion_copy_paste": False,
    "copy_paste_p": 0.5,             # per-sample paste probability when enabled
    # inference
    "tta": False,                    # no test-time augmentation by default
    # 2.5D consistency contract
    "spatial_ops_shared_across_channels": True,
}


def hflip_stack(stack: np.ndarray, axis: int = None) -> np.ndarray:
    """Mirror a ``(C, d0, d1)`` stack along one in-plane axis, identically across channels.

    ``axis`` is the STACK axis to mirror: 0 = ``d0`` (measured LATERAL — the legitimate flip),
    1 = ``d1`` (measured DEPTH/BEAM — forbidden by Inv. 13). It defaults to
    ``TRAIN_AUGMENT["flip_stack_axis"]`` (**0 since 2026-08-08**; it was 1 before, which is the
    Inv.-13 defect described there). Whichever is chosen, the same operation is applied to every
    channel so the 2.5D channels stay synchronised (Inv. 13).
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"expected (C, d0, d1), got shape {stack.shape}")
    a = TRAIN_AUGMENT["flip_stack_axis"] if axis is None else int(axis)
    if a not in (0, 1):
        raise ValueError(f"flip axis must be 0 (d0) or 1 (d1), got {a!r}")
    return (stack[:, :, ::-1] if a == 1 else stack[:, ::-1, :]).copy()
