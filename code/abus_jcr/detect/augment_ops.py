"""Inv.-13 detection augmentation: box-aware, channel-consistent (Phase 2).

Operates on a numpy 2.5D stack ``(C, d0, d1)`` and half-open boxes ``(N, 4)`` in
``(x1, y1, x2, y2)`` = ``(d1, d0, d1, d0)`` order, matching the schema. Every
**spatial** op samples ONE parameter set and applies it identically to all C
channel-slices (or the channels desynchronise) and transforms the boxes with the
image; **intensity** ops are grayscale and leave boxes untouched. Training-only.

**Forbidden (Inv. 13):** vertical flip, rotation, mosaic, mixup — never applied,
and enabling any of them in the policy raises. Torch-free.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage

from .. import conventions as C
from ..augment import TRAIN_AUGMENT

OnOp = Optional[Callable[[str, dict], None]]

# policy flags that must never be enabled for ABUS physics (Inv. 13).
_FORBIDDEN_FLAGS = ("vertical_flip", "large_rotation", "mosaic", "mixup")


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer translate a single ``(d0, d1)`` frame by ``(dy, dx)``, zero-filled.

    ``dy`` shifts along ``d0`` (depth/row), ``dx`` along ``d1`` (lateral/col).
    Content shifted out of frame is dropped; vacated pixels are 0.
    """
    out = np.zeros_like(frame)
    h, w = frame.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    yr0, yr1 = max(0, -dy), min(h, h - dy)
    xr0, xr1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = frame[yr0:yr1, xr0:xr1]
    return out


def scale_frame(frame: np.ndarray, s: float) -> np.ndarray:
    """Zoom a single ``(d0, d1)`` frame about its centre by factor ``s`` (>0), zero-fill.

    ``s < 1`` zooms out (content shrinks, borders zero-padded); ``s > 1`` zooms in
    (content magnified, edges cropped). Output keeps the input shape. Uses the same
    centre convention (``h/2, w/2``) as :func:`_scale_boxes` so boxes track content.
    """
    h, w = frame.shape
    cy, cx = h / 2.0, w / 2.0
    matrix = np.array([[1.0 / s, 0.0], [0.0, 1.0 / s]], dtype=float)
    offset = np.array([cy - cy / s, cx - cx / s], dtype=float)
    return ndimage.affine_transform(frame, matrix, offset=offset, order=1,
                                    mode="constant", cval=0.0).astype(frame.dtype, copy=False)


def rotate_frame(frame: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a single ``(d0, d1)`` frame by ``deg`` about its centre, keep shape, zero-fill."""
    return ndimage.rotate(frame, float(deg), reshape=False, order=1,
                          mode="constant", cval=0.0).astype(frame.dtype, copy=False)


def _scale_boxes(boxes: np.ndarray, s: float, h: int, w: int) -> np.ndarray:
    """Scale half-open boxes about the frame centre by ``s``, clip, drop the vanished."""
    if len(boxes) == 0:
        return boxes
    cx, cy = w / 2.0, h / 2.0
    out = boxes.copy()
    out[:, [0, 2]] = cx + (boxes[:, [0, 2]] - cx) * s
    out[:, [1, 3]] = cy + (boxes[:, [1, 3]] - cy) * s
    np.clip(out[:, [0, 2]], 0, w, out=out[:, [0, 2]])
    np.clip(out[:, [1, 3]], 0, h, out=out[:, [1, 3]])
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out[keep]


def _rotate_boxes(boxes: np.ndarray, deg: float, h: int, w: int) -> np.ndarray:
    """Rotate box corners about the centre, take the axis-aligned hull, clip, drop vanished.

    Small angles only (Inv. 13) keep the hull tight. Matches the forward image map of
    :func:`rotate_frame` (content at ``p`` appears at ``R·(p-c)+c``).
    """
    if len(boxes) == 0:
        return boxes
    cx, cy = w / 2.0, h / 2.0
    th = math.radians(float(deg))
    cos, sin = math.cos(th), math.sin(th)
    out = np.empty_like(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        xs = np.array([x1, x2, x1, x2], dtype=float) - cx
        ys = np.array([y1, y1, y2, y2], dtype=float) - cy
        rx = cos * xs - sin * ys + cx
        ry = sin * xs + cos * ys + cy
        out[i] = [rx.min(), ry.min(), rx.max(), ry.max()]
    np.clip(out[:, [0, 2]], 0, w, out=out[:, [0, 2]])
    np.clip(out[:, [1, 3]], 0, h, out=out[:, [1, 3]])
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out[keep]


def _hflip_boxes(boxes: np.ndarray, w: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, 0] = w - boxes[:, 2]  # x1' = W - x2
    out[:, 2] = w - boxes[:, 0]  # x2' = W - x1
    return out


def _shift_boxes(boxes: np.ndarray, dy: int, dx: int, h: int, w: int) -> np.ndarray:
    """Shift half-open boxes by ``(dx, dy)``, clip to the frame, drop the vanished."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] += dx
    out[:, [1, 3]] += dy
    np.clip(out[:, [0, 2]], 0, w, out=out[:, [0, 2]])
    np.clip(out[:, [1, 3]], 0, h, out=out[:, [1, 3]])
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out[keep]


def apply_train_augment(
    stack: np.ndarray,
    boxes: np.ndarray,
    rng: np.random.Generator,
    policy: dict = TRAIN_AUGMENT,
    on_op: OnOp = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the Inv.-13 train augmentation to ``(stack, boxes)`` -> new pair.

    ``stack`` is ``(C, d0, d1)`` float; ``boxes`` is ``(N, 4)`` half-open
    ``(x1, y1, x2, y2)``. ``on_op(name, params)`` (optional) is called for each op
    actually applied — used by the invariant test to confirm forbidden ops never
    fire and spatial params are shared. Returns copies; inputs are not mutated.
    """
    for flag in _FORBIDDEN_FLAGS:
        if policy.get(flag, False):
            raise ValueError(f"Inv. 13 violation: forbidden augmentation {flag!r} is enabled")

    stack = np.array(stack, dtype=np.float32, copy=True)
    boxes = np.array(boxes, dtype=np.float32, copy=True).reshape(-1, 4)
    _, h, w = stack.shape

    def emit(name, params):
        if on_op is not None:
            on_op(name, params)

    # --- spatial: horizontal flip (lateral d1), shared across channels ---
    if float(policy.get("horizontal_flip_p", 0.0)) > 0.0 and rng.random() < policy["horizontal_flip_p"]:
        stack = stack[:, :, ::-1].copy()          # all channels, same op
        boxes = _hflip_boxes(boxes, w)
        emit("hflip", {})

    # --- spatial: small integer translation, one (dy,dx) shared across channels ---
    if policy.get("small_translation", False):
        tf = float(policy.get("translate_frac", 0.0))
        dy = int(rng.integers(-int(tf * h), int(tf * h) + 1)) if int(tf * h) > 0 else 0
        dx = int(rng.integers(-int(tf * w), int(tf * w) + 1)) if int(tf * w) > 0 else 0
        if dy != 0 or dx != 0:
            stack = np.stack([shift_frame(stack[c], dy, dx) for c in range(stack.shape[0])], axis=0)
            boxes = _shift_boxes(boxes, dy, dx, h, w)
            emit("translate", {"dy": dy, "dx": dx})

    # --- spatial: isotropic zoom about centre, one s shared across channels (Inv. 13) ---
    if policy.get("scale_zoom", False):
        lo, hi = policy.get("scale_range", [1.0, 1.0])
        s = float(rng.uniform(float(lo), float(hi)))
        if s != 1.0:
            stack = np.stack([scale_frame(stack[c], s) for c in range(stack.shape[0])], axis=0)
            boxes = _scale_boxes(boxes, s, h, w)
            emit("scale", {"s": s})

    # --- spatial: small in-plane rotation, one angle shared across channels (Inv. 13) ---
    if policy.get("rotation", False):
        deg = float(rng.uniform(-float(policy.get("rotation_deg", 0.0)),
                                 float(policy.get("rotation_deg", 0.0))))
        if deg != 0.0:
            stack = np.stack([rotate_frame(stack[c], deg) for c in range(stack.shape[0])], axis=0)
            boxes = _rotate_boxes(boxes, deg, h, w)
            emit("rotate", {"deg": deg})

    # --- intensity: grayscale, identical across channels, boxes untouched ---
    if policy.get("intensity_jitter", False) and rng.random() < 0.5:
        lim = float(policy.get("brightness_contrast_limit", 0.2))
        brightness = float(rng.uniform(-lim, lim))
        contrast = float(rng.uniform(1.0 - lim, 1.0 + lim))
        mean = float(stack.mean())
        stack = np.clip((stack - mean) * contrast + mean + brightness, 0.0, 1.0).astype(np.float32)
        emit("intensity", {"brightness": brightness, "contrast": contrast})

    if policy.get("gaussian_blur", False) and rng.random() < 0.5:
        sigma = float(rng.uniform(0.0, 1.0))
        stack = np.stack([ndimage.gaussian_filter(stack[c], sigma) for c in range(stack.shape[0])], axis=0)
        emit("blur", {"sigma": sigma})

    if policy.get("gaussian_noise", False) and rng.random() < 0.5:
        sigma = float(rng.uniform(0.0, 0.05))
        noise = rng.normal(0.0, sigma, size=stack.shape[1:]).astype(np.float32)  # one field, all channels
        stack = np.clip(stack + noise[None, :, :], 0.0, 1.0).astype(np.float32)
        emit("noise", {"sigma": sigma})

    return stack, boxes
