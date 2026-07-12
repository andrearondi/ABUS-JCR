"""Shadow-aware lesion copy-paste (Inv. 13 amended, P2-UPDATE — DEFAULT-OFF).

Multiplies positive instances per image (attacks positive-scarcity + small-lesion
recall) WITHOUT breaking ABUS physics: a lesion is always pasted **with its
posterior-shadow column** and kept in its original **depth band** (never lifted
off / moved above its shadow). Naive copy-paste that detaches the shadow is the
forbidden variant the literature flags as harmful; this module implements only the
shadow-aware form. Off by default (``TRAIN_AUGMENT["lesion_copy_paste"] = False``)
so the primary re-run isolates the P0+P1 effect; enabled only in the Stage-3
experiment.

Torch-free (numpy); operates on a 2.5D stack ``(C, d0, d1)`` and half-open boxes,
channel-consistently. ``d0`` is depth (skin top, shadow down); ``d1`` is lateral.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def extract_lesion_crop(stack: np.ndarray, box: np.ndarray) -> Dict:
    """Crop a lesion **plus its posterior-shadow column** from ``stack``.

    ``box`` is half-open ``(x1, y1, x2, y2)`` (``x=d1``, ``y=d0``). The crop spans
    the box columns and runs from the lesion top ``y1`` **down to the frame bottom**
    (the acoustic shadow travels downward along ``d0``), across all C channels.
    Returns ``{crop (C,ch,cw), y0, lesion_h, w}``.
    """
    _, h, _ = stack.shape
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    crop = stack[:, y1:h, x1:x2].copy()
    return {"crop": crop, "y0": y1, "lesion_h": y2 - y1, "w": x2 - x1}


def build_crop_bank(stack_boxes: List) -> List[Dict]:
    """Assemble a bank from ``[(stack, boxes), ...]`` — one crop per lesion box."""
    bank: List[Dict] = []
    for stack, boxes in stack_boxes:
        for b in np.asarray(boxes).reshape(-1, 4):
            bank.append(extract_lesion_crop(stack, b))
    return bank


def paste_lesion(stack: np.ndarray, boxes: np.ndarray, crop: Dict,
                 rng: np.random.Generator, x_offset: Optional[int] = None,
                 blend: str = "max") -> tuple:
    """Paste one shadow-aware ``crop`` onto ``stack`` at a lateral offset, add its box.

    The crop keeps its **depth band** (``y0`` unchanged) — only the lateral (``d1``)
    position moves — so the lesion never leaves its shadow. ``blend='max'`` composites
    (bright lesion/echo over background). Returns ``(new_stack, new_boxes)``; a no-op
    (returns inputs) if the crop cannot fit laterally.
    """
    Cc, h, w = stack.shape
    cw = crop["w"]
    ch = crop["crop"].shape[1]
    y0 = crop["y0"]
    if w - cw < 0 or ch <= 0:
        return stack, boxes
    xnew = int(x_offset) if x_offset is not None else int(rng.integers(0, w - cw + 1))
    xnew = max(0, min(xnew, w - cw))
    ynew = y0                                  # PRESERVE depth band (shadow-aware)
    ch = min(ch, h - ynew)
    patch = crop["crop"][:, :ch, :cw]
    out = stack.copy()
    region = out[:, ynew:ynew + ch, xnew:xnew + cw]
    out[:, ynew:ynew + ch, xnew:xnew + cw] = np.maximum(region, patch) if blend == "max" else patch
    new_box = np.array([[xnew, ynew, xnew + cw, ynew + crop["lesion_h"]]], dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    new_boxes = np.concatenate([boxes, new_box], axis=0) if len(boxes) else new_box
    return out, new_boxes
