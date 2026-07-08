"""Reusable per-volume inference -> common detection schema (provisions Phase 3).

``run_detector_on_volume`` is called by Phase 2's diagnostic dump with the
permissive ``DET_DIAG_*`` knobs and by **Phase 3** unchanged at the
recall-saturating operating point with loosened NMS — the operating point is a
caller argument, never baked in here. No augmentation (Inv. 13). Torch is imported
lazily so importing this module never requires it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import conventions as C
from .. import cache as K
from ..slice_dataset import get_stack
from . import schema as S


def run_detector_on_volume(
    model,
    cache_root,
    volume_id: int,
    score_thresh: float,
    nms_thresh: float,
    detections_per_img: int,
    batch_size: int = C.DET_BATCH_SIZE,
    stack_fn=get_stack,
    device=None,
) -> pd.DataFrame:
    """Run ``model`` over every ``SLICE_AXIS`` slice of ``volume_id`` -> detection DF.

    Boxes come back in the iso-slice pixel frame (``x = d1``, ``y = d0``;
    torchvision inverts its internal resize), matching the schema directly.
    Returns a ``DETECTION_COLUMNS`` frame validated by :func:`schema.validate_detections`.
    """
    import torch

    model.eval()
    model.score_thresh = float(score_thresh)
    model.nms_thresh = float(nms_thresh)
    model.detections_per_img = int(detections_per_img)
    if device is None:
        device = next(model.parameters()).device

    n = int(K.read_meta(cache_root, volume_id)["iso_shape"][C.SLICE_AXIS])
    rows = []
    with torch.no_grad():
        for z0 in range(0, n, batch_size):
            zs = list(range(z0, min(z0 + batch_size, n)))
            imgs = [
                torch.as_tensor(np.ascontiguousarray(stack_fn(cache_root, volume_id, z)),
                                dtype=torch.float32).to(device)
                for z in zs
            ]
            outs = model(imgs)
            for z, out in zip(zs, outs):
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                for (x1, y1, x2, y2), sc in zip(boxes, scores):
                    rows.append({
                        "volume_id": int(volume_id), "slice_z": int(z),
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "score": float(sc),
                    })

    if not rows:
        return S.empty_detections()
    df = pd.DataFrame(rows, columns=S.DETECTION_COLUMNS)
    df["volume_id"] = df["volume_id"].astype("int64")
    df["slice_z"] = df["slice_z"].astype("int64")
    return S.validate_detections(df)
