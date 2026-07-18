"""ATSS adaptive anchor assignment for RetinaNet (P3-UPDATE D6; CONDITIONAL, default-OFF).

Fires only when ``conventions.DET_ASSIGNER == "atss"`` — the default ``"fixed"`` path (the
torchvision IoU ``Matcher`` with ``fg=0.4/bg=0.3``) is completely untouched otherwise. ATSS
(Zhang et al., CVPR 2020, arXiv:1912.02424 — the paper torchvision's v2 *cites* but does not
implement) replaces the fixed IoU thresholds with a per-GT adaptive threshold derived from
that GT's own candidate-anchor IoU statistics. Motivation for this dataset: ABUS lesions are a
narrow, non-COCO size distribution; a global IoU cutoff can starve small lesions of positives
(the D0 gate measures this). ATSS gives RetinaNet +2.3 AP on COCO with the largest gain on
small objects and is ~hyperparameter-free (k=9, flat over k=7..19); nnDetection ran BCE+ATSS
on this exact dataset (T3, 0.7704).

The correctness-critical assignment is a **pure numpy** function (``atss_match``) so it is
unit-tested on the laptop without torch; the thin ``RetinaNetATSS`` subclass only marshals
tensors to/from it and overrides ``compute_loss`` (reusing ``head.compute_loss`` unchanged).

``atss_match`` returns torchvision ``Matcher`` conventions: per-anchor matched-GT index, with
``-1`` = background and ``-2`` = ignore (``between_thresholds``). ``RetinaNetHead.compute_loss``
treats ``>= 0`` as foreground matched to that GT, ``== -1`` as background, ``== -2`` as ignored.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

BELOW_LOW_THRESHOLD = -1   # torchvision Matcher.BELOW_LOW_THRESHOLD (background)
BETWEEN_THRESHOLDS = -2    # torchvision Matcher.BETWEEN_THRESHOLDS (ignore)


def _centres(boxes: np.ndarray) -> np.ndarray:
    """(N,4) x1y1x2y2 -> (N,2) centres."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def _iou_matrix(gt: np.ndarray, anc: np.ndarray) -> np.ndarray:
    """(G,4) x (A,4) -> (G,A) IoU."""
    lt = np.maximum(gt[:, None, :2], anc[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], anc[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ag = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))[:, None]
    aa = ((anc[:, 2] - anc[:, 0]) * (anc[:, 3] - anc[:, 1]))[None, :]
    union = ag + aa - inter
    return np.where(union > 0, inter / union, 0.0)


def atss_match(gt_boxes: np.ndarray, anchors: np.ndarray,
               num_anchors_per_level: Sequence[int], topk: int) -> np.ndarray:
    """ATSS assignment -> per-anchor matched-GT index (Matcher conventions).

    Per GT ``g``: take the ``topk`` anchors *per FPN level* closest to ``g``'s centre (L2);
    let ``t_g = mean(IoU of candidates) + std(IoU of candidates)``; positives = candidates
    with ``IoU >= t_g`` AND centre inside ``g``. An anchor claimed by several GTs goes to the
    highest-IoU one. Non-positive anchors are background (``-1``). (No ignore band is produced
    by ATSS itself; ``-2`` is reserved for parity with the fixed matcher's API.)

    ``anchors`` are ``(A,4)`` x1y1x2y2 in one image's frame, ordered level-major to match
    ``num_anchors_per_level`` (sum == A). ``gt_boxes`` is ``(G,4)``. Returns ``(A,)`` int64.
    """
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    A = len(anchors)
    if len(gt_boxes) == 0:
        return np.full((A,), BELOW_LOW_THRESHOLD, dtype=np.int64)
    assert int(sum(num_anchors_per_level)) == A, \
        f"num_anchors_per_level sums to {sum(num_anchors_per_level)} != {A} anchors"

    anc_c = _centres(anchors)
    gt_c = _centres(gt_boxes)
    ious = _iou_matrix(gt_boxes, anchors)                 # (G, A)
    # per (gt, anchor) centre distance
    dist = np.linalg.norm(gt_c[:, None, :] - anc_c[None, :, :], axis=2)  # (G, A)

    G = len(gt_boxes)
    best_iou = np.full((A,), -1.0)
    matched = np.full((A,), BELOW_LOW_THRESHOLD, dtype=np.int64)

    for g in range(G):
        # candidate anchors: topk nearest per level
        cand = []
        start = 0
        for n_lvl in num_anchors_per_level:
            end = start + int(n_lvl)
            d_lvl = dist[g, start:end]
            k = min(int(topk), len(d_lvl))
            if k > 0:
                idx_lvl = np.argpartition(d_lvl, k - 1)[:k]
                cand.append(idx_lvl + start)
            start = end
        if not cand:
            continue
        cand = np.concatenate(cand)
        cand_iou = ious[g, cand]
        thr = cand_iou.mean() + cand_iou.std()            # t_g = mean + std
        # centre-inside constraint: anchor centre within gt box
        ac = anc_c[cand]
        inside = ((ac[:, 0] >= gt_boxes[g, 0]) & (ac[:, 0] <= gt_boxes[g, 2]) &
                  (ac[:, 1] >= gt_boxes[g, 1]) & (ac[:, 1] <= gt_boxes[g, 3]))
        pos = cand[(cand_iou >= thr) & inside]
        for a in pos:
            if ious[g, a] > best_iou[a]:
                best_iou[a] = ious[g, a]
                matched[a] = g
    return matched


def build_retinanet_atss(base_model):
    """Wrap a built RetinaNet so its loss uses ATSS assignment (torch; flag-gated caller).

    Re-parents ``base_model`` to a ``RetinaNetATSS`` subclass (same weights/modules) and wraps
    its anchor generator so each forward records the per-level anchor counts (needed by ATSS,
    which selects top-k *per FPN level*). Only called when ``DET_ASSIGNER == 'atss'``; the fixed
    path never touches this. Returns the same object (class swapped in place).
    """
    import torch as _t
    from .. import conventions as C

    # Wrap anchor_generator.forward to stash per-level anchor counts for the current batch.
    ag = base_model.anchor_generator
    _orig_forward = ag.forward

    def _recording_forward(image_list, feature_maps):
        napl = [int(ag.num_anchors_per_location()[i]) * int(fm.shape[-2]) * int(fm.shape[-1])
                for i, fm in enumerate(feature_maps)]
        base_model._num_anchors_per_level = napl
        return _orig_forward(image_list, feature_maps)

    ag.forward = _recording_forward

    class RetinaNetATSS(base_model.__class__):
        def compute_loss(self, targets, head_outputs, anchors):
            matched_idxs = []
            napl = getattr(self, "_num_anchors_per_level", None)
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0 or napl is None:
                    matched_idxs.append(_t.full((anchors_per_image.size(0),), BELOW_LOW_THRESHOLD,
                                                dtype=_t.int64, device=anchors_per_image.device))
                    continue
                m = atss_match(
                    targets_per_image["boxes"].detach().cpu().numpy(),
                    anchors_per_image.detach().cpu().numpy(),
                    napl, C.DET_ATSS_TOPK)
                matched_idxs.append(_t.as_tensor(m, dtype=_t.int64, device=anchors_per_image.device))
            return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    base_model.__class__ = RetinaNetATSS
    base_model._assigner = "atss"
    return base_model
