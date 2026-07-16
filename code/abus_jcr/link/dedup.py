"""Geometry-keyed centre-distance dedup for the candidate pool ([3.4c]).

The linker spawns one parallel tube per box in the loosened per-slice NMS stack, so a
single object carries ~hundreds of near-duplicate tubes (redundancy, [3.4b]). The
score-based 3D-NMS ([3.4b] / ``nms.nms_3d``) collapses them by keeping the highest
detector-**score** tube per cluster — but the detector's score is poorly calibrated
(TPs rank below FPs), so it often keeps a high-score FP and drops the low-score TP,
losing recall.

``dedup_by_centre_distance`` collapses each spatial neighbourhood to ONE representative
chosen by a **geometry** key (e.g. tube slice-count or fill-ratio — a real lesion makes
a long, well-filled tube; FPs make short sparse ones), which is independent of the
unreliable score. It is a greedy centre-distance suppression (no single-linkage
chaining): sort by key desc, keep the top, suppress every other tube whose centre lies
within ``radius`` iso-voxels, repeat. Candidate for the frozen aggregation (Inv. 4) if
the [3.4c] what-if shows it preserves the recall ceiling while cutting the pool.

Torch-free (numpy); operates on iso-voxel tube centres, so it is unit-tested on the
laptop and reused by ``scripts/phase3_geom_dedup_whatif.py``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def dedup_by_centre_distance(centres: Sequence, keys: Sequence, radius: float) -> List[int]:
    """Greedy centre-distance dedup keyed by ``keys`` (higher = better representative).

    ``centres`` is ``(n, 3)`` iso-voxel tube centres; ``keys`` an ``(n,)`` geometry
    criterion. Returns the kept indices in descending-key order: the max-key tube is
    kept and every other tube within ``radius`` Euclidean centre-distance is suppressed,
    then the process repeats over the survivors. ``radius <= 0`` keeps everything. Ties
    in ``keys`` break by lower index (stable). An isolated (far) tube always survives,
    regardless of its key — the recall-preservation property the score-NMS lacks.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 3)
    keys = np.asarray(keys, dtype=float).reshape(-1)
    n = len(centres)
    if n == 0:
        return []
    if radius <= 0:
        return list(range(n))

    order = np.argsort(-keys, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    keep: List[int] = []
    for i in order:
        i = int(i)
        if suppressed[i]:
            continue
        keep.append(i)
        d = np.linalg.norm(centres - centres[i], axis=1)
        suppressed |= (d <= radius)   # marks i (d=0) too; already kept, harmless
    return keep


def single_linkage_labels(points: Sequence, radius: float) -> np.ndarray:
    """Cluster labels (0..k-1) via single-linkage: points within ``radius`` join.

    Companion to ``probe.fp_structure._single_linkage_clusters`` (which returns only the
    count). Prone to chaining — kept for diagnostics/comparison, not the primary dedup.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=int)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= radius:
                parent[find(i)] = find(j)
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels
