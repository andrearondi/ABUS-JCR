"""Geometry-keyed centre-distance dedup for the candidate pool ([3.4c]).

Data-independent (numpy). Collapses near-duplicate parallel tubes on the same object
to ONE representative chosen by a GEOMETRY key (tube length / fill), not by the
unreliable detector score — the [3.4b] score-based 3D-NMS loses recall by keeping a
high-score FP over the low-score TP. Torch-free.
"""

import numpy as np

from abus_jcr.link.dedup import dedup_by_centre_distance, single_linkage_labels


def test_keeps_far_apart_and_collapses_near_neighbours():
    # two near centres (d=1) + one far (d=100); radius 5 -> keep the far one and the
    # higher-key of the near pair.
    centres = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0]], float)
    keys = np.array([1.0, 2.0, 3.0])          # tube "length": index1 > index0
    keep = dedup_by_centre_distance(centres, keys, radius=5.0)
    assert set(keep) == {1, 2}                 # index0 suppressed by its stronger neighbour


def test_representative_is_the_highest_key_in_the_neighbourhood():
    # within one cluster the longest/best-filled tube wins, regardless of order.
    centres = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], float)  # all within radius 5 chain
    keys = np.array([1.0, 9.0, 2.0])           # middle one is the strongest
    keep = dedup_by_centre_distance(centres, keys, radius=5.0)
    assert keep == [1]                          # one representative, the max-key


def test_radius_zero_keeps_everything():
    centres = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    keys = np.array([1.0, 2.0, 3.0])
    keep = dedup_by_centre_distance(centres, keys, radius=0.0)
    assert sorted(keep) == [0, 1, 2]


def test_empty_pool():
    assert dedup_by_centre_distance(np.zeros((0, 3)), np.zeros((0,)), radius=5.0) == []


def test_a_lonely_true_tube_survives_even_if_low_key():
    # the TP tube is isolated (far from the FP cloud) and has a LOW key: it must still
    # survive (this is the recall-preservation property the score-NMS lacks).
    centres = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [50, 0, 0]], float)
    keys = np.array([9.0, 8.0, 7.0, 0.5])      # the isolated tube (idx 3) has the lowest key
    keep = dedup_by_centre_distance(centres, keys, radius=5.0)
    assert 3 in keep                            # isolated TP survives despite low key
    assert keep == [0, 3]                       # FP cloud collapses to its best; TP kept


def test_single_linkage_labels_two_clusters():
    pts = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]], float)
    labels = single_linkage_labels(pts, radius=5.0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert len(set(labels)) == 2
