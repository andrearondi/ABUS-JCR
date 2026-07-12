"""The per-slice detection dataset (torch-free helpers, synthetic cache).

Verifies the correctness-critical logic without torch: the inclusive->half-open
box conversion, the negative-sampling ratio, and that augmentation keeps the box
and the C channel-slices in sync. The torch-tensor ``__getitem__`` path is
exercised on the server (it needs torchvision); here we test the pure helpers.
"""

import numpy as np
import pandas as pd

from abus_jcr import cache as K
from abus_jcr import conventions as C
from abus_jcr.detect import slice_det_dataset as D


def _write_synth_volume(cache_root, vid, d0=6, d1=8, nz=5):
    vol = np.zeros((d0, d1, nz), dtype=np.float32)
    for z in range(nz):
        vol[:, :, z] = float(z)  # each slice a constant, so channels are distinguishable
    mask = np.zeros_like(vol, dtype=np.uint8)
    meta = {"iso_shape": [d0, d1, nz], "native_shape": [d0, d1, nz],
            "zoom_factors": [1.0, 1.0, 1.0]}
    K.write_case(cache_root, vid, vol, mask, meta)
    return d0, d1, nz


def _boxes_df(rows):
    cols = ["volume_id", "slice_z", "r0", "c0", "r1", "c1", "component_id"]
    return pd.DataFrame(rows, columns=cols)


def test_inclusive_to_halfopen_conversion():
    df = _boxes_df([
        {"volume_id": 100, "slice_z": 2, "r0": 1, "c0": 3, "r1": 4, "c1": 6, "component_id": 0},
    ])
    boxes = D.boxes_halfopen_for(df, 100, 2)
    # (x1, y1, x2, y2) = (c0, r0, c1+1, r1+1) = (3, 1, 7, 5)
    np.testing.assert_array_equal(boxes, np.array([[3.0, 1.0, 7.0, 5.0]], dtype=np.float32))


def test_no_box_slice_returns_empty():
    df = _boxes_df([{"volume_id": 100, "slice_z": 2, "r0": 1, "c0": 3, "r1": 4, "c1": 6, "component_id": 0}])
    boxes = D.boxes_halfopen_for(df, 100, 0)
    assert boxes.shape == (0, 4)


def test_enumerate_marks_lesion_and_background(tmp_path):
    _write_synth_volume(tmp_path, 100, nz=5)
    df = _boxes_df([{"volume_id": 100, "slice_z": 2, "r0": 1, "c0": 1, "r1": 3, "c1": 3, "component_id": 0}])
    samples = D.enumerate_samples(tmp_path, [100], df)
    assert len(samples) == 5
    lesion = {z for (v, z, isl) in samples if isl}
    assert lesion == {2}


def test_sample_epoch_honours_neg_pos_ratio(tmp_path):
    _write_synth_volume(tmp_path, 100, nz=20)
    # 2 lesion slices -> ratio 3 -> 6 background sampled -> 8 total
    df = _boxes_df([
        {"volume_id": 100, "slice_z": 4, "r0": 1, "c0": 1, "r1": 2, "c1": 2, "component_id": 0},
        {"volume_id": 100, "slice_z": 9, "r0": 1, "c0": 1, "r1": 2, "c1": 2, "component_id": 0},
    ])
    samples = D.enumerate_samples(tmp_path, [100], df)
    order = D.sample_epoch(samples, neg_pos_ratio=3, seed=0, epoch=0)
    chosen = [samples[i] for i in order]
    n_lesion = sum(1 for (_, _, isl) in chosen if isl)
    n_bg = sum(1 for (_, _, isl) in chosen if not isl)
    assert n_lesion == 2       # all lesion slices kept
    assert n_bg == 6           # 3x background


def test_sample_epoch_reshuffles_between_epochs(tmp_path):
    _write_synth_volume(tmp_path, 100, nz=20)
    df = _boxes_df([{"volume_id": 100, "slice_z": 4, "r0": 1, "c0": 1, "r1": 2, "c1": 2, "component_id": 0}])
    samples = D.enumerate_samples(tmp_path, [100], df)
    o0 = D.sample_epoch(samples, neg_pos_ratio=3, seed=0, epoch=0)
    o1 = D.sample_epoch(samples, neg_pos_ratio=3, seed=0, epoch=1)
    o0b = D.sample_epoch(samples, neg_pos_ratio=3, seed=0, epoch=0)
    assert o0 == o0b            # deterministic per (seed, epoch)
    assert o0 != o1             # different epoch -> different draw


def test_load_numpy_sample_hflip_syncs_box_and_channels(tmp_path):
    d0, d1, nz = _write_synth_volume(tmp_path, 100, d0=6, d1=8, nz=5)
    df = _boxes_df([{"volume_id": 100, "slice_z": 2, "r0": 1, "c0": 1, "r1": 3, "c1": 4, "component_id": 0}])
    ds = D.SliceDetectionDataset(tmp_path, df, volume_ids=[100], train=True, seed=0,
                                 policy=dict(D.TRAIN_AUGMENT, horizontal_flip_p=1.0,
                                             small_translation=False, intensity_jitter=False,
                                             gaussian_blur=False, gaussian_noise=False,
                                             scale_zoom=False, rotation=False))
    rng = np.random.default_rng(0)
    stack, boxes = ds.load_numpy_sample(100, 2, rng)
    assert stack.shape == (C.C_CHANNELS, d0, d1)
    # centre channel is slice 2; near->far channels are slices 1,2,3
    np.testing.assert_array_equal(stack[0], np.full((d0, d1), 1.0)[:, ::-1])
    np.testing.assert_array_equal(stack[1], np.full((d0, d1), 2.0)[:, ::-1])
    # box (c0,r0,c1+1,r1+1)=(1,1,5,4) reflected in W=8 -> (8-5,1,8-1,4)=(3,1,7,4)
    np.testing.assert_array_equal(boxes, np.array([[3.0, 1.0, 7.0, 4.0]], dtype=np.float32))
