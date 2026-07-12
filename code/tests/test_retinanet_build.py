"""The 2.5D RetinaNet build (torchvision-API facts).

Data-independent but torch-dependent: pins the C=3 stem, the 1-lesion-class head
(finite dummy loss with label==DET_FG_LABEL), the anchor override, the transform,
and that eval() returns boxes. torch is absent on the laptop, so the whole module
SKIPs there; it runs wherever torch/torchvision are installed (the server).
Uses ``pretrained=False`` so no weights are downloaded.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from abus_jcr import conventions as C
from abus_jcr.detect.retinanet import build_retinanet, save_checkpoint, load_checkpoint


def _small_model(c_channels=3):
    return build_retinanet(
        c_channels=c_channels, num_classes=C.DET_NUM_CLASSES, pretrained=False,
        min_size=64, max_size=64,
        anchor_base_sizes=(16, 32, 64, 128, 256),
        anchor_aspect_ratios=(0.5, 1.0, 2.0),
    )


def test_c3_build_forward_backward_finite_loss():
    m = _small_model(3).train()
    img = torch.rand(3, 64, 64)
    target = {"boxes": torch.tensor([[5.0, 5.0, 20.0, 25.0]]),
              "labels": torch.tensor([C.DET_FG_LABEL])}
    losses = m([img], [target])
    total = sum(losses.values())
    assert torch.isfinite(total), losses
    total.backward()  # must not raise


def test_head_sized_to_num_classes():
    m = _small_model(3)
    assert m.head.classification_head.num_classes == C.DET_NUM_CLASSES


def test_anchor_override_sizes_and_ratios():
    m = _small_model(3)
    # each level: (s, int(s*2^(1/3)), int(s*2^(2/3))); 5 FPN levels
    assert len(m.anchor_generator.sizes) == 5
    assert m.anchor_generator.sizes[0] == (16, int(16 * 2 ** (1 / 3)), int(16 * 2 ** (2 / 3)))
    assert m.anchor_generator.aspect_ratios[0] == (0.5, 1.0, 2.0)


def test_matcher_thresholds_loosened_and_low_quality_on():
    # [P2-UPDATE B2] explicit fg/bg IoU thresholds (were torchvision default 0.5/0.4).
    m = _small_model(3)
    assert m.proposal_matcher.high_threshold == C.DET_FG_IOU_THRESH == 0.4
    assert m.proposal_matcher.low_threshold == C.DET_BG_IOU_THRESH == 0.3
    assert m.proposal_matcher.allow_low_quality_matches is True


def test_checkpoint_round_trip_preserves_matcher_thresholds(tmp_path):
    m = build_retinanet(c_channels=3, num_classes=C.DET_NUM_CLASSES, pretrained=False,
                        min_size=64, max_size=64, fg_iou_thresh=0.4, bg_iou_thresh=0.3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, m, {"regime": "fold", "fold_or_seed": 0})
    m2, cfg2 = load_checkpoint(path)
    assert cfg2["fg_iou_thresh"] == 0.4 and cfg2["bg_iou_thresh"] == 0.3
    assert m2.proposal_matcher.high_threshold == 0.4
    assert m2.proposal_matcher.low_threshold == 0.3


def test_transform_uses_iso_frame_sizes_and_channel_norm():
    m = _small_model(3)
    assert m.transform.min_size == (64,)
    assert m.transform.max_size == 64
    assert len(m.transform.image_mean) == 3 and len(m.transform.image_std) == 3


def test_eval_returns_boxes():
    m = _small_model(3).eval()
    with torch.no_grad():
        out = m([torch.rand(3, 64, 64)])
    assert isinstance(out, list) and "boxes" in out[0] and "scores" in out[0]


def test_non_rgb_stem_is_replaced_with_matching_in_channels():
    m = build_retinanet(c_channels=5, num_classes=C.DET_NUM_CLASSES, pretrained=False)
    assert m.backbone.body.conv1.in_channels == 5


def test_checkpoint_round_trip_rebuilds_identically(tmp_path):
    m = _small_model(3)
    cfg = {"regime": "full", "seed": 0, "c_channels": 3, "num_classes": C.DET_NUM_CLASSES,
           "min_size": 64, "max_size": 64,
           "anchor_base_sizes": (16, 32, 64, 128, 256), "anchor_aspect_ratios": (0.5, 1.0, 2.0),
           "image_mean": C.DET_IMAGE_MEAN, "image_std": C.DET_IMAGE_STD}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, m, cfg)
    m2, cfg2 = load_checkpoint(path)
    assert cfg2["regime"] == "full" and cfg2["c_channels"] == 3
    # state dicts identical after reload
    sd1, sd2 = m.state_dict(), m2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k])
