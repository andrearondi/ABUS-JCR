"""The 2.5D torchvision RetinaNet (Inv. 1, 6): C-channel stem, 1 lesion class.

``build_retinanet`` adapts ``retinanet_resnet50_fpn_v2`` (COCO_V1) to C=3 axial
channels and a single lesion class, overriding the anchor generator to the
Train-derived iso-pixel scales and the input transform to the iso frame size +
per-channel normalisation. Output boxes are returned in the **iso-slice pixel
frame** (torchvision inverts its internal resize), so linking stays in iso space.

Torch/torchvision are imported lazily inside the functions so importing this
module never requires torch (the laptop env has none). ``test_retinanet_build``
pins the torchvision-API facts wherever torch is installed (the server).
"""

from __future__ import annotations

from typing import Dict, Tuple

from .. import conventions as C


def _anchor_sizes(base_sizes) -> Tuple[Tuple[int, int, int], ...]:
    """One ``(s, s*2^(1/3), s*2^(2/3))`` sub-octave triple per FPN level."""
    return tuple((int(s), int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3))) for s in base_sizes)


def _freeze_backbone_bn(model) -> str:
    """Replace every ``nn.BatchNorm2d`` in the backbone with ``FrozenBatchNorm2d``.

    [P3-UPDATE D1] ``retinanet_resnet50_fpn_v2`` builds the ResNet-50 backbone with
    **live** ``nn.BatchNorm2d`` (v1 uses ``FrozenBatchNorm2d``; the v2 factory dropped it
    because its reference recipe used ``--sync-bn`` over a large effective batch). At batch
    8 with no sync-bn, and with ``_val_loss`` forwarding val slices in ``train()`` mode,
    the backbone BN running statistics were being overwritten with validation data every
    epoch (``requires_grad_(False)`` freezes *parameters*, not the BN *buffers*). Converting
    to ``FrozenBatchNorm2d`` — carrying the COCO ``weight/bias/running_mean/running_var`` —
    makes the backbone a pure function of its input (what v1 does by construction) and is the
    correct choice for small-batch fine-tuning. The GroupNorm heads are stateless and are
    left untouched. Idempotent. Returns a description recorded in the build cfg.
    """
    from torch import nn
    from torchvision.ops.misc import FrozenBatchNorm2d

    n_converted = 0

    def convert(module):
        nonlocal n_converted
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                fbn = FrozenBatchNorm2d(child.num_features, eps=child.eps)
                fbn.weight.data.copy_(child.weight.data)
                fbn.bias.data.copy_(child.bias.data)
                fbn.running_mean.data.copy_(child.running_mean.data)
                fbn.running_var.data.copy_(child.running_var.data)
                setattr(module, name, fbn)
                n_converted += 1
            else:
                convert(child)

    convert(model.backbone)
    return f"froze {n_converted} backbone BatchNorm2d -> FrozenBatchNorm2d (COCO stats preserved)"


def _adapt_stem(model, c_channels: int) -> str:
    """Adapt ``backbone.body.conv1`` (7x7, 3->64) to ``c_channels`` inputs.

    ``c_channels == 3`` keeps the pretrained RGB stem as-is (the Phase-1 design
    intent: C=3 maps 1:1). Otherwise replace it with a fresh conv whose weight is
    the pretrained weight averaged over the input dim, tiled across ``c_channels``,
    and scaled by ``3/c_channels`` (energy-preserving). Returns a description of
    the branch taken (recorded in the build cfg).
    """
    import torch
    from torch import nn

    conv1 = model.backbone.body.conv1
    if c_channels == 3:
        return "kept pretrained RGB stem (C=3 maps 1:1)"
    w = conv1.weight.data  # (64, 3, 7, 7)
    mean_w = w.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    new_w = mean_w.repeat(1, c_channels, 1, 1) * (3.0 / c_channels)
    new_conv = nn.Conv2d(c_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                         stride=conv1.stride, padding=conv1.padding, bias=(conv1.bias is not None))
    with torch.no_grad():
        new_conv.weight.copy_(new_w)
    model.backbone.body.conv1 = new_conv
    return f"replaced stem: pretrained weight averaged+tiled to C={c_channels}, scaled 3/{c_channels}"


def build_retinanet(
    c_channels: int = C.C_CHANNELS,
    num_classes: int = C.DET_NUM_CLASSES,
    pretrained: bool = True,
    **overrides,
):
    """Build the adapted RetinaNet. ``overrides`` may set ``min_size``,
    ``max_size``, ``image_mean``, ``image_std``, ``anchor_base_sizes``,
    ``anchor_aspect_ratios`` (else the ``conventions.DET_*`` values are used)."""
    from functools import partial

    from torch import nn
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import (
        RetinaNet_ResNet50_FPN_V2_Weights,
        RetinaNetClassificationHead,
        RetinaNetRegressionHead,
    )
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    min_size = int(overrides.get("min_size", C.DET_MIN_SIZE))
    max_size = int(overrides.get("max_size", C.DET_MAX_SIZE))
    image_mean = float(overrides.get("image_mean", C.DET_IMAGE_MEAN))
    image_std = float(overrides.get("image_std", C.DET_IMAGE_STD))
    base_sizes = tuple(overrides.get("anchor_base_sizes", C.DET_ANCHOR_BASE_SIZES))
    aspect_ratios = tuple(float(a) for a in overrides.get("anchor_aspect_ratios", C.DET_ANCHOR_ASPECT_RATIOS))
    fg_iou_thresh = float(overrides.get("fg_iou_thresh", C.DET_FG_IOU_THRESH))
    bg_iou_thresh = float(overrides.get("bg_iou_thresh", C.DET_BG_IOU_THRESH))

    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = retinanet_resnet50_fpn_v2(weights=weights, weights_backbone=weights_backbone)

    stem_branch = _adapt_stem(model, c_channels)
    bn_branch = _freeze_backbone_bn(model)   # [P3-UPDATE D1] kill the live-BN corruption

    # --- anchors: Train-derived iso-pixel scales, 5 FPN levels ---
    sizes = _anchor_sizes(base_sizes)
    ars = (aspect_ratios,) * len(sizes)
    model.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ars)
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    # --- heads: classification -> num_classes, regression kept (rebuilt for num_anchors) ---
    in_channels = model.backbone.out_channels
    norm_layer = partial(nn.GroupNorm, 32)
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer)
    model.head.regression_head = RetinaNetRegressionHead(
        in_channels, num_anchors, norm_layer=norm_layer)
    # [P3-UPDATE D5] The rebuilt regression head defaults to _loss_type="l1"; the torchvision v2
    # factory sets "giou" *after* construction, so our rebuild silently reverted it. Restore it.
    model.head.regression_head._loss_type = "giou"

    # --- [P2-UPDATE B2] anchor<->GT matcher: explicit (loosened) thresholds ---
    # torchvision's default (0.5/0.4, never set) starved the cls head of positives on
    # small/odd boxes. Loosen so more anchors per GT clear the fg bar; allow_low_quality
    # still guarantees each GT >= 1 positive.
    from torchvision.models.detection import _utils as det_utils
    model.proposal_matcher = det_utils.Matcher(
        fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)

    # --- transform: iso frame size + per-channel uniform normalisation ---
    model.transform = GeneralizedRCNNTransform(
        min_size, max_size, [image_mean] * c_channels, [image_std] * c_channels)

    # --- diagnostic inference knobs (settable; Phase 3 owns the operating point) ---
    model.score_thresh = C.DET_DIAG_SCORE_THRESH
    model.nms_thresh = C.DET_DIAG_NMS_THRESH
    model.detections_per_img = C.DET_DIAG_DETECTIONS_PER_IMG

    model._build_cfg = {
        "backbone": C.DET_BACKBONE,
        "c_channels": c_channels,
        "num_classes": num_classes,
        "pretrained": pretrained,
        "min_size": min_size,
        "max_size": max_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "anchor_base_sizes": tuple(int(s) for s in base_sizes),
        "anchor_aspect_ratios": aspect_ratios,
        "fg_iou_thresh": fg_iou_thresh,
        "bg_iou_thresh": bg_iou_thresh,
        "stem_branch": stem_branch,
        "backbone_bn": "frozen",          # [P3-UPDATE D1]
        "bn_branch": bn_branch,
        "reg_loss": "giou",               # [P3-UPDATE D5]
        "assigner": C.DET_ASSIGNER,       # [P3-UPDATE D6]
    }
    # [P3-UPDATE D6] Optionally swap the fixed IoU matcher for ATSS adaptive assignment. Default
    # "fixed" leaves the model exactly as built above; "atss" re-parents to the ATSS loss subclass.
    if C.DET_ASSIGNER == "atss":
        from .atss import build_retinanet_atss
        model = build_retinanet_atss(model)
    return model


def save_checkpoint(path, model, cfg: Dict) -> None:
    """Persist ``state_dict`` + the full build config so Phase 3 rebuilds identically."""
    import torch

    full_cfg = dict(getattr(model, "_build_cfg", {}))
    full_cfg.update(cfg)
    torch.save({"state_dict": model.state_dict(), "cfg": full_cfg}, path)


def load_checkpoint(path):
    """Rebuild the model from a saved checkpoint -> ``(model, cfg)``.

    Reconstructs the architecture from the recorded cfg (so anchors/stem/transform
    match byte-for-byte), then loads the weights.
    """
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    model = build_retinanet(
        c_channels=cfg["c_channels"],
        num_classes=cfg["num_classes"],
        pretrained=False,
        min_size=cfg["min_size"],
        max_size=cfg["max_size"],
        image_mean=cfg["image_mean"],
        image_std=cfg["image_std"],
        anchor_base_sizes=cfg["anchor_base_sizes"],
        anchor_aspect_ratios=cfg["anchor_aspect_ratios"],
        fg_iou_thresh=cfg.get("fg_iou_thresh", C.DET_FG_IOU_THRESH),
        bg_iou_thresh=cfg.get("bg_iou_thresh", C.DET_BG_IOU_THRESH),
    )
    model.load_state_dict(blob["state_dict"])
    return model, cfg
