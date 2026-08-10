"""[4.2] The shared 3D encoder — ONE instance for every candidate, variant and detector (Inv. 5).

The encoder consumes a 48³ crop **re-extracted from the iso cache** at the candidate's frozen
coordinates. No detector feature map is ever forwarded — the detector is not even loaded in
Phase 4. That is what keeps the appearance space comparable across the three heterogeneous
detectors of Phase 6.

**Mode: pretrain-then-freeze** (per Inv. 10 + the small-data reality). For each seed ``r``:

1. train ``CandidateEncoder + B1Head`` end-to-end on the **train** pool — this run *is* rung B1;
2. freeze the encoder and cache ``a_i`` for every train and val candidate, **un-augmented**;
3. every set-module variant reads the cached ``a_i``.

That guarantees the appearance features are *identical* across B1/B2/A1/A2/FULL, so the
ablation isolates the set module and never the encoder — and makes each set-module run cost
seconds instead of GPU-hours.

**Capacity (spec Open escalation #2).** MONAI DenseNet-121-3D (~11 M params) is the default;
on the promoted pool it sees **1 265 positives across 9 118 loss-bearing train crops** (was
943 / 8 061), still few, but because the encoder is frozen and *shared*, any overfit is
common to every rung and cannot confound the ablation. The pre-registered fallback is
:class:`SmallCandidateEncoder` (~1 M params), re-run at [4.3].

Its trigger is exit check 4 failing — B1 val CPM ≤ the **measured** B0 — **and** B1's
per-candidate balanced accuracy landing well below the pool's own single-feature ceiling
(0.811 on the promoted val pool, [F.9] §1). Both halves are required: a B1 that discriminates
as well as the pool's best single feature yet does not beat B0 on CPM is telling you B0 is a
strong ranking (``score_max`` δ 0.713, 39.8 % of the volume-trust signal), not that the
encoder is broken — and swapping the encoder would be the wrong response.

Torch-only module.
"""

from __future__ import annotations

from typing import Optional

from .. import conventions as C

__all__ = ["CandidateEncoder", "SmallCandidateEncoder", "B1Head", "PretrainModel", "build_encoder"]

try:
    import torch
    from torch import nn

    _TORCH_OK = True
except ImportError:  # pragma: no cover - laptop path
    _TORCH_OK = False


if not _TORCH_OK:  # pragma: no cover - laptop path

    class _NeedsTorch:
        def __init__(self, *args, **kwargs):
            raise ImportError("the Phase-4 encoder needs torch + monai; run it in the server env "
                              "(/home/maia-user/Andre2/envs/abus-jcr).")

    CandidateEncoder = SmallCandidateEncoder = B1Head = PretrainModel = _NeedsTorch  # type: ignore

    def build_encoder(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("build_encoder needs torch + monai; run it in the server env.")

else:

    from .setmodel import B1Rescorer

    #: Rung B1's head is exactly the per-candidate MLP of the ladder — one class, one
    #: parameter budget, so the fairness contract cannot drift between [4.3] and [4.6].
    B1Head = B1Rescorer

    class CandidateEncoder(nn.Module):
        """MONAI DenseNet121 (3D) as a feature extractor.

        ``features -> ReLU -> adaptive 3D GAP -> Linear(1024 -> RESC_D_APP) -> ReLU``.
        ``forward(x: (B, 1, 48, 48, 48)) -> (B, d_app)``.

        The ReLU after ``features`` is applied explicitly: MONAI's ``DenseNet.features``
        ends at ``norm5`` (a BatchNorm), with the ReLU living in the discarded
        classification head.
        """

        def __init__(self, d_app: Optional[int] = None, dropout: Optional[float] = None):
            super().__init__()
            from monai.networks.nets import DenseNet121

            d_app = int(C.RESC_D_APP if d_app is None else d_app)
            dropout = float(C.RESC_ENCODER_DROPOUT if dropout is None else dropout)
            net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=d_app,
                              dropout_prob=dropout)
            self.features = net.features
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.dropout = nn.Dropout(dropout)
            self.n_features = self._probe_feature_channels()
            self.fc = nn.Linear(self.n_features, d_app)
            self.d_app = d_app

        def _probe_feature_channels(self) -> int:
            """Measure the backbone's output width with one dry forward.

            DenseNet-121 is 1024-wide, but measuring beats hard-coding: a MONAI version bump
            or a different ``block_config`` would otherwise surface as a shape error on the
            first training batch, after the crop cache and a dataloader are already up."""
            was_training = self.training
            self.features.eval()
            with torch.no_grad():
                probe = torch.zeros(1, 1, int(C.RESC_CROP_OUT), int(C.RESC_CROP_OUT),
                                    int(C.RESC_CROP_OUT))
                n = int(self.features(probe).shape[1])
            self.features.train(was_training)
            return n

        def forward(self, x):
            h = torch.relu(self.features(x))
            h = self.pool(h).flatten(1)
            return torch.relu(self.fc(self.dropout(h)))

    class SmallCandidateEncoder(nn.Module):
        """PRE-REGISTERED FALLBACK (~1 M params), NOT deployed by default.

        Four ``Conv3d-BN-ReLU-Conv3d-BN-ReLU-MaxPool`` blocks at ``RESC_SMALL_CNN_WIDTHS``,
        then GAP -> ``Linear(-> d_app)`` -> ReLU. Same input/output contract as
        :class:`CandidateEncoder`, so [4.3] can swap it in without touching anything else.
        """

        def __init__(self, d_app: Optional[int] = None, dropout: Optional[float] = None,
                     widths=None):
            super().__init__()
            d_app = int(C.RESC_D_APP if d_app is None else d_app)
            dropout = float(C.RESC_ENCODER_DROPOUT if dropout is None else dropout)
            widths = tuple(C.RESC_SMALL_CNN_WIDTHS if widths is None else widths)
            blocks, cin = [], 1
            for w in widths:
                blocks += [
                    nn.Conv3d(cin, w, 3, padding=1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True),
                    nn.Conv3d(w, w, 3, padding=1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                ]
                cin = w
            self.features = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(cin, d_app)
            self.d_app = d_app

        def forward(self, x):
            h = self.pool(self.features(x)).flatten(1)
            return torch.relu(self.fc(self.dropout(h)))

    class PretrainModel(nn.Module):
        """[4.3] ``CandidateEncoder + B1Head`` trained end-to-end — this run IS rung B1.

        The head sees the FULL token: the encoder's ``a_i`` concatenated with the
        already-standardised non-appearance blocks (``abs_geom``, ``score_stats``,
        ``tube_geom``, ``rank``) supplied as ``rest``.
        """

        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def embed(self, crops):
            """``(B, 1, 48, 48, 48) -> (B, d_app)`` — the frozen feature extractor path."""
            return self.encoder(crops)

        def forward(self, crops, rest, mask=None):
            a = self.encoder(crops)                       # (B, d_app)
            feats = torch.cat([a, rest], dim=-1)[:, None, :]   # a set of ONE (B1 is per-candidate)
            logits = self.head(feats, None, None, None)
            return logits.squeeze(1)

    def build_encoder(name: Optional[str] = None, d_app: Optional[int] = None,
                      dropout: Optional[float] = None):
        """``RESC_ENCODER`` -> DenseNet121; ``RESC_ENCODER_FALLBACK`` -> the small CNN."""
        name = str(C.RESC_ENCODER if name is None else name)
        if name == C.RESC_ENCODER:
            return CandidateEncoder(d_app=d_app, dropout=dropout)
        if name == C.RESC_ENCODER_FALLBACK:
            return SmallCandidateEncoder(d_app=d_app, dropout=dropout)
        raise ValueError(f"unknown encoder {name!r}; expected {C.RESC_ENCODER!r} or "
                         f"{C.RESC_ENCODER_FALLBACK!r}")
