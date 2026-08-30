"""[4.5] The set module — a Set-Transformer over one volume's candidates (§1.3).

``RGFrocRescorer`` = ``TokenProjection -> L x SAB (+ GeometryBias iff use_geometry) ->
Linear(H, 1)``, a **fresh** per-candidate logit (never a residual on ``score_max``, so the
score-stats ablation stays clean).

**Per-volume sets only.** Attention never crosses sets and there is NO batch-level
normalisation of the logits: that would be transductive (against Inv. 9's spirit) and
``g(m,n)`` is meaningless between patients.

**Plain SAB, not ISAB.** On the promoted pool the largest set is **509** (train fold0 vol14;
val's worst is 292) and ``RESC_MAX_SET_SIZE`` is 576, so the worst-case n² attention is
~2.6·10⁵ entries — still cheap, and batches pad to the batch max, so the typical cost is set
by the val median of 85. ISAB is deliberately absent: its inducing points would delete the
full n×n term that IS the Axis-A contribution.

Attention is hand-rolled rather than ``nn.MultiheadAttention`` for two concrete reasons:
the Axis-A bias must be added **per head** to the pre-softmax logits, and padded query rows
must stay finite (a fully-masked query row in the stock module yields NaN).

Torch-only module: it raises a clear ImportError on the laptop.
"""

from __future__ import annotations

import math
from typing import Optional

from .. import conventions as C
from .losses import to_probability  # re-exported: the [0, 1-eps) contract lives with the losses

__all__ = ["SAB", "RGFrocRescorer", "B1Rescorer", "to_probability", "build_rescorer"]

try:
    import torch
    from torch import nn

    _TORCH_OK = True
except ImportError:  # pragma: no cover - laptop path
    _TORCH_OK = False


if not _TORCH_OK:  # pragma: no cover - laptop path

    class _NeedsTorch:
        def __init__(self, *args, **kwargs):
            raise ImportError("the Phase-4 set module needs torch; run it in the server env "
                              "($SW/envs/abus-jcr).")

    SAB = RGFrocRescorer = B1Rescorer = _NeedsTorch  # type: ignore[misc,assignment]

    def build_rescorer(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("build_rescorer needs torch; run it in the server env.")

else:

    from .geometry_bias import GeometryBias
    from .tokens import TokenProjection

    class SAB(nn.Module):
        """Set-Transformer self-attention block.

        Multi-head attention + residual + LayerNorm, then ``FFN(H -> 2H -> H)`` + residual +
        LayerNorm, with dropout ``RESC_SET_DROPOUT``.

        ``mask`` is ``(B, N)`` with **True = a real candidate**; padded KEYS are removed from
        the softmax, so padded slots cannot influence a real candidate's output and padded
        query rows still attend over the real keys (hence never NaN).

        ``geom_bias`` is ``(B, n_heads, N, N)``, added to the attention logits before the
        softmax; ``None`` gives plain appearance attention (rung B2).
        """

        def __init__(self, d_model: int, n_heads: int, dropout: Optional[float] = None):
            super().__init__()
            if d_model % n_heads:
                raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
            dropout = C.RESC_SET_DROPOUT if dropout is None else float(dropout)
            self.n_heads = int(n_heads)
            self.d_head = d_model // int(n_heads)
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            self.o = nn.Linear(d_model, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
            self.drop = nn.Dropout(dropout)

        def attention(self, x, mask=None, geom_bias=None):
            """Returns ``(context, attn)`` — ``attn`` is exposed for the [4.9] probe."""
            b, n, d = x.shape
            def split(t):
                return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,N,dh)
            q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
            logits = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)           # (B,H,N,N)
            if geom_bias is not None:
                logits = logits + geom_bias
            if mask is not None:
                keep = mask[:, None, None, :]                                   # mask KEYS
                logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
            ctx = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, d)
            return self.o(ctx), attn

        def forward(self, x, mask=None, geom_bias=None, return_attn: bool = False):
            ctx, attn = self.attention(x, mask, geom_bias)
            h = self.ln1(x + self.drop(ctx))
            h = self.ln2(h + self.drop(self.ff(h)))
            return (h, attn) if return_attn else h

    class RGFrocRescorer(nn.Module):
        """The joint rungs: B2 (``use_geometry=False``), A1/FULL (``use_geometry=True``).

        ``forward(feats, coord, length, mask) -> logits (B, N)``. ``coord``/``length`` are
        the record's **official** box columns (``coordX..z_length``), so the Axis-A
        descriptor is byte-identical to the one that produced the recorded priors; they are
        ignored entirely when ``use_geometry`` is False.
        """

        def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                     dropout: Optional[float] = None, use_geometry: bool = False,
                     geom_mechanism: Optional[str] = None, geom_pe_dim: Optional[int] = None):
            super().__init__()
            dropout = C.RESC_SET_DROPOUT if dropout is None else float(dropout)
            self.d_in, self.d_model = int(d_in), int(d_model)
            self.n_layers, self.n_heads = int(n_layers), int(n_heads)
            self.use_geometry = bool(use_geometry)
            self.token = TokenProjection(d_in, d_model)
            self.blocks = nn.ModuleList([SAB(d_model, n_heads, dropout) for _ in range(n_layers)])
            self.geom = (GeometryBias(n_heads, pe_dim=geom_pe_dim, mechanism=geom_mechanism)
                         if use_geometry else None)
            self.head = nn.Linear(d_model, 1)

        def forward(self, feats, coord=None, length=None, mask=None, return_attn: bool = False):
            h = self.token(feats)
            gb = None
            if self.geom is not None:
                if coord is None or length is None:
                    raise ValueError("use_geometry=True needs coord/length (the official box columns)")
                gb = self.geom(coord, length)
            attns = []
            for blk in self.blocks:
                if return_attn:
                    h, a = blk(h, mask, gb, return_attn=True)
                    attns.append(a)
                else:
                    h = blk(h, mask, gb)
            logits = self.head(h).squeeze(-1)
            return (logits, attns) if return_attn else logits

    class B1Rescorer(nn.Module):
        """Rung B1: a per-candidate MLP on the FULL token — **no attention at all**.

        A candidate's logit cannot depend on its set-mates, which is precisely what makes
        B2-vs-B1 the jointness comparison. ``hidden`` is chosen by
        ``variants.match_b1_capacity`` so the parameter count lands within ±10 % of the
        selected set module (the §4.6 fairness contract: never handicap the baseline).
        """

        def __init__(self, d_in: int, d_model: int = 128, hidden: int = 128, depth: int = 2,
                     dropout: Optional[float] = None):
            super().__init__()
            dropout = C.RESC_SET_DROPOUT if dropout is None else float(dropout)
            self.d_in, self.d_model = int(d_in), int(d_model)
            self.hidden, self.depth = int(hidden), int(depth)
            self.token = TokenProjection(d_in, d_model)
            layers, prev = [], d_model
            for _ in range(int(depth)):
                layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
                prev = hidden
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(prev, 1)

        def forward(self, feats, coord=None, length=None, mask=None, return_attn: bool = False):
            logits = self.head(self.mlp(self.token(feats))).squeeze(-1)
            return (logits, []) if return_attn else logits

    def build_rescorer(variant: str, d_in: int, capacity, hidden: Optional[int] = None,
                       dropout: Optional[float] = None, geom_mechanism: Optional[str] = None):
        """Construct the module for one rung of the §4.7 ladder.

        ``capacity`` is a ``RESC_SET_CAPACITY_GRID`` entry ``(tag, n_layers, d_model, n_heads)``
        — the SAME capacity for every set rung, and the parameter-match target for B1.
        """
        from .variants import VARIANTS

        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
        _tag, n_layers, d_model, n_heads = capacity
        spec = VARIANTS[variant]
        if spec["module"] == "mlp":
            if hidden is None:
                raise ValueError("B1 needs `hidden` from variants.match_b1_capacity(...)")
            return B1Rescorer(d_in, d_model=d_model, hidden=hidden, depth=2, dropout=dropout)
        return RGFrocRescorer(d_in, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                              dropout=dropout, use_geometry=bool(spec["geometry"]),
                              geom_mechanism=geom_mechanism)
