"""[4.4] Axis A — the pairwise relative-geometry attention bias (the core novelty).

Two pieces:

1. :func:`relative_geometry_batch` — the 6-D descriptor ``g(m,n)``, computed on the
   record's **official** box columns (``coordX..z_length``) so it is byte-identical to
   :func:`probe.pool_diag.relative_geometry`, the function that produced every recorded
   Axis-A number. It is written **array-agnostically** (pure broadcasting + a two-line
   numpy/torch dispatch), so the *same* code path is parity-tested on the laptop and runs
   under autograd on the server — the parity claim is not deferred to a machine we cannot
   reach.

   AXIS NOTE — read this before interpreting any per-component result. CORRECTED
   2026-08-09; the previous note here said ``coordZ`` was the depth/beam axis and it was
   **wrong**, in the same direction as the Inv.-13 defect. ``PERM_STORAGE_TO_ITK = (2,1,0)``
   maps official ``(x, y, z)`` to storage ``(d2, d1, d0)``, and the storage roles were
   *measured* on 129/130 volumes (``results/AXIS_CHECK.md``, four independent lines) to be
   ``d0 = lateral``, ``d1 = depth/beam``, ``d2 = sweep``. Therefore::

       coordX / x_length  <->  d2  =  SWEEP
       coordY / y_length  <->  d1  =  DEPTH / BEAM      <- the axis Phase-0b implicates
       coordZ / z_length  <->  d0  =  LATERAL

   So in ``g(m,n)`` the ``log|dy|/h`` component is the depth offset and ``log|dz|/d`` is the
   lateral one. This matters for reporting, not for arithmetic: every component is a ratio,
   so the native anisotropic voxel units cancel and the descriptor is unchanged either way.
   It matters a great deal for the write-up — on the promoted val pool the largest pairwise
   effect is ``log|dz|/d`` at ``|δ| = 0.126`` ([F.9] §3), which is the **lateral** axis, and
   calling it "the depth axis, exactly where Phase-0b expected the signal" would invert the
   finding. (See also ``conventions.FP_PROBE_ANISO_DEPTH_AXIS`` and ``[I.6b]``.)

2. :class:`GeometryBias` — the learned per-head attention bias. ``additive`` (default,
   iRPE-style) is **zero-initialised**, so at step 0 rung A1 is numerically identical to
   rung B2: geometry can only be *learned*, never assumed. ``multiplicative`` (Relation
   Networks form) is the pre-registered fallback, switched on only if A1 == B2 within CI,
   to distinguish "no signal" from "wrong mechanism".

**Prior, pre-registered:** the direct test of this exact descriptor on the PROMOTED pool
([F.9] §3) found TP-FP vs FP-FP ``max |δ| = 0.126`` (0.112 / 0.181 / 0.072 per seed) — the
same band as the archived pool's 0.082, i.e. still a **WEAK** prior; and the independent
per-candidate FP-structure probe returns ``structure_present = false`` again. A null A1−B2 is
a legitimate, publishable outcome, not a bug. The contrast that IS strong is TP-TP vs TP-FP
at ``|δ| = 0.955`` — co-location/consensus, which jointness (B2) captures natively.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import conventions as C

__all__ = ["relative_geometry_batch", "relative_geometry_torch", "sinusoidal_pe", "GeometryBias"]


# ----------------------------------------------------------------------------- array dispatch
def _xp(x):
    """The array module backing ``x`` — numpy or torch. Keeps one code path for both."""
    if type(x).__module__.split(".")[0] == "torch":
        import torch
        return torch
    return np


def _concat(xp, parts, axis=-1):
    return xp.cat(parts, dim=axis) if xp is not np else np.concatenate(parts, axis=axis)


# ----------------------------------------------------------------------------- the descriptor
def relative_geometry_batch(coord, length, eps: Optional[float] = None):
    """``(..., N, 3), (..., N, 3) -> (..., N, N, 6)`` relative-log-geometry ``g(m,n)``.

    ``g(m,n) = [log(|dx|/w_m + eps), log(|dy|/h_m + eps), log(|dz|/d_m + eps),
                log(w_n/w_m), log(h_n/h_m), log(d_n/d_m)]``

    with ``m`` the query (row) and ``n`` the key (column) — the same argument order as
    :func:`probe.pool_diag.relative_geometry`. ``eps`` defaults to ``RESC_GEOM_EPS``, which
    a unit test pins equal to ``pool_diag.EPS``.
    """
    eps = C.RESC_GEOM_EPS if eps is None else float(eps)
    xp = _xp(coord)
    c_m, c_n = coord[..., :, None, :], coord[..., None, :, :]
    l_m, l_n = length[..., :, None, :], length[..., None, :, :]
    dist = xp.log(xp.abs(c_m - c_n) / (l_m + eps) + eps)
    size = xp.log((l_n + eps) / (l_m + eps))
    # broadcast both to the full (..., N, N, 3) shape before concatenating
    dist = dist + xp.zeros_like(size)
    size = size + xp.zeros_like(dist)
    return _concat(xp, [dist, size], axis=-1)


#: Spec name (§4.4). The implementation is deliberately shared with the numpy path.
relative_geometry_torch = relative_geometry_batch


def sinusoidal_pe(x, dim: int):
    """Vaswani sinusoidal embedding of a CONTINUOUS scalar field.

    ``x`` of shape ``S`` -> ``S + (dim,)``: ``[sin(x·ω_0..ω_{d/2-1}), cos(x·ω_...)]`` with
    ``ω_i = 10000^(-2i/dim)``. Array-agnostic (numpy or torch).
    """
    if dim % 2:
        raise ValueError(f"sinusoidal_pe needs an even dim, got {dim}")
    xp = _xp(x)
    half = dim // 2
    i = np.arange(half, dtype=np.float64)
    freq = np.power(10000.0, -2.0 * i / float(dim))
    if xp is not np:
        import torch
        freq = torch.as_tensor(freq, dtype=x.dtype, device=x.device)
    ang = x[..., None] * freq
    return _concat(xp, [xp.sin(ang), xp.cos(ang)], axis=-1)


# ----------------------------------------------------------------------------- the learned bias
try:  # torch is absent on the laptop; the descriptor above must still import
    import torch
    from torch import nn

    _TORCH_OK = True
except ImportError:  # pragma: no cover - exercised only on the laptop
    _TORCH_OK = False


if _TORCH_OK:

    class GeometryBias(nn.Module):
        """Per-head attention bias ``b_h(m,n)`` read off ``PE(g(m,n))``.

        ``mechanism == "additive"`` (DEFAULT): ``b_h = w_b^h · PE(g)`` via
        ``Linear(6·d_g -> n_heads, bias=False)`` with **zero-initialised** weights, ADDED
        to the appearance attention logit before the softmax. At initialisation the bias
        is exactly 0, so A1 ≡ B2 (pinned by ``tests/test_geometry_bias_reduces.py``).

        ``mechanism == "multiplicative"`` (FALLBACK): ``ω = ReLU(W_G · PE(g))`` and the
        attention logit becomes ``log(ω + eps) + qk`` — still an additive term, but one
        that can *gate* rather than *shift*. Setting ``relu=False`` and forcing the linear
        to output 1 reduces it to B2 as well.

        ``forward(coord, length) -> (B, n_heads, N, N)``.
        """

        def __init__(self, n_heads: int, pe_dim: Optional[int] = None,
                     mechanism: Optional[str] = None, eps: Optional[float] = None,
                     relu: bool = True):
            super().__init__()
            self.n_heads = int(n_heads)
            self.pe_dim = int(C.RESC_GEOM_PE_DIM if pe_dim is None else pe_dim)
            self.mechanism = str(C.RESC_GEOM_MECHANISM if mechanism is None else mechanism)
            if self.mechanism not in ("additive", "multiplicative"):
                raise ValueError(f"unknown geometry mechanism {self.mechanism!r}")
            self.eps = float(C.RESC_GEOM_EPS if eps is None else eps)
            self.relu = bool(relu)
            self.proj = nn.Linear(6 * self.pe_dim, self.n_heads, bias=False)
            if self.mechanism == "additive":
                nn.init.zeros_(self.proj.weight)          # A1 == B2 at step 0
            else:
                nn.init.zeros_(self.proj.weight)
                # omega starts at 1 -> log(1 + eps) ~ 0, so the fallback also starts at B2
                self.omega_bias = nn.Parameter(torch.ones(self.n_heads))

        def forward(self, coord: "torch.Tensor", length: "torch.Tensor") -> "torch.Tensor":
            g = relative_geometry_batch(coord, length, eps=self.eps)      # (B,N,N,6)
            pe = sinusoidal_pe(g, self.pe_dim)                            # (B,N,N,6,d_g)
            pe = pe.reshape(*pe.shape[:-2], 6 * self.pe_dim)              # (B,N,N,6*d_g)
            out = self.proj(pe)                                           # (B,N,N,H)
            if self.mechanism == "multiplicative":
                out = out + self.omega_bias
                if self.relu:
                    out = torch.relu(out)
                out = torch.log(out + self.eps)
            return out.permute(0, 3, 1, 2).contiguous()                   # (B,H,N,N)

else:  # pragma: no cover - laptop path

    class GeometryBias:  # type: ignore[no-redef]
        """Placeholder raised on a torch-free machine; the descriptor above still works."""

        def __init__(self, *args, **kwargs):
            raise ImportError("GeometryBias needs torch; run it in the server env "
                              "($SW/envs/abus-jcr).")
