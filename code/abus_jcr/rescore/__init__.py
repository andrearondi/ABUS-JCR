"""Phase 4 — the RG-FROC-Rescorer: a set rescorer over the FROZEN Phase-3 candidate pool.

The pool is never regenerated here (Inv. 8): every rung of the ablation ladder emits a new
``probability`` column for the identical rows, boxes and recall ceiling. Nothing in this
package loads a detector — the appearance features are RE-EXTRACTED from the 0.4 mm iso
cache at each candidate's frozen ``cen_d*``/``ext_d*`` (Inv. 5).

Torch-free modules (importable on the laptop, unit-tested locally):
``crops``, ``crop_aug``, ``tokens`` (feature-matrix core), ``geometry_bias``
(descriptor core), ``losses``, ``variants``, ``evaluate``.
Torch-only modules (server): ``encoder``, ``setmodel``, ``train``, ``cost``, and the
``nn.Module`` classes inside ``tokens``/``geometry_bias``.
"""
