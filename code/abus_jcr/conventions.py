"""Single source of truth for coordinate conventions, schemas, and thresholds.

No magic numbers anywhere else in the package: every geometry/metric constant
is defined here and imported. These values are *re-verified* locally on the
Validation split (see ``scripts/phase0a_*`` and the double-check test), never
assumed.

Coordinate spaces (Inv. 6 — one space):
- **Storage order** ``(d0, d1, d2)`` — NRRD/numpy array indexing. ``d0`` is the
  largest axis for this dataset. Internal representation for masks/candidates.
- **Official/ITK order** ``(x, y, z)`` — the space ``det_score.py`` scores in.
  Reached from storage by the self-inverse permutation ``PERM_STORAGE_TO_ITK``.

The official scoring space is **native voxel indices, ITK (x, y, z) order,
centre + full extent**.
"""

from __future__ import annotations

# --- axis permutation -------------------------------------------------------
# storage axis (d0, d1, d2) -> ITK (x, y, z). Self-inverse: applying it twice
# is the identity, so the same tuple maps official -> storage as well.
PERM_STORAGE_TO_ITK = (2, 1, 0)

# --- physical spacing (INJECTED, never read from the NRRD header) -----------
# The NRRD header carries an identity-matrix placeholder for spacing; it MUST
# be ignored. Real spacing comes from the official challenge description.
# Order is storage (d0, d1, d2) millimetres.
SPACING_STORAGE_MM = (0.073, 0.200, 0.475674)
# => ITK order (x, y, z) = (0.475674, 0.200, 0.073)

# --- CSV schemas ------------------------------------------------------------
# Official column names read by det_score.py (GT) and written for predictions.
GT_COLUMNS = ["public_id", "coordX", "coordY", "coordZ", "x_length", "y_length", "z_length"]
PRED_COLUMNS = GT_COLUMNS + ["probability"]  # probability in [0, 1)

# The shipped bbx_labels.csv uses documented (non-official) names; this maps
# them to the official schema. `id` is the integer case id == DATA/MASK case_id.
GT_RENAME = {
    "id": "public_id",
    "c_x": "coordX",
    "c_y": "coordY",
    "c_z": "coordZ",
    "len_x": "x_length",
    "len_y": "y_length",
    "len_z": "z_length",
}

# --- metric thresholds (mirror det_score.py) --------------------------------
IOU_HIT_THRESHOLD = 0.3  # strict '>' — a hit iff max_iou > 0.3
KEY_FP = (0.125, 0.25, 0.5, 1, 2, 4, 8)

# --- lesion audit (descriptive only) ---------------------------------------
# Floor separating genuine lesions from sub-voxel mask specks. Affects no GT
# box, no model, no label — raw counts are always reported alongside.
LESION_MIN_VOXELS = 1000

# --- Phase 1: isotropic cache + slice contract ------------------------------
# The 2.5D detection frame (Inv. 1). Storage axis d2 = elevational/sweep is the
# stack axis; each slice is the (d0, d1) B-mode frame. d0 = depth/beam (image
# vertical, near-field at top) => NO vertical flip. d1 = lateral (image
# horizontal) => horizontal flip allowed (Inv. 13). Confirmed on Val overlays.
SLICE_AXIS = 2
IN_PLANE_ROW_AXIS = 0  # d0 = depth/beam -> image "y"/row; NO vertical flip
IN_PLANE_COL_AXIS = 1  # d1 = lateral    -> image "x"/col; horizontal flip OK

# One isotropic space, cached once (Inv. 6). 0.4 mm target (chosen over the 0.5 mm
# default after the [1.7] fidelity sweep: 0.5 mm left a small-lesion tail — 20/100
# Train cases had a perfect-candidate round-trip IoU below 0.85, min 0.576, driven
# by the ~6.85x depth-axis downsample. 0.4 mm nearly clears that tail — Val 5->1
# case below 0.85, min 0.750->0.817 — at ~2x voxels/slices, preserving more IoU
# budget for real candidates. Changing this invalidates the cache (preprocess_hash).
ISO_SPACING_MM = 0.4

# uint8 -> float32 [0,1]; identical across all three detectors. Cache-invalidating.
INTENSITY_NORM = {"method": "scale", "divisor": 255.0}

# scipy.ndimage.zoom parameters. image: linear (order 1); mask: nearest (order 0)
# so it stays {0,1}. grid_mode + grid-constant is edge-aligned so physical extent
# is preserved (n_out = round(n_in * f)). Cache-invalidating.
RESAMPLE = {"image_order": 1, "mask_order": 0, "grid_mode": True, "mode": "grid-constant"}

# 2.5D stack width (centre +/- 1). Maps 1:1 to pretrained 3-channel stems.
# A *dataloading* param, NOT cache-invalidating.
C_CHANNELS = 3
EDGE_SLICE_POLICY = "clamp"  # replicate boundary slices by index-clamping

# Keep every non-empty 2D mask component (box-set == mask-set exactly). OFF by default.
MIN_2D_BOX_AREA = 0

# k-fold for out-of-fold rescorer candidates (Inv. 10). Stratified by B/M, seeded.
KFOLD_K = 5
KFOLD_SEED = 0
KFOLD_STRATIFY_BY = "label"

# [1.7] gate semantics (recalibrated). This is a FROC-hit SAFETY MARGIN, not a
# fidelity target: a perfect-localisation candidate round-tripped iso->native must
# retain IoU > this against the native official box, staying comfortably above the
# 0.3 hit threshold (Inv. 3) so the resampling never eats a real candidate's hit
# budget. 0.50 = 1.67x the hit threshold; Train min at 0.5 mm was already 0.576 and
# 0.4 mm only lifts it, so this is a safe regression tripwire (a coordinate/affine
# bug would tank it) that no legitimate small-lesion case trips. The ACTUAL ceiling
# distribution (min/median/percentiles) is characterised, not asserted, by
# scripts/phase1_resample_fidelity.py and handed to Phase 3 as its tolerance input.
RESAMPLE_IOU_FLOOR = 0.50
