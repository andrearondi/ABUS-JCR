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

# ============================================================================
# Phase 2 — RetinaNet 2.5D detector constants (single source of truth)
# ----------------------------------------------------------------------------
# Two groups. (A) The fixed recipe + the pinned derivation RULE (how [2.0]'s Train
# stats map to the data-dependent constants) — split-independent, frozen forever.
# (B) The data-dependent constants FROZEN from the [2.0] Train probe — written here
# as PROVISIONAL placeholders (the Val-measured ballpark) and reconciled against the
# Train probe output BEFORE any training run: if derive_constants(Train) reproduces
# them they stand; else they are replaced by the Train-derived values and the change
# is recorded in RESULTS_PHASE_2.md [2.0]. No training run may precede reconciliation.
# ============================================================================

# --- Phase 2 (A): fixed recipe + the pinned derivation rule -----------------
DET_BACKBONE            = "retinanet_resnet50_fpn_v2"     # torchvision; COCO_V1 weights
DET_NUM_CLASSES         = 2        # torchvision convention: background(0) + lesion(1). Matches Faster R-CNN (Phase 6)
DET_FG_LABEL            = 1        # foreground (lesion) label used in targets
# Derivation-rule parameters (consumed by detect.det_stats.derive_constants on the [2.0] Train stats):
DET_RULE = {
    "min_size_round": 32,          # min_size = round_up(max Train ISO d0-frame, 32)  (near-native depth; no downscale)
    "max_size_round": 32,          # max_size = round_up(max Train ISO d1-frame, 32)
    "anchor_diag_lo_pct": 1,       # smallest anchor scale ~ p1 of Train lesion-box diag (iso px)
    "anchor_diag_hi_pct": 99,      # largest anchor scale*2^(2/3) must cover p99 of diag
    "anchor_n_levels": 5,          # P3..P7; bases geometric (ratio 2), rounded; x 3 sub-octaves at build
    "aspect_pcts": [10, 50, 90],   # aspect_ratios = {h/w at these Train pcts} snapped to grid, U {1.0}
    "aspect_grid": [0.2, 0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0],
    "anchor_min_base": 16,         # [P2-UPDATE B3] floor the smallest base at 16 even if union boxes lift
                                   # diag-p1, so the small-lesion tail keeps anchor coverage.
    "intensity_sample_slices": 4000,  # seeded sample of Train iso slices for image_mean/std (float32 [0,1])
    "intensity_seed": 0,
}
# [P2-UPDATE B1] Per-slice 2D GT boxes enclose each LESION, not each raw component (Inv. 11 amended):
# mask components whose bounding boxes are within DET_LABEL_MERGE_GAP iso px are unioned into one box
# (speckle/shadow fragments of one lesion); genuinely separate foci stay distinct. gap=inf -> global
# union; gap=0 -> per-component (old behaviour). Train-validated at [2.0'] (keep case-93 foci separate).
DET_LABEL_MERGE_GAP     = 8
# [P2-UPDATE B2] Anchor<->GT matcher thresholds (were torchvision defaults 0.5/0.4, never set -> the
# positive-starvation bug). Loosened so real small/mid boxes clear the fg bar with several anchors.
DET_FG_IOU_THRESH       = 0.4      # anchor IoU >= this -> positive (was default 0.5)
DET_BG_IOU_THRESH       = 0.3      # anchor IoU <  this -> negative (was default 0.4)
# Diagnostic-dump inference knobs (NOT the Phase-3 operating point; permissive so the dump is informative)
DET_DIAG_SCORE_THRESH   = 0.05
DET_DIAG_NMS_THRESH     = 0.5
DET_DIAG_DETECTIONS_PER_IMG = 300
# Training regime (split-independent)
DET_NEG_POS_SLICE_RATIO = 2        # [P2-UPDATE B6] background:lesion slices/epoch (was 3; ~2:1 enriches
                                   # positive exposure vs the ~1:9.8 natural rate; watch FP at [2.4']).
DET_FOLD_SEED           = 0        # single seed for the 5 k-fold candidate detectors (Inv. 10: they only make data)
DET_FULL_SEEDS          = (0, 1, 2)# 3 seeds for the full-train deployment detectors (Inv. 14: reported mean±std)
DET_OPTIMIZER           = {"name": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4}
DET_LR_SCHEDULE         = {"warmup_iters": 500, "warmup_factor": 0.01, "kind": "cosine"}
DET_MAX_EPOCHS          = 50
# [P2-UPDATE B5] Model selection is on a DETECTION metric, not val-loss (Inv. 2 amended). The metric is
# a 2D CPM-proxy: mean per-slice recall at the FROC FP/slice budgets (a pre-linking foreshadow of the
# Inv.-3 CPM = mean recall at fixed FP operating points). Better-aligned than generic AP for a candidate
# generator and discriminative where per-volume recall saturates. val_ap + val_loss stay logged.
DET_SELECTION_METRIC    = "val_cpm_proxy_2d@0.3"
DET_SELECTION_FP_BUDGETS = (0.125, 0.25, 0.5, 1, 2, 4, 8)  # per-slice FP budgets (mirror KEY_FP; per-slice
                                   # at Phase 2 pre-linking, not per-volume). Selection = mean recall over these.
DET_EARLYSTOP_PATIENCE  = 10       # epochs of no CPM-proxy improvement (was 8 on val-loss; the metric is noisy)
DET_AP_IOU_THRESH       = 0.30     # IoU threshold for the selection metrics (CPM-proxy + logged AP)
DET_BATCH_SIZE          = 8        # slices per step; A6000 48 GB, ~160x352 input (VRAM-probe for 16 in RB)
DET_PER_SLICE_RECALL    = {"score_thresh": 0.05, "iou_thresh": 0.30}  # 2D diagnostic recall readout

# --- Phase 2 (B): FROZEN FROM THE [2.0] TRAIN PROBE (reconciled 2026-07-08) ---
# [P2-UPDATE B3] STALE PENDING RE-RECONCILIATION: these were derived from the OLD per-component boxes.
# After the union-box regen, RB_PHASE_2_UPDATE [2.0'] re-runs phase2_train_stats.py on the corrected
# boxes and overwrites the anchor_* fields below (record before/after in RESULTS_PHASE_2_UPDATE). Expect
# base sizes to shift up (fragment diag-p1=1.4 floor removed); aspect ratios stay wide (already correct).
# RECONCILED against phase2_train_stats.py on the 100-case Train split (RESULTS_PHASE_2 [2.0b]).
# These ARE the Train-derived design constants; the earlier Val ballpark survives in comments only.
# Reconciliation: min_size/max_size/image_mean/anchor_base_sizes matched the provisional; image_std
# moved 0.16->0.1658 (Val guess -> exact Train stat, was within tolerance); anchor_aspect_ratios
# moved (0.25,0.5,1.0,2.0)->(0.2,0.25,0.5,1.0) (Val guess wrongly included tall 2.0 and dropped the
# wide 0.2 — Train h/w p10/p50/p90=0.161/0.250/0.442 is wide-skewed).
DET_MIN_SIZE            = 160      # round_up(max Train ISO d0-frame=158, 32). Val ballpark: d0==158
DET_MAX_SIZE            = 352      # round_up(max Train ISO d1-frame=341, 32). Val ballpark: d1<=341
DET_IMAGE_MEAN          = 0.23     # Train iso-slice mean (float32 [0,1]); per-channel uniform. Val: 0.228
DET_IMAGE_STD           = 0.1658   # Train iso-slice std (n=4000 seeded slices); per-channel uniform. Val: 0.160
DET_ANCHOR_BASE_SIZES   = (16, 32, 64, 128, 256)   # Train diag p1..p99 via the grow-to-cover rule. Val: p5=9..p95=139
DET_ANCHOR_ASPECT_RATIOS = (0.2, 0.25, 0.33, 1.0)   # Train h/w p10/p50/p90=0.161/0.250/0.442 snapped +{1.0}

# ============================================================================
# Phase 3 — recall-saturated candidate generation + the fixed 3D aggregation.
# ============================================================================
# --- Phase 3 (A): fixed 3D aggregation — FROZEN once, reused for ALL detectors (Inv. 4) ---
LINK_IOU            = 0.30   # 2D IoU to continue a tube into the adjacent slice
LINK_MAX_Z_GAP      = 1      # bridge up to this many non-firing slices within one tube
LINK_MIN_TUBE_LEN   = 2      # discard tubes shorter than this many boxes (single-slice spikes)
LINK_SCORE_AGG      = "max"  # per-tube baseline score = peak per-slice score (Inv./brief: committed)
# --- Phase 3 (B): candidate-generation operating point (per-slice read-off; calibrated on VAL) ---
LINK_NMS_THRESH        = 0.70   # loosened per-slice NMS (> DET_DIAG_NMS_THRESH 0.5); NOT disabled (Inv. 2)
LINK_DETECTIONS_PER_IMG = 500   # per-slice cap feeding the linker (> DET_DIAG 300)
LINK_OP_SCORE_THRESH   = 0.05   # PROVISIONAL; frozen at the VAL linked-recall knee in [3.4], RECORD final
PREFILTER_SCORE_FLOOR  = 0.0    # optional tube score_max floor (NoduleSAT-style); raised only to meet the
                                # pool budget; RECORD its recall cost (0.0 = off)
CANDIDATE_POOL_BUDGET  = 150    # soft target candidates/volume for the Phase-4 3D encoder (escape valve)
# --- Phase 3 (C): candidate labeling (Inv. 11 ignore-band) — reuses geometry.iou_official (== iou_3d) ---
LABEL_POS_IOU = 0.30            # candidate IoU with official GT box > this -> positive
LABEL_NEG_IOU = 0.10            # candidate IoU < this -> negative; [0.10, 0.30] -> ignore (dropped)
# --- Phase 3 (D): GT reconstruction-consistency tolerance (driven by Phase-1 measured fidelity) ---
RECON_IOU_WARN_FRAC = 0.90     # >= this fraction of Train cases must clear 0.85 recon IoU
RECON_IOU_SOFT      = 0.85     # typical-case target (Val median 0.942, Train median 0.936)
# RECON hard floor is the existing RESAMPLE_IOU_FLOOR (0.50); any case below it = a linking/coord BUG.
# --- Phase 3 (E): Phase-0b FP-structure probe ---
FP_PROBE_ANISO_DEPTH_AXIS = 0  # d0 = depth/beam; anisotropy = extent_d0 / mean(extent_d1, extent_d2)
FP_PROBE_CLUSTER_RADIUS   = 10.0  # iso-voxel single-linkage radius for the FP-cluster count

# The score-statistics vector column names are FROZEN (consumed verbatim by Phase 4).
SCORE_STAT_COLUMNS = ["score_max", "score_mean", "score_std", "score_min",
                      "slice_count", "z_span", "fill_ratio"]  # fill_ratio = slice_count / z_span
