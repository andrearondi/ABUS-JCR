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

import os

# ============================================================================
# AXIS PROFILE — which spacing map the cache is built with (added 2026-08-09)
# ----------------------------------------------------------------------------
# Two substrates coexist in this repo. Selected ONCE per process by the
# environment variable ``ABUS_AXIS_PROFILE``; there is no CLI flag, on purpose —
# a constant that names a substrate must not be settable per-script.
#
#   "legacy"   (DEFAULT) — the DEPLOYED arm. `SPACING_STORAGE_MM` keeps the
#              inverted in-plane roles, so `preprocess_hash` stays
#              ab1fdf28... and every recorded number reproduces byte-for-byte.
#              Nothing about this profile changed when the profile switch was
#              introduced; it is the file as it stood, unmodified.
#   "measured" — the ISOTROPIC-CACHE branch (`iso/RB_ISO_REBUILD.md`). The
#              spacing map is the one measured on 129/130 volumes
#              (results/AXIS_CHECK.md), so the cache really is 0.4 mm cubed.
#              A DIFFERENT `preprocess_hash` ⇒ a different cache directory ⇒
#              `cache.assert_hash` refuses to read one profile's cache under the
#              other. That guard is what makes a forgotten `export` fail loudly
#              on the first cache read instead of silently training on the wrong
#              substrate.
#
# Every value this profile changes is collected in ONE block at the END of this
# file, so the deployed values below stay where they have always been, with
# their original provenance comments intact and auditable.
AXIS_PROFILE = os.environ.get("ABUS_AXIS_PROFILE", "legacy").strip().lower()
if AXIS_PROFILE not in ("legacy", "measured"):
    raise RuntimeError(
        f"ABUS_AXIS_PROFILE={AXIS_PROFILE!r} is not a known axis profile; "
        "use 'legacy' (deployed) or 'measured' (isotropic rebuild)."
    )

# --- axis permutation -------------------------------------------------------
# storage axis (d0, d1, d2) -> ITK (x, y, z). Self-inverse: applying it twice
# is the identity, so the same tuple maps official -> storage as well.
PERM_STORAGE_TO_ITK = (2, 1, 0)

# --- physical spacing (INJECTED, never read from the NRRD header) -----------
# The NRRD header carries an identity-matrix placeholder for spacing; it MUST
# be ignored. Real spacing comes from the official challenge description.
# Order is storage (d0, d1, d2) millimetres.
#
# *** THIS MAP IS KNOWN TO BE WRONG. IT IS RETAINED ON PURPOSE. ***  [2026-08-02]
# Measured on 129/130 volumes (30 Val + 100 Train, disjoint), four independent lines:
#     TRUE map = (0.200, 0.073, 0.475674)  ->  d0 = LATERAL, d1 = DEPTH/BEAM, d2 = sweep
# i.e. the two IN-PLANE axes are swapped below. d2 = sweep is correct, so SLICE_AXIS = 2
# and Inv. 1 stand. Evidence + reproduction: results/AXIS_CHECK.md, and
# results/RESULTS_INTENSITY_PROBE.md [I.3].
#
# Why it is not simply corrected here: this constant feeds `preprocess_hash`, so changing
# it invalidates the Phase-1 iso cache and every artefact built on it (all 8 detectors, both
# candidate pools, Phases 3-4). That is a costed decision, not a typo fix — see AXIS_CHECK.md
# §5-7. Reported CPM/FROC numbers are NOT affected: the metric runs in native ITK space and
# IoU is invariant under a per-axis (diagonal) scaling.
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
# stack axis; each slice is the (d0, d1) B-mode frame. That much is correct and
# measured: d2 = sweep, so SLICE_AXIS = 2 and Inv. 1 stand.
#
# *** THE TWO IN-PLANE ROLES BELOW ARE INVERTED. MEASURED 2026-08-02. ***
#   d0 is LATERAL  (image horizontal; a flip here is the legitimate L-R symmetry)
#   d1 is DEPTH/BEAM (image vertical, near-field at top; a flip here is FORBIDDEN by Inv. 13)
# The comment previously read "Confirmed on Val overlays" — no such confirmation exists
# anywhere in this repo, and DATA_INFO.md recorded the axis as "provisionally axis 0 — to be
# visually confirmed". The confirmation has now actually been done, on 129/130 volumes, and
# it came out the other way: results/AXIS_CHECK.md.
#
# CONSEQUENCE, NOT YET ACTED ON: the deployed augmentation flips the axis these constants
# call "lateral", which is really DEPTH — see augment.AUGMENT_POLICY and
# rescore/crop_aug.FLIP_AXIS, both of which resolve to d1. Correcting the values here
# silently changes both, so it is a retraining decision (AXIS_CHECK.md §5), not an edit.
SLICE_AXIS = 2
IN_PLANE_ROW_AXIS = 0  # DECLARED "d0 = depth/beam"; MEASURED d0 = lateral
IN_PLANE_COL_AXIS = 1  # DECLARED "d1 = lateral";    MEASURED d1 = depth/beam

# The PHYSICAL axis roles. These are facts about the scanner, not about the cache,
# so they are the SAME under both axis profiles — measured four independent ways on
# 129/130 volumes (results/AXIS_CHECK.md). Under the "legacy" profile they are
# deliberately INCONSISTENT with SPACING_STORAGE_MM above: that inconsistency IS the
# defect, and encoding the falsehood instead would hide it. Read these when you need
# to know which axis is depth; read SPACING_STORAGE_MM when you need to know what the
# cache was actually built with. (`FP_PROBE_ANISO_DEPTH_AXIS` below is a separate,
# profile-dependent constant whose only job on "legacy" is byte-reproduction of the
# recorded probe outputs — it is NOT the physical answer.)
DEPTH_AXIS   = 1  # d1 = depth/beam (0.073 mm native): skin at index ~0, shadows +d1
LATERAL_AXIS = 0  # d0 = lateral    (0.200 mm native): the ~L-R symmetry axis
SWEEP_AXIS   = 2  # d2 = elevational sweep (0.475674 mm native) == SLICE_AXIS

# *** THE CACHE IS NOT ACTUALLY ISOTROPIC. MEASURED 2026-08-02. *** `preprocess.zoom_factors`
# scales axis a by SPACING_STORAGE_MM[a] / ISO_SPACING_MM, so with the wrong spacing map above
# one cache voxel really spans ISO_SPACING_MM * true[a] / declared[a] mm:
#       1.0959 x 0.1460 x 0.4000 mm   (d0 lateral, d1 depth, d2 sweep)
# — an in-plane aspect error of (0.200/0.073)^2 ~ 7.5x. Inv. 6's "ONE coordinate space" holds
# (it is single and consistent, applied identically to images, GT boxes and candidates in every
# split, so IoU and the metric are unaffected); Inv. 6's "isotropic" does not. Any mm figure
# derived from cache voxels below is wrong by these factors. See results/AXIS_CHECK.md §5.
#
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
# [P3-UPDATE D6] Anchor<->GT assignment. "fixed" = the torchvision IoU Matcher above (DEFAULT).
# "atss" = adaptive per-GT threshold (abus_jcr/detect/atss.py) — fired ONLY if [D0] anchor coverage
# shows starvation or the Stage-1 gate underperforms. nnDetection ran BCE+ATSS on this dataset (0.7704).
DET_ASSIGNER            = "atss"   # {"fixed","atss"} — [P3U.3] D0 gate FIRED (2026-07-18): ~19% of Train
                                   # GT boxes get 0 fg-clearing anchors ([0,16) 99.8% zero, [16,32) 42.6%),
                                   # a structural fixed-IoU starvation D1-D5 can't touch. ATSS = training-only.
DET_ATSS_TOPK           = 9        # ATSS k (per FPN level); paper-flat over 7..19
# Diagnostic-dump inference knobs (NOT the Phase-3 operating point; permissive so the dump is informative)
DET_DIAG_SCORE_THRESH   = 0.05
DET_DIAG_NMS_THRESH     = 0.5
DET_DIAG_DETECTIONS_PER_IMG = 300
# Training regime (split-independent)
DET_NEG_POS_SLICE_RATIO = 2        # [P2-UPDATE B6] background:lesion slices/epoch (was 3; ~2:1 enriches
                                   # positive exposure vs the ~1:9.8 natural rate; watch FP at [2.4']).
DET_FOLD_SEED           = 0        # single seed for the 5 k-fold candidate detectors (Inv. 10: they only make data)
DET_FULL_SEEDS          = (0, 1, 2)# 3 seeds for the full-train deployment detectors (Inv. 14: reported mean±std)
# [P3-UPDATE D4] Optimiser = AdamW at the recipe that PRODUCED the retinanet_resnet50_fpn_v2 COCO weights
# (PR #5756: AdamW lr=1e-4, wd=0.05, --norm-weight-decay 0.0). The old SGD lr=0.01 was the *v1* recipe
# applied to a v2 model — a COCO-scale LR (118k images) on ~100 independent lesions. Per-architecture
# hyperparameters may differ (Inv. 2 A1); Phase-6 YOLO/Faster R-CNN recipes are independent.
DET_OPTIMIZER           = {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.05}
DET_NORM_WEIGHT_DECAY   = 0.0      # exempt norm-layer affine + biases from weight decay (v2 --norm-weight-decay 0.0)
DET_LR_SCHEDULE         = {"warmup_iters": 500, "warmup_factor": 0.01, "kind": "cosine"}
# [P3-UPDATE D2] Fixed, properly-annealed schedule (cosine LR -> ~0 over DET_TRAIN_EPOCHS) with NO
# metric-based early stopping. The old (MAX_EPOCHS=50, EARLYSTOP_PATIENCE=10) pair stopped ~epoch 11-19,
# always at ~98% of peak LR, so the deployed checkpoint never saw an annealed convergence phase.
DET_TRAIN_EPOCHS        = 30       # fixed budget; cosine reaches ~0 here. Every epoch is saved to disk.
DET_MAX_EPOCHS          = DET_TRAIN_EPOCHS   # back-compat alias (retired; use DET_TRAIN_EPOCHS)
# [P3-UPDATE A1/D3] Model selection = POST-HOC, once, among the converged epochs, on the TRUE linked 3D
# val CPM (Inv.-3 average_recall via the Phase-3 detect->link->oracle path). NOT a per-epoch per-slice
# proxy. The per-slice CPM-proxy below is kept as a LOGGED DIAGNOSTIC only (it drives nothing) — its
# per-slice FP budgets span 50-3254 FP/volume, a regime the Inv.-3 metric (0.125-8 FP/vol) never visits.
DET_SELECTION_METRIC    = "val_linked_cpm_3d@0.3_posthoc"
# [P3-UPDATE D3, revised 2026-07-19] The AdamW recipe converges by ~epoch 6 then OVERFITS (val-loss
# U-shape), so the old min_epoch=15 floor selected the overfit tail. Floor lowered to 3 (skip only the
# pre-convergence epochs 0-2); the selector still EVALUATES + prints epochs from 0 for completeness.
DET_SELECT_MIN_EPOCH    = 3        # earliest epoch eligible for SELECTION (0-2 shown but not selectable)
DET_SELECT_CPM_TOL      = 0.10     # [Inv. 2 / A1-tau AMENDMENT, ADOPTED 2026-08-08] was 0.02.
                                   # Among epochs whose CPM is within this of the max AND whose post-floor
                                   # pool <= RESCORER_POOL_BUDGET, take the highest recall ceiling, then the
                                   # higher CPM, then the earliest epoch. BOTH regimes (fold + full-seed).
                                   # 0.02 was a NOISE width ("CPM moves in ~1/30 steps on 30 val lesions");
                                   # the rule is also asked "how much CPM will I pay for recall?", which is a
                                   # PREFERENCE. One constant was doing both jobs. Recall is what Inv. 8 makes
                                   # unrecoverable downstream, and A2 was already recall-first while A1 was
                                   # CPM-first -- so this removes an inconsistency, it does not add a novelty.
                                   # Measured (RESULTS_FOLD_FLIP [F.6b]): fold coverage 77 -> 81/100 and seed
                                   # ceiling 75 -> 77/90 vs tau 0.02, pools unchanged, -0.027 seed CPM. A NO-OP
                                   # on the pre-2026-08-08 arm (its CPM and ceiling are co-monotone) and on 2 of
                                   # 3 promoted seeds. Declared SECONDARY read: tau_seed = 0.02 (the max-CPM
                                   # checkpoint = the "detector as normally used" Inv.-8 baseline).
                                   # Forking-path caveat: AUGMENTATION_NOTES.md 14.7 -- adopted 2026-08-05 after
                                   # the 14.3 simulation showed its effect, and it flips the promote outcome.
                                   # Not headroom-gaming: B0' RISES to 0.6327 and headroom SHRINKS 0.310 -> 0.223.
DET_SELECT_OP_THRESH    = 0.03     # reference op for post-hoc selection. NOTE (P3U): the retrained detector
                                   # de-compressed its scores, so 0.03 sits ABOVE its recall knee — after
                                   # [P3U.4b] reveals the saturating op, set this to it and re-select Stage-2.
# [P3U3 2026-07-29] CONTINGENCY selection policy — **NOT DEPLOYED**. The deployed rule is unchanged
# (DET_SELECTION_METRIC above: max linked-val CPM with the ceiling tie-break). These two constants
# parameterise `detect.select.select_epoch_guarded`, used ONLY to build the side-by-side high-coverage
# TRAINING pool (candidates_train_hicov) held in reserve for Phase 4. Promoting it to the deployed rule
# would require an Inv. 2 / A1 amendment. Simulated effect (P3U2.SIM): fold coverage 0.697 -> 0.807,
# fold CPM -0.042, ALL SEEDS UNCHANGED (evaluation/B0'/operating point untouched), max pool 187 <= budget.
DET_SELECT_COVERAGE_FLOOR = 0.80   # min recall ceiling an epoch must reach to qualify (matches the seeds'
                                   # 0.867 evaluation coverage -> Inv.-10 train/eval distribution match)
DET_SELECT_CPM_GUARD      = 0.15   # max CPM shortfall vs the run's best epoch; blocks degenerate
                                   # unconverged checkpoints (fold4 ep5: CPM 0.14 vs 0.41 max). 0.20 is
                                   # equivalent on the recorded tables (robustness plateau); 0.10 is too tight.
DET_SELECTION_FP_BUDGETS = (0.125, 0.25, 0.5, 1, 2, 4, 8)  # per-SLICE FP budgets for the LOGGED diagnostic
                                   # proxy only (mirror KEY_FP's numbers; NOT the Inv.-3 per-volume metric).
DET_EARLYSTOP_PATIENCE  = 10       # RETIRED (P3-UPDATE D2): no early stopping. Kept for back-compat only.
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
LINK_MIN_TUBE_LEN   = 8      # [P3U2.7 FROZEN 2026-07-24] largest recall-neutral on the 5 OOF fold detectors:
                            # fold recall 0.720 flat for min_len 2..8 (pool 135.8 -> 94.0, ~30% free cut),
                            # drops at 10. Sound: TP slice_count p10=18 >> FP p90=13 ([P3U2.diag] §6).
                            # Precedent: Oh et al. 2023 ABUS 2D-link uses tau_s=5.
LINK_SCORE_AGG      = "max"  # per-tube baseline score = peak per-slice score (Inv./brief: committed)
# [P3-UPDATE L1] Tube drift/length caps — the pre-P3-UPDATE linker random-walked with NO cap and
# reconstructed an unbounded union hull, producing whole-volume "candidates" (box_diag max 1052.9 vs the
# 1154.1 volume diagonal; z_span max 414 in a ~407-slice volume) that consumed real lesions' boxes and
# made linked recall NON-MONOTONE in the threshold. Both caps are Train-GT-derived at [3.3'] (no leakage,
# not per-detector — Inv. 4). None = uncapped (asserted-set before the frozen generation run).
LINK_MAX_TUBE_ZSPAN     = 182   # set at [3.3']: round(1.8 * Train GT z-extent p99, iso slices)
LINK_MAX_CENTROID_DRIFT = 342   # set at [3.3']: round(1.5 * Train GT in-plane extent p99, iso px)
# [P3U2 3.C] Membership-only 3D NMS over reconstructed candidates: keep the highest-score_max box in
# each 3D-IoU cluster, drop the rest (coordinates UNCHANGED). DEFAULT OFF (None) after [P3U2.diag]: the
# per-candidate score floor (PREFILTER_SCORE_FLOOR) already brings the pool to ~56/vol at recall 0.933, so
# no dedup is needed for the budget — and collapsing the ~15 redundant TP tubes/vol would DISCARD distinct
# 3D crops + the consensus signal the Phase-4 set/geometry rescorer can exploit. Kept as an OPTIONAL Phase-4
# ablation ("does de-duplicating help or hurt the rescorer?"), not a baked-in reducer. If ever enabled, it is
# frozen on the OOF fold detectors at [P3U2.7] (Inv. 4). The reducer machinery + gate remain for that ablation.
LINK_3DNMS_IOU          = None  # OFF by default; the score floor is the pool lever (see PREFILTER_SCORE_FLOOR).
# --- Phase 3 (B): candidate-generation operating point (per-slice read-off; calibrated on VAL) ---
LINK_NMS_THRESH        = 0.70   # PROVISIONAL: swept {0.5,0.6,0.7} at [3.3'], freeze the recall-neutral min
                                # (MONAI medical RetinaNet uses 0.22; 0.5 is conservative). Loosened, not
                                # disabled (Inv. 2). Still > DET_DIAG_NMS_THRESH 0.5 until re-frozen.
LINK_CONTAINMENT_THRESH = 1.0   # [P3U2.set 2026-07-21] OFF (>=1.0 = no-op). Provisional: relaxed from 0.80 so
                                # the over-trimmed distinct lesions (110/121/122) rejoin the pool (recall 0.833
                                # -> 0.933 on seed0); the score floor cuts the resulting FP tail. FROZEN on the
                                # OOF fold detectors at [P3U2.7]. [P3-UPDATE L4] semantics: drop a lower-score
                                # box if inter/area_small >= this vs a higher-score box (kills nested duplicates).
LINK_DETECTIONS_PER_IMG = 500   # per-slice cap feeding the linker (> DET_DIAG 300)
LINK_OP_SCORE_THRESH   = 0.03   # [P3U2.8 FROZEN 2026-07-25] ranking-aware VAL operating point: among thresholds
                                # with recall >= 0.98*max, the max-CPM one. 3-seed mean @ 0.03: recall 0.867
                                # (std 0.047), CPM 0.553, 101.2 cands/vol (<= budget 200). == DET_SELECT_OP_THRESH.
                                # [RB_AUG_FLIP_AB A.4f, 2026-08-04] CONFIRMED by the first JOINT (op x epoch)
                                # search — [P3U2.8] swept the op with the epochs held FIXED, so op and epoch had
                                # only ever been optimised sequentially. Re-selecting every epoch at op 0.02
                                # leaves the deployed arm preferring 0.03: dCPM +0.0018 (a tie inside
                                # DET_SELECT_CPM_TOL) for -4 lesions of recall ceiling (0.8667 -> 0.8222) and
                                # 2.6x the CPM variance (std 0.0146 -> 0.0376). 0.03 is confirmed, not inherited.
                                # NOTE: the A2 rule that picked it is not well-posed on a MONOTONE recall curve —
                                # see INVARIANTS_AMENDMENTS_PROPOSED.md (dormant; deployed linker is non-monotone).
PREFILTER_SCORE_FLOOR  = 0.08   # [P3U2.7 FROZEN 2026-07-24] LUNA/NoduleSAT-style per-candidate score_max floor:
                                # drop tubes whose peak per-slice score < this, applied BEFORE the 3D NMS, in
                                # EVERY pool path (generate, linked_recall, select, calibrate, reducer gate). THE
                                # primary pool lever: the FP pool is a low-confidence tail (FP score_max median
                                # 0.045 vs TP 0.166; [P3U2.diag]). FOLD-VALIDATED at [P3U2.7]: on the 5 OOF fold
                                # detectors the floor costs only 1/100 lesion vs floor 0.0 (recall 0.730->0.720)
                                # while cutting the pool 613.6->135.8/vol -> NOT over-cutting the weak folds; the
                                # low fold ceiling (~0.72) is the 80-vol detectors themselves, not the floor.
# [P3U2 3.B] TWO pool numbers. The pool the Phase-4 O(n^2) set module consumes must be LOW HUNDREDS,
# not the ~2000 MEMORY ceiling: PHASE_4 §1.3 is designed for "tens", and NoduleSAT/Liao/PAIR-Former
# pre-filter to a small high-value set (the geometry/Axis-A signal dilutes and overfits at n~1000 on
# ~100 volumes). So the linker may run at a large RECALL pool, and the 3D NMS (3.C) + operating point
# bring the FROZEN pool down to RESCORER_POOL_BUDGET.
RESCORER_POOL_BUDGET   = 200    # [P3U2 3.B] hard-ish target for the FROZEN (post-3D-NMS) pool Phase 4
                                # consumes. DEFAULT 200 (low hundreds); user-vetoable (spec Open esc. #1).
LINK_RECALL_POOL_MAX   = 1800   # [P3U2 3.B] diagnostic ceiling on the PRE-reduction linker pool (holds the
                                # recall ceiling; NMS + op reduce it to RESCORER_POOL_BUDGET). Soft.
CANDIDATE_POOL_BUDGET  = RESCORER_POOL_BUDGET   # [P3U2] back-compat alias -> the post-NMS rescorer pool
                                # (calibration/ablation over-budget checks now target it). Was 1000 (L6);
                                # the "comfortable to ~2000" note was a MEMORY argument, not a design one.
RECALL_CEILING_FRAC    = 0.98   # [P3-UPDATE L5/A2] operating point = among thresholds with linked recall
                                # >= this * max, pick the one maximising linked val CPM (ranking-aware).
# --- Phase 3 (C): candidate labeling (Inv. 11 ignore-band) — reuses geometry.iou_official (== iou_3d) ---
LABEL_POS_IOU = 0.30            # candidate IoU with official GT box > this -> positive
LABEL_NEG_IOU = 0.10            # candidate IoU < this -> negative; [0.10, 0.30] -> ignore (dropped)
# --- Phase 3 (D): GT reconstruction-consistency tolerance (driven by Phase-1 measured fidelity) ---
RECON_IOU_WARN_FRAC = 0.85     # >= this fraction of Train cases must clear RECON_IOU_SOFT recon IoU.
                               # RECONCILED [3.2] 2026-07-15: the provisional 0.90 contradicted the
                               # spec's own Phase-1 input (measured 11/100 Train < 0.85 -> 89/100
                               # clear). The [3.2] gate reproduced that fidelity to the decimal
                               # (min 0.677, median 0.936, 11 < 0.85) with the native-hull control
                               # =1.0 on 100/100 and 0 below the 0.50 hard floor -> the tail is
                               # intrinsic 0.4mm depth-axis quantization, not a bug. 0.85 keeps a
                               # ~4-case margin below the pre-measured 0.89 so the gate still trips on
                               # a real coordinate regression. Not tuning-to-pass: the target was
                               # pre-registered in Phase 1 before this gate ran.
RECON_IOU_SOFT      = 0.85     # typical-case target (Val median 0.942, Train median 0.936)
# RECON hard floor is the existing RESAMPLE_IOU_FLOOR (0.50); any case below it = a linking/coord BUG.
# --- Phase 3 (E): Phase-0b FP-structure probe ---
# [2026-08-02] This constant is DECORATIVE — nothing reads it; `probe/fp_structure._anisotropy`,
# `probe/pool_diag.augment` and `rescore/tokens._block_abs_geom` all hardcode d0. Its stated
# meaning is also wrong: d0 is LATERAL, so every recorded `anisotropy` number measures LATERAL
# elongation in a cache whose in-plane aspect is off by ~7.5x. The near-null effect sizes on
# record (|delta| 0.014-0.358) are therefore about the wrong axis, and do NOT show that
# depth-elongated candidates are absent. That was finally tested at [I.6b]: elongation about
# the true beam axis separates TP/FP at |delta| ~0.27 (7/8 detectors), but NO candidate in
# either pool exceeds 2x beam-elongation — there is no ray-shaped population. A physically
# cubic candidate reads 0.195 on the deployed scale, not 1.0.
FP_PROBE_ANISO_DEPTH_AXIS = 0  # d0 = LATERAL (declared "depth/beam"); see above. Unused.
FP_PROBE_CLUSTER_RADIUS   = 10.0  # iso-voxel single-linkage radius for the FP-cluster count

# The score-statistics vector column names are FROZEN (consumed verbatim by Phase 4).
SCORE_STAT_COLUMNS = ["score_max", "score_mean", "score_std", "score_min",
                      "slice_count", "z_span", "fill_ratio"]  # fill_ratio = slice_count / z_span
# [P3U2 3.D] Tube-geometry feature block — a NEW, SEPARATE ablatable block for Phase 4 (§1.2),
# added before the pool is generated. Kept apart from the frozen SCORE_STAT_COLUMNS above so the
# existing score-stats vector stays byte-stable and the two blocks ablate independently.
# Pruned at [P3U2.diag]: area_peak_pos (no TP/FP signal) and area_monotonicity (length-confounded,
# redundant with slice_count/area_cv) — both dropped; the two survivors validated as discriminative.
TUBE_GEOM_COLUMNS = ["centroid_jitter", "area_cv"]

# ============================================================================
# Phase 4 — RG-FROC-Rescorer (set rescorer over the frozen Phase-3 pool)
# ============================================================================
# --- 4 (A): the 3D crop the shared encoder sees (Inv. 5, 6) ------------------
# ADAPTIVE CUBIC ROI, not a fixed 48^3 voxel box. PHASE_4.md §1.1 offers 48^3 (NoduleSAT)
# "unless the size audit shows lesions routinely exceeding this" — it does: Train union-box
# in-plane diag p1/p50/p99 = 6.7/54.1/231.1 iso px (RESULTS_PHASE_2_UPDATE [2.0']) and the
# Train GT in-plane extent p99 = 228 iso px / z-extent p99 = 101 iso slices (the [3.3'] cap
# derivation). A 48-voxel (19.2 mm) box is smaller than the MEDIAN lesion's in-plane diagonal.
# Local check (6 Val cases): GT iso extents d0 8-54, d1 40-183, d2 16-86 -> a fixed 48^3 crop
# fails to contain the lesion in 5/6 cases. So the ROI is sized from the CANDIDATE's own iso
# extents and resampled to a fixed grid (PHASE_4.md's second sanctioned option).
RESC_CROP_OUT        = 48    # output grid, 48^3 (the NoduleSAT tensor size is kept)
RESC_CROP_CONTEXT    = 1.5   # ROI side = CONTEXT * max(ext_d0, ext_d1, ext_d2)
RESC_CROP_MIN_SIDE   = 48    # iso vox (19.2 mm): never UPSAMPLE (resample factor <= 1 always)
RESC_CROP_MAX_SIDE   = 160   # iso vox (64 mm) == DET_MIN_SIZE == round_up(max Train iso d0
                             # frame, 32). An ROI deeper than the volume itself can only add
                             # zero padding, never real context.
RESC_CROP_INTERP     = 1     # trilinear (scipy.ndimage.map_coordinates order=1)
RESC_CROP_PAD_VALUE  = 0.0   # outside the volume = anechoic black; NEVER edge-replication
                             # (replicating the skin line downward is acoustically impossible)
# --- 4 (B): the shared 3D encoder (Inv. 5) ----------------------------------
RESC_ENCODER         = "monai_densenet121_3d"   # PHASE_4.md §1.1 default (NoduleSAT backbone)
RESC_ENCODER_DROPOUT = 0.2
RESC_D_APP           = 128   # appearance embedding width after GAP + linear
RESC_ENC_EPOCHS      = 30    # fixed annealed budget, save every epoch, post-hoc selection
RESC_ENC_BATCH       = 32
RESC_ENC_OPT         = {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.05}
                       # `mirror_lateral_p` = p of the ONLY permitted mirror, along `d0` = the
                       # MEASURED LATERAL axis (rescore/crop_aug.FLIP_AXIS = 0). There is
                       # deliberately NO flip along d1 (depth/beam) — Inv. 13.
                       # RENAMED 2026-08-09 from `hflip_d1_p`, which named the axis it must never
                       # touch. No recorded config changes, because no Phase-4 run has produced one
                       # ([4.1] onward were empty when this was renamed) — and HISTORY.md §8's own
                       # lesson is that routing a flip through a name that lies about its axis is
                       # exactly what caused the Inv.-13 defect. Free to fix now; not later.
RESC_ENC_AUG         = {"mirror_lateral_p": 0.5, "centre_jitter_frac": 0.10}
# Pre-registered FALLBACK encoder (spec Open escalation #2). NOT deployed. Fires only if exit
# check 4 fails (B1 val CPM <= the MEASURED B0 of the deployed pool): swap to a 4-block
# ~1M-param 3D CNN and re-run [4.3]. The floor is measured at run time, never hard-coded — see
# the 4 (H) note below.
RESC_ENCODER_FALLBACK = "small_cnn_3d"
RESC_SMALL_CNN_WIDTHS = (16, 32, 64, 128)
# --- 4 (C): the per-candidate token (§1.2) ----------------------------------
RESC_RANK_PE_DIM     = 16    # sinusoidal embedding width of the integer `rank`
RESC_TOKEN_BLOCKS    = ("appearance", "abs_geom", "score_stats", "tube_geom", "rank")
# --- 4 (D): the set module (§1.3) -------------------------------------------
RESC_SET_CAPACITY_GRID = (("L2H128h4", 2, 128, 4), ("L2H256h8", 2, 256, 8))  # selected ONCE on B2
RESC_SET_DROPOUT     = 0.1
RESC_SET_EPOCHS      = 60
RESC_SET_BATCH_SETS  = 8     # sets per step, padded + masked
RESC_SET_OPT         = {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01}
RESC_MAX_SET_SIZE    = 576   # CAP on the largest set, asserted at [4.1]. RAISED 320 -> 576 on
                             # 2026-08-09: on the PROMOTED pool the worst single set is train
                             # fold0 vol14 at **509** candidates (then 355/321/292/255) and val's
                             # worst is **292** ([F.7]; RESULTS_PHASE_4 pre-registration). 320 was
                             # sized for the ARCHIVED pool's max of 253 ([P3U2.10]) and would have
                             # failed [4.1] on the first run. NOT a memory knob: batches pad to the
                             # BATCH max, so this constant only forbids a pool the pipeline was
                             # never sized for. `RESCORER_POOL_BUDGET` (200) constrains the MEAN
                             # pool, not the max — the two are different quantities
                             # (AUGMENTATION_NOTES §14.6). 509^2 ~ 259k attention entries is
                             # trivial for the set module; do NOT re-tune anything for it.
RESC_NEG_POS_RATIO   = None  # NO negative subsampling: train sets == eval sets exactly, and the
                             # listwise loss is imbalance-robust to >=1:10 000 (RS-loss, ICCV'21
                             # Tab. S7). If Phase 6 ever needs it, the SAME ratio for all detectors.
# --- 4 (E): the Axis-A relative-geometry bias (§1.4) ------------------------
RESC_GEOM_MECHANISM  = "additive"   # iRPE-style per-head bias (DEFAULT); "multiplicative" fallback
RESC_GEOM_PE_DIM     = 64           # sinusoidal embedding of g(m,n)
RESC_GEOM_EPS        = 1e-6         # MUST equal probe/pool_diag.EPS (byte-identical descriptor)
# --- 4 (F): losses (§2) -----------------------------------------------------
RESC_LOSS_RANK       = "smooth_ap"  # Phase-0a single-lesion dominance (99/100 Train, 29/30 Val)
                                    # -> the RS "sort positives by IoU" term is inert; DO NOT use RS.
RESC_SMOOTH_AP_TAU   = 0.01         # sigmoid temperature of the rank indicator
RESC_FOCAL_GAMMA     = 2.0
RESC_CE_SEARCH       = ({"alpha": 0.25, "lr": 1e-3}, {"alpha": 0.50, "lr": 1e-3},
                        {"alpha": 0.25, "lr": 3e-4}, {"alpha": 0.50, "lr": 3e-4})   # 4 trials
RESC_LAMBDA_SEARCH   = (0.1, 0.3, 1.0, 3.0)   # 4 trials for the ranking variants (lr/alpha = B2's)
RESC_LAMBDA_DIAGNOSTIC = 0.0        # pure-ranking endpoint: REPORTED, excluded from selection
# --- 4 (G): seeds + selection ----------------------------------------------
RESC_SEEDS           = (0, 1, 2)    # rescorer seed r is PAIRED with detector full_seed{r}
RESC_SELECT_METRIC   = "val_cpm"    # official average_recall; NEVER val loss
RESC_SELECT_CPM_TOL  = 0.02         # within this of the max = a tie on 30 val lesions -> earliest epoch
                                    # DELIBERATELY NOT the detector's DET_SELECT_CPM_TOL (0.10). The
                                    # Inv. 2 / A1-tau amendment raised THAT constant to buy recall
                                    # CEILING, which is unrecoverable downstream (Inv. 8). Here there
                                    # is no ceiling to buy: by Inv. 8 every rung and every epoch
                                    # re-ranks one frozen pool, so max_recall is identical throughout
                                    # and the ceiling tie-break is a no-op. tau stays at the noise
                                    # width it was calibrated as (~1/30 of the val lesions).
RESC_PROB_EPS        = 1e-6         # clamp sigmoid to [0, 1-eps) for the det_score sweep
# --- 4 (H): the Phase-3 floor this phase must beat -- MEASURED, never hard-coded ------------
# There is deliberately NO `RESC_B0_CPM` / `RESC_RECALL_CEILING` constant here.
#
# There were, until 2026-08-09, and they were WRONG: they held 0.5567 / 0.8667, the ARCHIVED
# (d1-flip) arm's values, while the promoted arm's B0' is 0.6327 +- 0.0526 at ceiling 0.8556
# ([F.8]). `phase4_eval_grid.py` gated exit check 4 on that constant, so a B1 landing at 0.59
# would have PASSED a gate while sitting 0.04 BELOW the baseline it exists to beat -- the gate
# was not merely stale, it was inverted in effect. RESULTS_FOLD_FLIP's stale-number sweep caught
# the same defect in RB_PHASE_4.md's prose and missed it here.
#
# The fix is structural, not a new number: B0 is the frozen pool ranked by `score_max`, so every
# script that needs it MEASURES it on the pool it is actually holding (one oracle call), exactly
# as the ceiling is already read per seed from `max_recall`. A constant that names a substrate is
# the same class of bug as a detection-cache key that names a run (HISTORY.md §8) -- when the
# substrate is promoted, the name survives and the meaning does not.


# ============================================================================
# ============================================================================
# "measured" AXIS-PROFILE OVERRIDES — the isotropic-cache branch
# ----------------------------------------------------------------------------
# EVERYTHING above this line is the DEPLOYED ("legacy") arm, unchanged. Nothing
# below executes unless ABUS_AXIS_PROFILE=measured. Runbook: iso/RB_ISO_REBUILD.md.
#
# Two classes of value live here, and they must not be confused:
#
#   (1) MEASURED FACTS — settled before the branch starts, not re-derivable by a
#       gate:  SPACING_STORAGE_MM, FP_PROBE_ANISO_DEPTH_AXIS.
#   (2) PROVISIONAL PLACEHOLDERS — every data-dependent constant, carrying the
#       arithmetic prediction of what the corresponding gate should print. They
#       follow the SAME contract Phase 2 (B) has always used: written provisional,
#       reconciled against the probe BEFORE any training run, and the change
#       recorded in iso/RESULTS_ISO_REBUILD.md. A gate that reads GATE PASS on a
#       placeholder is a coincidence to be recorded, never an assumption.
#
# The two profiles are NOT comparable constant-for-constant. One iso voxel means
# 1.0959 x 0.1460 x 0.4000 mm under "legacy" and 0.4 mm cubed under "measured", so
# any constant denominated in iso voxels (crop sides, drift caps, merge gap,
# cluster radius) changes meaning even where the number is unchanged.
# ============================================================================
if AXIS_PROFILE == "measured":

    # --- (1) MEASURED FACTS -------------------------------------------------
    # results/AXIS_CHECK.md + RESULTS_INTENSITY_PROBE.md [I.3]: attenuation
    # (Spearman -0.938, peak at 2% of the axis, 129/130 volumes), the physical-
    # extent envelope (exactly 1 of 6 permutations survives), sample-count
    # structure, direct visual, and the GE Invenia hardware numbers
    # (768 elements x 0.20 mm ~ the quoted 15.3 cm aperture).
    #   d0 = LATERAL 0.200 | d1 = DEPTH/BEAM 0.073 | d2 = SWEEP 0.475674
    # Feeds preprocess_hash => a DIFFERENT cache dir from the deployed ab1fdf28...
    SPACING_STORAGE_MM = (0.200, 0.073, 0.475674)
    # => zoom factors at 0.4 mm: (0.500000, 0.182500, 1.189185)
    #    a 865 x 682 x 353 volume caches as 432 x 124 x 420 (vs 158 x 341 x 420),
    #    i.e. the SAME voxel budget with 2.7x finer LATERAL resolution.

    # Now the true beam axis, so `anisotropy` finally measures what its name says.
    # On "legacy" this stays 0 purely so every recorded probe number reproduces.
    FP_PROBE_ANISO_DEPTH_AXIS = DEPTH_AXIS   # = 1

    # --- (2) PROVISIONAL — Phase 2 (B), reconciled at the [I2.1] HARD GATE ---
    # Frame maxima invert: d0 865 -> 432 (was the SHORT side, becomes the long one)
    # and d1 {546,608,682} -> {100,111,124}. See the derive_constants axis-role fix
    # (detect/det_stats.py): min_size is the SHORT side, max_size the long one --
    # backward-compatible on "legacy", where it reproduces 160 / 352 exactly.
    DET_MIN_SIZE  = 128    # predicted round_up(max Train iso SHORT side = 124, 32)
    DET_MAX_SIZE  = 448    # predicted round_up(max Train iso LONG  side = 432, 32)
    DET_IMAGE_MEAN = 0.23      # carried over; the resampling weights change slightly
    DET_IMAGE_STD  = 0.1658    # -> both are re-measured by the probe, not assumed
    # RECONCILED at the [I2.1] HARD GATE (RESULTS_ISO_REBUILD.md [I2.1]); the
    # (16,32,64,128,256) above was a placeholder copied from "legacy". DET_RULE
    # derives the ladder from the Train diag percentiles measured on THIS cache:
    #   b0 = 2**round(log2(diag_p1=5.8)) = 8; anchor_min_base enters as min(8,16) = 8
    #   top check: diag_p99 145.6 / 2**(2/3) = 91.7 <= 128 -> no octave shift.
    # The whole ladder drops one octave because at a uniform 0.4 mm grid the same
    # lesions are smaller IN PIXELS than on a cache whose d0 pitch was 1.096 mm.
    DET_ANCHOR_BASE_SIZES    = (8, 16, 32, 64, 128)
    # THE constant that most visibly encoded the distortion. Deployed is
    # (0.2, 0.25, 0.33, 1.0) -- a 7.5x wide skew that is the in-plane aspect error,
    # not lesion morphology. Undistorted, the SAME boxes give h/w p10/p50/p90
    # = 1.176 / 1.800 / 2.902 measured at [I2.1] (deployed 0.162/0.246/0.405 x ~7.3;
    # the arithmetic 7.5068 is the factor for ONE rescaled box, and these are
    # percentiles of a re-derived 3802-box set), agreeing with the independently
    # measured GT box shape 1.89/0.63/0.76 and with "real breast masses are wider-
    # than-tall" (HISTORY.md 5). RECONCILED at [I2.1]: the provisional (1.0,1.5,2.0)
    # was a hand-snap that skipped DET_RULE's nearest-grid rule. Applying it --
    #   1.176 -> 1.0 | 1.800 -> 2.0 | 2.902 -> 2.0 (grid max), U {1.0}
    # -- gives (1.0, 2.0); 1.5 never wins a snap for any of the three percentiles.
    DET_ANCHOR_ASPECT_RATIOS = (1.0, 2.0)

    # --- (2) PROVISIONAL — the Inv.-11 union-box merge distance --------------
    # 8 iso px means 8.8 mm laterally / 1.17 mm in depth on the deployed cache and
    # 3.2 mm isotropically here, so the label contract itself changes. Re-validated
    # at [I1.5] on the criterion the original carried: fragments of one lesion merge,
    # case 93's two genuine foci stay separate.
    DET_LABEL_MERGE_GAP = 8

    # --- (2) PROVISIONAL — Phase 3 (A) linker, re-frozen at [I3.2] (Inv. 4) --
    # d2 = sweep is unchanged by the correction, so every z-denominated constant
    # (LINK_MIN_TUBE_LEN, LINK_MAX_TUBE_ZSPAN, LINK_MAX_Z_GAP) is expected to
    # re-derive UNCHANGED; the in-plane one is not.
    LINK_IOU              = 0.30
    LINK_MAX_Z_GAP        = 1
    LINK_MIN_TUBE_LEN     = 8
    LINK_MAX_TUBE_ZSPAN   = 182     # 1.8 x Train-GT z-extent p99 (101.11) -- z is unchanged
    # Deployed 342 = 1.5 x an in-plane extent p99 of 228 iso px, which [P3U2.PD] already
    # noticed is ">= the volume's full lateral width 341, i.e. laterally a no-op" -- a
    # direct consequence of the p99 being dominated by the 7.5x-stretched depth axis
    # (228 px x 0.146 mm ~ 33 mm of DEPTH). Undistorted the p99 is set by the lateral
    # extent instead; prediction ~112 px -> cap ~168, i.e. the cap BINDS for the first
    # time. Re-derived at [I3.2]; if it comes out a no-op again, say so.
    # [I2.6] DERIVED 2026-08-13: inplane_extent_p99 = 119.99 px -> round(1.5 * 119.99) = 180.
    # Prediction ~168 was close (the p99 came in at 120 rather than ~112). The substantive
    # half of P5 holds: 342 -> 180 is a sharp drop and the cap now BINDS laterally
    # (180 < the 432-px d0 frame) where the deployed 342 >= the 341-px frame was a no-op.
    # It is still a no-op on d1 (180 > the 124-px depth frame) -- which is correct here,
    # because a lesion cannot drift further than the axis is wide.
    LINK_MAX_CENTROID_DRIFT = 180
    LINK_CONTAINMENT_THRESH = 1.0   # off, as deployed
    LINK_3DNMS_IOU          = None  # off, as deployed (Phase-4 ablation only)
    LINK_NMS_THRESH         = 0.70

    # --- (2) PROVISIONAL — Phase 3 (B) operating point, re-derived at [I3.3] -
    # These are properties of a DETECTOR's score distribution, and the detectors on
    # this branch do not exist yet. Carried over only so the pipeline runs; both are
    # re-derived, and the A2-repair rule (Inv. 2, adopted 2026-08-08) is what picks
    # the op: among thresholds whose POST-FLOOR pool <= RESCORER_POOL_BUDGET,
    # maximise linked val CPM; tie-break to higher ceiling, then smaller pool.
    PREFILTER_SCORE_FLOOR = 0.08
    LINK_OP_SCORE_THRESH  = 0.03
    DET_SELECT_OP_THRESH  = 0.03

    # --- DELIBERATELY NOT OVERRIDDEN (record the reason, do not "fix" these) --
    # ISO_SPACING_MM (0.4)       : held fixed so the ONLY changed variable between the
    #                              two substrates is the spacing MAP. Same voxel budget,
    #                              same training cost. Re-justified, not inherited, by
    #                              the [I1.4] fidelity read.
    # DET_ASSIGNER ("atss")      : a recipe choice frozen by the [P3U.3] D0 gate, not a
    #                              substrate constant. The gate is RE-RUN at [I2.2] for
    #                              the record -- on undistorted boxes for the first time.
    # DET_SELECT_CPM_TOL (0.10)  : Inv. 2 / A1-tau. A selection POLICY, identical across
    #                              architectures and substrates by Inv. 2's own wording.
    # TRAIN_AUGMENT flip axis (0): already the measured lateral axis. Correct in BOTH
    #                              profiles, for different reasons (augment.py).
    # RESCORER_POOL_BUDGET (200) : a Phase-4 O(n^2) budget, not a physical quantity.
    # RESC_CROP_* (48/1.5/48/160): voxel counts. UNCHANGED numbers, but they stop being
    #                              a 7.5:1 slab and become a physical cube -- 48 vox is
    #                              19.2 mm on every axis here, against 52.6 x 7.0 x 19.2
    #                              mm on the deployed cache. This is the single largest
    #                              reason the rebuild reaches the CONTRIBUTION (Inv. 5)
    #                              and not just the substrate under it.
    # FP_PROBE_CLUSTER_RADIUS(10): 10 iso vox is an 11.0 x 1.5 x 4.0 mm ELLIPSOID on the
    #                              deployed cache and a 4 mm SPHERE here. The FP-cluster
    #                              count (14.0/vol vs 1.0 for TP) is the primary evidence
    #                              for the STRONG jointness prior and WILL move. That is
    #                              the measurement, not a regression -- the two profiles'
    #                              cluster counts are not comparable by construction.
