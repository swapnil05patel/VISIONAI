# Changelog

All notable changes to this project are documented here.
Dates reflect the implementation order, not calendar dates.

---

## [Unreleased] — Phase 1–4 Implementation

### Phase 1 — ReID Backbone

**Changed files:** `tracker/embedder.py`, `tracker/osnet.py`, `tracker/config.py`

- Replaced ImageNet-only ResNet50 as the default backbone with **OSNet-x0.25 trained on
  MSMT17** (`512-D` L2-normalized embeddings). ResNet50 (`2048-D`) is retained as a
  configurable baseline via `REID_BACKBONE=resnet50`.
- Added `USE_PART_EMBEDDING` flag: `False` (default) = full-body crop; `True` = `NUM_PARTS`
  horizontal strips, each embedded independently.
- All embeddings are validated for finite values and correct dimension before being returned.
  Extraction returns an empty list `[]` on failure — never random fallback vectors.
- OSNet weights load from `weights/osnet_x0_25_msmt17.pth` (local cache) or download from
  HuggingFace on first use. Load/download failures raise `RuntimeError` immediately instead
  of silently leaving a randomly initialized network.
- Removed all legacy try/except fallback imports (`from config import ...`) across all
  tracker modules. All imports are now explicit package imports (`from tracker.config import ...`).
- `REID_STRICT_THRESH` (0.40) and `GALLERY_COSINE_THRESH` (0.35) were set based on OSNet
  MSMT17 embedding distribution knowledge. They were **not** blindly retuned — no labeled
  validation set was available.

### Phase 2 — Tracking and Gallery

**Changed files:** `tracker/reid_tracker.py`, `tracker/gallery.py`, `tracker/track.py`,
`tracker/config.py`

- **ByteTrack stage-2:** After primary Hungarian assignment, unmatched tracks are associated
  with low-confidence detections using IoU only (`ENABLE_BYTETRACK_STAGE2`, `STAGE2_IOU_THR`).
  Low-confidence boxes never spawn new track IDs.
- **Configurable association cost weights:** `MATCH_WEIGHT_IOU`, `MATCH_WEIGHT_REID`,
  `MATCH_WEIGHT_DIRECTION` are all configurable and passed through `PersonReIDTracker.__init__`.
- **Gallery representation:** configurable as `mean`, `ema`, or `exemplars`
  (`GALLERY_REPRESENTATION`). EMA uses `GALLERY_EMA_ALPHA`; exemplars use
  `GALLERY_MAX_EXEMPLARS` and `GALLERY_EXEMPLAR_MIN_DIST`.
- **Revival confirmation:** gallery-revived tracks require `REVIVAL_CONFIRMATION_FRAMES`
  consecutive matches before being marked confirmed. The gallery entry stays locked during
  the pending period to prevent duplicate revivals.
- **Spatial gallery gating:** configurable via `GALLERY_USE_SPATIAL_GATE` and
  `GALLERY_SPATIAL_MAX_DIST`. Disable for cross-camera scenarios (though cross-camera ReID
  is not implemented or validated).
- **Static track pruning:** configurable via `ENABLE_STATIC_PRUNING`. Uses hybrid detection:
  Kalman velocity < `STATIC_VELOCITY_THRESH` AND displacement < `STATIC_DISPLACEMENT_THRESH`
  for `STATIC_FRAMES_REQUIRED` consecutive frames.
- **Gallery eviction:** TTL-based eviction runs every `GALLERY_EVICT_INTERVAL` frames.
  Maximum gallery size enforced by evicting the oldest unlocked entry.
- All new parameters have defaults that preserve existing behavior.

### Phase 3 — Evaluation

**New files:** `evaluation/evaluate.py`, `evaluation/mot_io.py`, `evaluation/mot_metrics.py`,
`evaluation/__init__.py`, `evaluation/results.md`, `evaluation/preds/`

- `evaluation/evaluate.py`: runs the tracker on MOT-format sequences or a built-in synthetic
  two-box sequence. Accepts `--seq`, `--gt/--frames`, or `--synthetic`. Writes MOT-format
  predictions to `evaluation/preds/` and metrics to `evaluation/results.md`.
- `evaluation/mot_io.py`: MOT-format file loader and writer. Handles `frame,id,x,y,w,h,conf`
  format. `load_mot_file` skips zero-confidence rows and negative IDs.
- `evaluation/mot_metrics.py`: pure-Python implementations of CLEAR MOT (MOTA, IDSW, FRAG,
  TP, FP, FN), IDF1, and HOTA (averaged over IoU thresholds 0.05–0.95). Uses Hungarian
  matching per frame at the specified IoU threshold.
- Ablation flags: `--disable-gallery`, `--use-part-embedding`, `--backbone`, `--ablate`
  (runs all four configs on the same sequence).
- TrackEval is used if installed (`pip install trackeval`); otherwise falls back to the
  built-in implementation.
- `evaluation/results.md` is written only from actual runs. Metrics are never fabricated.

**Measured results (synthetic sequence, CPU):**

| Config | HOTA | IDF1 | MOTA | IDSW | FPS |
|--------|------|------|------|------|-----|
| OSNet-x0.25 + gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~21 |
| OSNet-x0.25 no gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~14 |
| OSNet-x0.25 + parts | 0.7521 | 0.8571 | 0.7167 | 0 | ~13 |
| ResNet50 + gallery | 0.7521 | 0.8571 | 0.7167 | 0 | ~12 |

All configs produce identical accuracy on the synthetic sequence (expected — the sequence
is too simple to differentiate backbone quality). FPS differences are real.

### Phase 4 — Documentation

**Changed files:** `README.md`, `CHANGELOG.md`

- README updated to reflect OSNet as the default backbone, all configuration parameters,
  evaluation commands, and project structure.
- Limitations section explicitly states: no camera-motion compensation, no cross-camera
  association, thresholds not recalibrated on labeled validation data.
- Cross-camera ReID is documented as **not implemented or validated**.

### Tests

**New files:** `tests/test_kalman.py`, `tests/test_integration.py`
**Existing files:** `tests/test_embedder.py`, `tests/test_gallery.py`, `tests/test_tracker.py`,
`tests/test_metrics.py`, `tests/test_utils.py` (all retained, all passing)

**New tests added (20 new, 37 total):**

`tests/test_kalman.py` (8 tests):
- `test_initiate_state_matches_measurement` — state vector initialized correctly from xywh
- `test_predict_advances_position_by_velocity` — position advances by vx*dt, vy*dt
- `test_predict_increases_covariance` — uncertainty grows on predict
- `test_update_corrects_toward_measurement` — state moves toward measurement after update
- `test_update_reduces_covariance` — uncertainty shrinks on update
- `test_stationary_box_stays_near_origin` — repeated update on same measurement stays stable
- `test_box_conversion_round_trip` — xyxy→xywh→xyxy is lossless
- `test_multiple_predict_steps_without_update` — position accumulates correctly over N steps

`tests/test_integration.py` (12 tests):
- `test_mot_io_round_trip` — write then load preserves all MOT fields
- `test_mot_io_index_by_frame` — index_by_frame groups correctly
- `test_mot_io_skips_zero_conf_rows` — zero-confidence rows are filtered
- `test_gallery_ttl_eviction` — expired entries are not returned by query
- `test_gallery_ema_representation` — EMA gallery queries correctly
- `test_gallery_area_ratio_gate` — large area difference blocks match
- `test_tracker_reset_clears_state` — reset() clears tracks, gallery, ID counter
- `test_id_preserved_across_long_gap_with_gallery` — gallery revives original ID after gap
- `test_static_track_is_pruned` — stationary track removed after STATIC_FRAMES_REQUIRED
- `test_evaluate_synthetic_produces_real_metrics` — evaluator produces valid metric ranges
- `test_embedding_dimension_mismatch_returns_empty` — mismatched dims handled safely
- `test_cosine_distance_edge_cases` — None, NaN, mismatched dims all return 1.0

**Test run result:** `37 passed in ~7s` (CPU, no GPU required)

---

## Known Remaining Issues

1. No real MOT-format dataset evaluation (MOT17/MOT20 not available in this environment).
   Metrics on real footage are expected to differ from synthetic results.
2. `REID_STRICT_THRESH` and `GALLERY_COSINE_THRESH` have not been tuned on a labeled
   validation set. They are reasonable defaults based on OSNet MSMT17 embedding properties.
3. No camera-motion compensation. Moving cameras will degrade IoU/spatial gating.
4. Cross-camera ReID is not implemented. The gallery uses image-coordinate spatial gating
   which assumes the same camera view.
5. The Streamlit app (`app.py`) has not been end-to-end tested in this pass — it requires
   a browser environment and an uploaded video file.
