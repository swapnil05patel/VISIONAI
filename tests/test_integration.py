"""
tests/test_integration.py
--------------------------
Integration and regression tests:
- evaluate.py synthetic sequence produces real (non-fabricated) metrics
- MOT IO round-trip: write then load preserves all fields
- Gallery TTL eviction removes expired entries
- Gallery EMA representation queries correctly
- Static track pruning removes non-moving tracks
- ID preserved across a longer gap (gallery recovery)
- Tracker reset clears all state
- Embedding dimension mismatch is handled safely
"""

import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluation.mot_io import xyxy_to_mot_row, write_mot_file, load_mot_file, index_by_frame
from evaluation.mot_metrics import compute_all_metrics
from tracker.gallery import TrackGallery
from tracker.track import Track
from tracker.utils import l2_normalize_vec
from tests.dummy_embedder import ColorHashEmbedder
from tracker.reid_tracker import PersonReIDTracker


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _unit(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize_vec(rng.normal(size=dim).astype(np.float32))


def _make_track(tid: int, box, emb: np.ndarray, hits: int = 5) -> Track:
    tr = Track(box, [emb], tid, frame_idx=1, dt=1 / 30.0, frame_size=(640, 480))
    tr.hits = hits
    return tr


def _colored_frame(boxes_colors, size=(240, 320)):
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for box, color in boxes_colors:
        x1, y1, x2, y2 = [int(v) for v in box]
        frame[y1:y2, x1:x2] = color
    return frame


# ------------------------------------------------------------------ #
# MOT IO round-trip                                                    #
# ------------------------------------------------------------------ #

def test_mot_io_round_trip():
    rows = [
        xyxy_to_mot_row(1, 1, [10.0, 20.0, 60.0, 100.0]),
        xyxy_to_mot_row(2, 1, [12.0, 22.0, 62.0, 102.0]),
        xyxy_to_mot_row(1, 2, [200.0, 50.0, 260.0, 150.0]),
    ]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        write_mot_file(tmp_path, rows)
        loaded = load_mot_file(tmp_path)
        assert len(loaded) == 3
        for orig, back in zip(rows, loaded):
            assert orig["frame"] == back["frame"]
            assert orig["id"] == back["id"]
            np.testing.assert_allclose(orig["xywh"], back["xywh"], atol=0.1)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_mot_io_index_by_frame():
    rows = [
        xyxy_to_mot_row(1, 1, [10.0, 10.0, 50.0, 80.0]),
        xyxy_to_mot_row(1, 2, [100.0, 10.0, 140.0, 80.0]),
        xyxy_to_mot_row(2, 1, [12.0, 10.0, 52.0, 80.0]),
    ]
    by_frame = index_by_frame(rows)
    assert len(by_frame[1]) == 2
    assert len(by_frame[2]) == 1


def test_mot_io_skips_zero_conf_rows():
    rows = [
        xyxy_to_mot_row(1, 1, [10.0, 10.0, 50.0, 80.0], conf=0.0),
        xyxy_to_mot_row(2, 1, [12.0, 10.0, 52.0, 80.0], conf=1.0),
    ]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        write_mot_file(tmp_path, rows)
        loaded = load_mot_file(tmp_path, ignore_zero_conf=True)
        assert len(loaded) == 1
        assert loaded[0]["frame"] == 2
    finally:
        tmp_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# Gallery TTL eviction                                                 #
# ------------------------------------------------------------------ #

def test_gallery_ttl_eviction():
    gallery = TrackGallery(representation="mean", use_spatial_gate=False)
    emb = _unit(seed=10)
    box = [50.0, 50.0, 100.0, 200.0]
    tr = _make_track(42, box, emb)
    gallery.save(tr, frame_idx=1)
    assert len(gallery) == 1

    # Entry should be found with current time
    hit = gallery.query(box, emb, time.time())
    assert hit == 42

    # Simulate expired entry by querying with a far-future time
    future_time = time.time() + 700  # > GALLERY_TTL_SECONDS (600)
    hit_expired = gallery.query(box, emb, future_time)
    assert hit_expired is None

    # evict_expired should remove it
    gallery.evict_expired(future_time)
    assert len(gallery) == 0


# ------------------------------------------------------------------ #
# Gallery EMA representation                                           #
# ------------------------------------------------------------------ #

def test_gallery_ema_representation():
    gallery = TrackGallery(representation="ema", use_spatial_gate=False)
    emb = _unit(seed=20)
    box = [30.0, 30.0, 80.0, 160.0]
    tr = _make_track(5, box, emb)
    saved = gallery.save(tr, frame_idx=1)
    assert saved
    hit = gallery.query(box, emb, time.time())
    assert hit == 5


# ------------------------------------------------------------------ #
# Gallery area ratio gate                                              #
# ------------------------------------------------------------------ #

def test_gallery_area_ratio_gate():
    gallery = TrackGallery(
        representation="mean",
        use_spatial_gate=False,
        area_ratio_max=2.0,
    )
    emb = _unit(seed=30)
    small_box = [10.0, 10.0, 30.0, 50.0]   # area = 20*40 = 800
    large_box = [10.0, 10.0, 110.0, 210.0]  # area = 100*200 = 20000 (ratio > 2)
    tr = _make_track(11, small_box, emb)
    gallery.save(tr, frame_idx=1)
    # Same box: should match
    assert gallery.query(small_box, emb, time.time()) == 11
    # Very different area: should be blocked
    assert gallery.query(large_box, emb, time.time()) is None


# ------------------------------------------------------------------ #
# Tracker reset                                                        #
# ------------------------------------------------------------------ #

def test_tracker_reset_clears_state():
    embedder = ColorHashEmbedder()
    tracker = PersonReIDTracker(
        device="cpu", frame_size=(320, 240), fps=30.0,
        embedder=embedder, enable_gallery=True,
        enable_static_pruning=False, min_hits=1,
    )
    box = [40.0, 40.0, 90.0, 160.0]
    frame = _colored_frame([(box, (20, 40, 200))])
    parts = embedder.extract(frame, box)
    tracker.step([box], [parts], 1)
    assert len(tracker.tracks) > 0

    tracker.reset()
    assert len(tracker.tracks) == 0
    assert len(tracker.gallery) == 0
    assert tracker.next_id == 1
    assert tracker._frames_processed == 0


# ------------------------------------------------------------------ #
# ID preserved across longer gap (gallery recovery)                   #
# ------------------------------------------------------------------ #

def test_id_preserved_across_long_gap_with_gallery():
    embedder = ColorHashEmbedder()
    tracker = PersonReIDTracker(
        device="cpu", frame_size=(320, 240), fps=30.0,
        embedder=embedder, enable_gallery=True,
        enable_static_pruning=False, revival_confirmation_frames=1,
        min_hits=1, max_age=3, iou_thr=0.2,
    )
    box = [40.0, 40.0, 90.0, 160.0]
    color = (20, 40, 200)
    frame = _colored_frame([(box, color)])
    parts = embedder.extract(frame, box)

    # Establish track
    tracker.step([box], [parts], 1)
    tracker.step([box], [parts], 2)
    orig_id = next(iter(tracker.tracks))

    # Gap: track retires after max_age=3 missed frames
    tracker.step([], [], 3)
    tracker.step([], [], 4)
    tracker.step([], [], 5)
    tracker.step([], [], 6)
    assert orig_id not in tracker.tracks
    assert len(tracker.gallery) >= 1

    # Reappear: gallery should revive original ID
    tracks = tracker.step([box], [parts], 7)
    assert orig_id in tracks, f"Expected ID {orig_id} to be revived, got {list(tracks.keys())}"


# ------------------------------------------------------------------ #
# Static track pruning                                                 #
# ------------------------------------------------------------------ #

def test_static_track_is_pruned():
    from tracker.config import STATIC_FRAMES_REQUIRED
    embedder = ColorHashEmbedder()
    tracker = PersonReIDTracker(
        device="cpu", frame_size=(320, 240), fps=30.0,
        embedder=embedder, enable_gallery=False,
        enable_static_pruning=True, min_hits=1, max_age=500,
    )
    # Stationary box — never moves
    box = [100.0, 80.0, 150.0, 200.0]
    frame = _colored_frame([(box, (200, 200, 200))])
    parts = embedder.extract(frame, box)

    tid = None
    for i in range(1, STATIC_FRAMES_REQUIRED + 10):
        tracks = tracker.step([box], [parts], i)
        if tid is None and tracks:
            tid = next(iter(tracks))

    # After enough static frames, the track should be pruned
    assert tid not in tracker.tracks, (
        f"Static track {tid} should have been pruned after {STATIC_FRAMES_REQUIRED} frames"
    )


# ------------------------------------------------------------------ #
# Evaluate.py synthetic sequence produces real metrics                 #
# ------------------------------------------------------------------ #

def test_evaluate_synthetic_produces_real_metrics():
    """Run the synthetic evaluator and verify metrics are non-fabricated numbers."""
    from evaluation.evaluate import _make_synthetic_sequence, run_tracker_on_frames
    from evaluation.mot_io import index_by_frame

    frames, gt_rows = _make_synthetic_sequence(num_frames=30)
    assert len(frames) == 30
    assert len(gt_rows) > 0

    gt_by_frame = index_by_frame(gt_rows)
    pred_rows, fps, info = run_tracker_on_frames(
        frames,
        gt_by_frame,
        backbone="osnet_x0_25",
        use_part_embedding=False,
        enable_gallery=True,
        device="cpu",
    )

    assert info["embedder_available"] is True
    assert info["feature_dim"] == 512
    assert fps > 0.0
    assert len(pred_rows) > 0

    metrics = compute_all_metrics(gt_rows, pred_rows)
    # Metrics must be real numbers in valid ranges
    assert 0.0 <= metrics["MOTA"] <= 1.0, f"MOTA out of range: {metrics['MOTA']}"
    assert 0.0 <= metrics["IDF1"] <= 1.0, f"IDF1 out of range: {metrics['IDF1']}"
    assert 0.0 <= metrics["HOTA"] <= 1.0, f"HOTA out of range: {metrics['HOTA']}"
    assert metrics["IDSW"] >= 0
    assert metrics["TP"] >= 0
    assert metrics["FP"] >= 0
    assert metrics["FN"] >= 0
    print(
        f"Synthetic eval: MOTA={metrics['MOTA']:.4f} IDF1={metrics['IDF1']:.4f} "
        f"HOTA={metrics['HOTA']:.4f} IDSW={metrics['IDSW']} FPS={fps:.1f}"
    )


# ------------------------------------------------------------------ #
# Embedding dimension mismatch handled safely                          #
# ------------------------------------------------------------------ #

def test_embedding_dimension_mismatch_returns_empty():
    """compute_weighted_mean must handle mismatched dimensions without crashing."""
    from tracker.embedder import PartEmbedder
    embedder = PartEmbedder(device="cpu", backbone="osnet_x0_25")
    v512 = _unit(512, seed=1)
    v256 = _unit(256, seed=2)
    # Mixed dimensions: only the consistent ones should be used
    result = embedder.compute_weighted_mean([v512, v256])
    # Should return None or a valid 512-D vector (not crash)
    if result is not None:
        assert result.shape[0] == 512
        assert np.isfinite(result).all()
        assert abs(np.linalg.norm(result) - 1.0) < 1e-4


# ------------------------------------------------------------------ #
# Cosine distance edge cases                                           #
# ------------------------------------------------------------------ #

def test_cosine_distance_edge_cases():
    from tracker.utils import cosine_distance_vec, l2_normalize_vec
    v = l2_normalize_vec(np.ones(512, dtype=np.float32))
    # Identical vectors: distance = 0
    assert cosine_distance_vec(v, v) == pytest.approx(0.0, abs=1e-5)
    # None inputs: distance = 1
    assert cosine_distance_vec(None, v) == 1.0
    assert cosine_distance_vec(v, None) == 1.0
    # NaN input: distance = 1
    nan_v = np.full(512, np.nan, dtype=np.float32)
    assert cosine_distance_vec(nan_v, v) == 1.0
    # Mismatched dims: distance = 1
    short = l2_normalize_vec(np.ones(256, dtype=np.float32))
    assert cosine_distance_vec(v, short) == 1.0
