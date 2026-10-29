import numpy as np

from tests.dummy_embedder import ColorHashEmbedder
from tracker.reid_tracker import PersonReIDTracker


def _frame_with_boxes(boxes_and_colors, size=(240, 320)):
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for box, color in boxes_and_colors:
        x1, y1, x2, y2 = [int(v) for v in box]
        frame[y1:y2, x1:x2] = color
    return frame


def test_tracker_keeps_id_across_short_gap():
    embedder = ColorHashEmbedder()
    tracker = PersonReIDTracker(
        device="cpu",
        frame_size=(320, 240),
        fps=30.0,
        embedder=embedder,
        enable_gallery=True,
        enable_bytetrack_stage2=True,
        enable_static_pruning=False,
        revival_confirmation_frames=1,
        min_hits=1,
        max_age=20,
    )
    box = [40, 40, 90, 160]
    color = (20, 40, 200)
    frame = _frame_with_boxes([(box, color)])
    parts = embedder.extract(frame, box)
    tracks = tracker.step([box], [parts], 1)
    tid = next(iter(tracks))
    tracker.step([], [], 2)
    moved = [48, 40, 98, 160]
    frame2 = _frame_with_boxes([(moved, color)])
    parts2 = embedder.extract(frame2, moved)
    tracks = tracker.step([moved], [parts2], 3)
    assert tid in tracks
    assert tracks[tid].time_since_update == 0


def test_bytetrack_stage2_associates_low_conf_box():
    embedder = ColorHashEmbedder()
    tracker = PersonReIDTracker(
        device="cpu",
        frame_size=(320, 240),
        fps=30.0,
        embedder=embedder,
        enable_gallery=False,
        enable_bytetrack_stage2=True,
        enable_static_pruning=False,
        min_hits=1,
        iou_thr=0.3,
        stage2_iou_thr=0.3,
    )
    box = [30, 30, 80, 150]
    frame = _frame_with_boxes([(box, (10, 200, 30))])
    parts = embedder.extract(frame, box)
    tracker.step([box], [parts], 1)
    low = [34, 34, 82, 148]
    tracks = tracker.step([], [], 2, low_conf_detections=[low])
    assert len(tracks) == 1
    assert next(iter(tracks.values())).time_since_update == 0


def test_gallery_revival_requires_consecutive_matches():
    embedder = ColorHashEmbedder()
    tracker = PersonReIDTracker(
        device="cpu",
        frame_size=(320, 240),
        fps=30.0,
        embedder=embedder,
        enable_gallery=True,
        enable_static_pruning=False,
        revival_confirmation_frames=2,
        min_hits=1,
        max_age=2,
        iou_thr=0.2,
    )
    box = [20, 20, 70, 140]
    color = (180, 20, 20)
    frame = _frame_with_boxes([(box, color)])
    parts = embedder.extract(frame, box)
    tracker.step([box], [parts], 1)
    tracker.step([box], [parts], 2)
    orig_id = next(iter(tracker.tracks))
    tracker.step([], [], 3)
    tracker.step([], [], 4)
    tracker.step([], [], 5)
    assert orig_id not in tracker.tracks
    assert len(tracker.gallery) >= 1

    tracks = tracker.step([box], [parts], 6)
    assert orig_id in tracks
    assert tracks[orig_id].confirmed is False
    tracks = tracker.step([box], [parts], 7)
    assert orig_id in tracks
    assert tracks[orig_id].confirmed is True
