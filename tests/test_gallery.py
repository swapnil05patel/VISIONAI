import time

import numpy as np

from tracker.gallery import TrackGallery
from tracker.track import Track
from tracker.utils import l2_normalize_vec


def _unit(dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return l2_normalize_vec(rng.normal(size=dim).astype(np.float32))


def _track(tid, box, emb, hits=5):
    tr = Track(box, [emb], tid, frame_idx=1, dt=1 / 30.0, frame_size=(640, 480))
    tr.hits = hits
    return tr


def test_gallery_save_and_query_mean():
    gallery = TrackGallery(representation="mean", use_spatial_gate=True, spatial_max_dist=200)
    emb = _unit(seed=1)
    box = [100, 100, 160, 280]
    assert gallery.save(_track(7, box, emb), frame_idx=10)
    assert len(gallery) == 1
    hit = gallery.query(box, emb, time.time())
    assert hit == 7


def test_gallery_spatial_gate_can_be_disabled():
    gallery = TrackGallery(use_spatial_gate=False, spatial_max_dist=5.0, representation="mean")
    emb = _unit(seed=2)
    gallery.save(_track(3, [10, 10, 50, 90], emb), frame_idx=1)
    far_box = [400, 300, 450, 400]
    assert gallery.query(far_box, emb, time.time()) == 3

    gated = TrackGallery(use_spatial_gate=True, spatial_max_dist=5.0, representation="mean")
    gated.save(_track(3, [10, 10, 50, 90], emb), frame_idx=1)
    assert gated.query(far_box, emb, time.time()) is None


def test_gallery_exemplars_and_lock():
    gallery = TrackGallery(representation="exemplars", use_spatial_gate=False)
    e1 = _unit(seed=4)
    e2 = _unit(seed=5)
    tr = _track(9, [20, 20, 80, 200], e1)
    tr.embeds.append(e2)
    assert gallery.save(tr, 2)
    gallery.lock(9)
    assert gallery.query([20, 20, 80, 200], e2, time.time()) is None
    gallery.unlock(9)
    assert gallery.query([20, 20, 80, 200], e2, time.time()) == 9


def test_short_track_not_saved():
    gallery = TrackGallery()
    emb = _unit(seed=8)
    tr = _track(1, [0, 0, 40, 80], emb, hits=1)
    assert gallery.save(tr, 1) is False
    assert len(gallery) == 0
