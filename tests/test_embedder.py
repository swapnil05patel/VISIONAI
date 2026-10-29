"""
tests/test_embedder.py
----------------------
Unit and regression tests for PartEmbedder:
- OSNet backbone vs ResNet50 baseline
- Dimension validation (512D for OSNet, 2048D for ResNet50)
- L2-normalization check (norm == 1.0)
- Finite values check (no NaN / Inf)
- Full-body mode vs 3-part mode
- Safe handling of invalid, zero-sized, and boundary boxes (zero random fallbacks)
"""

import numpy as np
import pytest
from tracker.embedder import PartEmbedder


def test_osnet_full_body_extraction():
    embedder = PartEmbedder(device="cpu", backbone="osnet_x0_25", use_part_embedding=False)
    assert embedder.available is True
    assert embedder.feature_dim == 512

    # Dummy image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    box = [50, 50, 200, 350]

    feats = embedder.extract(frame, box)
    assert len(feats) == 1
    feat = feats[0]
    assert feat.shape == (512,)
    assert np.isfinite(feat).all()
    # Check L2 normalization
    norm = np.linalg.norm(feat)
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_osnet_part_extraction():
    embedder = PartEmbedder(device="cpu", num_parts=3, backbone="osnet_x0_25", use_part_embedding=True)
    assert embedder.available is True

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    box = [100, 80, 220, 400]

    feats = embedder.extract(frame, box)
    assert len(feats) == 3
    for f in feats:
        assert f.shape == (512,)
        assert np.isfinite(f).all()
        assert pytest.approx(np.linalg.norm(f), abs=1e-4) == 1.0


def test_resnet50_baseline():
    embedder = PartEmbedder(device="cpu", backbone="resnet50", use_part_embedding=False)
    assert embedder.available is True
    assert embedder.feature_dim == 2048

    frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    box = [20, 20, 150, 250]

    feats = embedder.extract(frame, box)
    assert len(feats) == 1
    assert feats[0].shape == (2048,)
    assert pytest.approx(np.linalg.norm(feats[0]), abs=1e-4) == 1.0


def test_invalid_and_degenerate_boxes_return_empty():
    embedder = PartEmbedder(device="cpu", backbone="osnet_x0_25")
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Empty frame
    assert embedder.extract(np.empty((0, 0, 3), dtype=np.uint8), [10, 10, 50, 50]) == []

    # Box with zero width
    assert embedder.extract(frame, [10, 10, 10, 50]) == []

    # Tiny box (<8px)
    assert embedder.extract(frame, [10, 10, 14, 15]) == []

    # Out of bounds box
    assert embedder.extract(frame, [200, 200, 300, 300]) == []


def test_unknown_backbone_is_unavailable():
    embedder = PartEmbedder(device="cpu", backbone="not_a_real_backbone")
    assert embedder.available is False
    frame = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    assert embedder.extract(frame, [10, 10, 50, 70]) == []


def test_compute_weighted_mean():
    embedder = PartEmbedder(device="cpu", backbone="osnet_x0_25")
    assert embedder.compute_weighted_mean([]) is None

    v1 = np.ones(512, dtype=np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.zeros(512, dtype=np.float32)
    v2[0] = 1.0

    mean_v = embedder.compute_weighted_mean([v1, v2], weights=[0.8, 0.2])
    assert mean_v is not None
    assert mean_v.shape == (512,)
    assert pytest.approx(np.linalg.norm(mean_v), abs=1e-4) == 1.0
