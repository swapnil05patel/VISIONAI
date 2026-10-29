import numpy as np
import pytest

from tracker.utils import cosine_distance_vec, l2_normalize_vec


def test_cosine_identical_is_zero():
    v = l2_normalize_vec(np.ones(8, dtype=np.float32))
    assert cosine_distance_vec(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_mismatched_dims_returns_one():
    a = l2_normalize_vec(np.ones(8, dtype=np.float32))
    b = l2_normalize_vec(np.ones(4, dtype=np.float32))
    assert cosine_distance_vec(a, b) == 1.0
    assert cosine_distance_vec(None, a) == 1.0
    bad = np.array([np.nan, 0.0], dtype=np.float32)
    assert cosine_distance_vec(bad, bad) == 1.0
