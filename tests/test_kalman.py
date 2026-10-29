"""
tests/test_kalman.py
--------------------
Unit tests for KalmanBox:
- State initialization from measurement
- Predict advances position by velocity
- Update corrects state toward measurement
- Covariance grows on predict, shrinks on update
- Handles zero-velocity (stationary box)
"""

import numpy as np
import pytest
from tracker.kalman import KalmanBox
from tracker.utils import box_xyxy_to_xywh, box_xywh_to_xyxy


def test_initiate_state_matches_measurement():
    kf = KalmanBox(dt=1.0)
    xywh = np.array([100.0, 200.0, 50.0, 120.0], dtype=np.float32)
    x, P = kf.initiate(xywh)
    assert x.shape == (8,)
    assert P.shape == (8, 8)
    np.testing.assert_allclose(x[:4], xywh, atol=1e-5)
    # Velocity initialized to zero
    np.testing.assert_allclose(x[4:], 0.0, atol=1e-5)


def test_predict_advances_position_by_velocity():
    kf = KalmanBox(dt=1.0)
    xywh = np.array([100.0, 200.0, 50.0, 120.0], dtype=np.float32)
    x, P = kf.initiate(xywh)
    # Manually set velocity
    x[4] = 5.0   # vx
    x[5] = 3.0   # vy
    x_pred, P_pred = kf.predict(x, P)
    # cx should advance by vx*dt
    assert pytest.approx(x_pred[0], abs=1e-4) == 105.0
    assert pytest.approx(x_pred[1], abs=1e-4) == 203.0


def test_predict_increases_covariance():
    kf = KalmanBox(dt=1.0)
    x, P = kf.initiate(np.array([50.0, 50.0, 40.0, 80.0], dtype=np.float32))
    _, P_pred = kf.predict(x, P)
    # Trace of P should increase after prediction (uncertainty grows)
    assert np.trace(P_pred) > np.trace(P)


def test_update_corrects_toward_measurement():
    kf = KalmanBox(dt=1.0)
    xywh = np.array([100.0, 200.0, 50.0, 120.0], dtype=np.float32)
    x, P = kf.initiate(xywh)
    x_pred, P_pred = kf.predict(x, P)

    # Measurement is offset from prediction
    z = np.array([110.0, 205.0, 52.0, 122.0], dtype=np.float32)
    x_upd, P_upd = kf.update(x_pred, P_pred, z)

    # Updated position should be between prediction and measurement
    assert x_pred[0] <= x_upd[0] <= z[0] or z[0] <= x_upd[0] <= x_pred[0]


def test_update_reduces_covariance():
    kf = KalmanBox(dt=1.0)
    x, P = kf.initiate(np.array([50.0, 50.0, 40.0, 80.0], dtype=np.float32))
    x_pred, P_pred = kf.predict(x, P)
    z = np.array([52.0, 51.0, 41.0, 81.0], dtype=np.float32)
    _, P_upd = kf.update(x_pred, P_pred, z)
    # Trace of P should decrease after update (uncertainty shrinks)
    assert np.trace(P_upd) < np.trace(P_pred)


def test_stationary_box_stays_near_origin():
    kf = KalmanBox(dt=1.0)
    xywh = np.array([200.0, 150.0, 60.0, 100.0], dtype=np.float32)
    x, P = kf.initiate(xywh)
    # Repeatedly update with same measurement — should stay near measurement
    for _ in range(10):
        x, P = kf.predict(x, P)
        x, P = kf.update(x, P, xywh)
    np.testing.assert_allclose(x[:4], xywh, atol=5.0)


def test_box_conversion_round_trip():
    box_xyxy = [100.0, 50.0, 200.0, 250.0]
    xywh = box_xyxy_to_xywh(box_xyxy)
    back = box_xywh_to_xyxy(xywh)
    np.testing.assert_allclose(back, box_xyxy, atol=1e-4)


def test_multiple_predict_steps_without_update():
    kf = KalmanBox(dt=1.0)
    x, P = kf.initiate(np.array([100.0, 100.0, 50.0, 100.0], dtype=np.float32))
    x[4] = 2.0  # constant vx
    for _ in range(5):
        x, P = kf.predict(x, P)
    # After 5 steps with vx=2, cx should be near 110
    assert pytest.approx(x[0], abs=2.0) == 110.0
