import numpy as np

from calibration_pipeline.se3_utils import (
    compose_T,
    invert_T,
    mat_from_t_q,
    rotation_error_deg,
    t_q_from_mat,
    translation_error_m,
)


def test_transform_round_trip_and_inverse():
    T = mat_from_t_q([1.0, -2.0, 0.5], [0.1, -0.2, 0.3, 0.9])
    t, q = t_q_from_mat(T)
    reconstructed = mat_from_t_q(t, q)
    assert np.allclose(reconstructed, T)
    assert np.allclose(compose_T(T, invert_T(T)), np.eye(4), atol=1e-12)


def test_error_metrics():
    identity = np.eye(3)
    quarter_turn = mat_from_t_q([0, 0, 0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])[:3, :3]
    assert rotation_error_deg(identity, quarter_turn) == pytest.approx(90.0)
    assert translation_error_m(np.zeros(3), np.array([3.0, 4.0, 0.0])) == 5.0


import pytest

