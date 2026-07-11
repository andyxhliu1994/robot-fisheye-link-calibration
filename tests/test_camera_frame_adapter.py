import numpy as np
import pytest

from calibration_pipeline.camera_frame_adapter import camera_frame_adapter_from_config


def test_identity_adapter():
    adapter = camera_frame_adapter_from_config({"type": "identity"})
    assert np.allclose(adapter([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])


def test_flip_y_matrix_adapter():
    adapter = camera_frame_adapter_from_config(
        {
            "type": "matrix_3x3",
            "name": "flip_y_to_unity_camera_frame",
            "matrix": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
        }
    )
    assert np.allclose(adapter([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0])


def test_nonorthogonal_adapter_is_rejected():
    with pytest.raises(ValueError, match="orthogonal"):
        camera_frame_adapter_from_config(
            {"type": "matrix_3x3", "matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}
        )

