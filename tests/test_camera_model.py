import json

import numpy as np

from calibration_pipeline.camera_frame_adapter import camera_frame_adapter_from_config
from calibration_pipeline.camera_model import OCamCalibCameraModel
from calibration_pipeline.dataset_loader import load_json_document


def test_ocamcalib_center_pixel_points_along_positive_z():
    model = OCamCalibCameraModel([100.0, 50.0], np.eye(2), [10.0, 0.0])
    ray = model.pixel_to_ray(100.0, 50.0, 200, 100)
    assert np.allclose(ray, [0.0, 0.0, 1.0])
    assert np.linalg.norm(ray) == pytest.approx(1.0)


def test_dataset_ocamcalib_model_and_adapter_are_usable():
    config, _ = load_json_document("dataset/camera_model_config.json")
    calibration = json.loads(
        open("dataset/" + config["default_calibration_file"], encoding="utf-8").read()
    )
    model = OCamCalibCameraModel.from_mapping(calibration)
    adapter = camera_frame_adapter_from_config(config["ray_frame_adapter"])
    raw = model.pixel_to_ray(1024.0, 1024.0, 2048, 2048)
    adapted = adapter(raw)
    assert np.isfinite(adapted).all()
    assert np.linalg.norm(adapted) == pytest.approx(1.0)
    assert adapted[1] == pytest.approx(-raw[1])


import pytest

