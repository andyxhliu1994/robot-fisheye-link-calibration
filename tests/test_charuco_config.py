import cv2
import pytest

from calibration_pipeline.charuco_config import CharucoBoardConfig


def test_dataset_charuco_config_loads_and_builds_expected_board():
    config = CharucoBoardConfig.from_json("dataset/charuco_board_config.json")
    assert config.dictionary == "DICT_4X4_1000"
    assert (config.squares_x, config.squares_y) == (10, 7)
    assert config.square_length_m == pytest.approx(0.08)
    assert config.marker_length_m == pytest.approx(0.056)
    assert config.marker_count == 35
    assert hasattr(cv2, "aruco")
    board = config.create_board()
    assert board.getIds().size == 35


def test_invalid_board_geometry_is_rejected():
    with pytest.raises(ValueError, match="smaller"):
        CharucoBoardConfig("DICT_4X4_1000", 10, 7, 0.08, 0.08)

