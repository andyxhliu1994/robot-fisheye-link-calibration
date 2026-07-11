import json

import cv2
import numpy as np

from calibration_pipeline.charuco_config import CharucoBoardConfig
from calibration_pipeline.charuco_detector import CharucoDetector
from calibration_pipeline.run_charuco_detection import process_camera


def make_config() -> CharucoBoardConfig:
    return CharucoBoardConfig(
        dictionary="DICT_4X4_1000",
        squares_x=10,
        squares_y=7,
        square_length_m=0.08,
        marker_length_m=0.056,
        marker_count=35,
    )


def render_board(config: CharucoBoardConfig) -> np.ndarray:
    board = config.create_board()
    if hasattr(board, "generateImage"):
        return board.generateImage((1000, 700), marginSize=30, borderBits=1)
    return board.draw((1000, 700), marginSize=30, borderBits=1)


def test_detector_initialization_and_partial_visibility_thresholds():
    detector = CharucoDetector(make_config())
    image = render_board(detector.config)
    detection = detector.detect(image, "frame_000000", "synthetic_camera")
    assert detection.valid
    assert detection.marker_count >= 4
    assert detection.charuco_corner_count >= 8
    assert detection.board_bbox_area_px is not None
    assert detection.board_bbox_area_px > 0
    overlay = detector.draw_overlay(image, detection)
    assert overlay.shape == (700, 1000, 3)


def test_blank_frame_is_saved_as_invalid_jsonl_record(tmp_path):
    dataset = tmp_path / "dataset"
    rgb_dir = dataset / "cameras" / "test_camera" / "rgb"
    rgb_dir.mkdir(parents=True)
    cv2.imwrite(str(rgb_dir / "frame_000000.jpg"), np.full((700, 1000), 255, np.uint8))
    cv2.imwrite(str(rgb_dir / "frame_000001.jpg"), render_board(make_config()))

    output = tmp_path / "outputs"
    summary = process_camera(
        dataset,
        output,
        "test_camera",
        CharucoDetector(make_config()),
        max_frames=2,
        frame_stride=1,
        save_overlays=True,
    )
    lines = (output / "detections" / "test_camera.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert set(records[0]) == {
        "frame_id",
        "camera_name",
        "valid",
        "marker_count",
        "charuco_corner_count",
        "board_bbox_area_px",
        "reason",
    }
    assert records[0]["valid"] is False
    assert records[0]["reason"] == "no_markers_detected"
    assert records[1]["valid"] is True
    assert summary["total_frames"] == 2
    assert len(list((output / "debug_overlays" / "test_camera").glob("*.jpg"))) == 2

