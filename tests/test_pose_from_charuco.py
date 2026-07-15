import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration_pipeline.charuco_config import CharucoBoardConfig
from calibration_pipeline.charuco_detector import CharucoDetector
from calibration_pipeline.pose_from_charuco import (
    BOARD_POSE_FRAME,
    BoardPoseEstimate,
    estimate_pose_from_rays,
    ray_residuals,
)
from calibration_pipeline.run_pose_from_charuco import (
    load_camera_ray_components,
    process_camera_poses,
)
from calibration_pipeline.se3_utils import rotation_error_deg


def test_board_corner_coordinates_are_indexed_by_charuco_id():
    config = CharucoBoardConfig.from_json("dataset/charuco_board_config.json")
    corners = config.chessboard_corners()
    assert corners.shape == (54, 3)
    assert np.allclose(corners[0], [0.08, 0.08, 0.0])
    assert np.allclose(corners[-1], [0.72, 0.48, 0.0])
    selected = config.corner_points_for_ids(np.array([[0], [17], [53]]))
    assert np.allclose(selected, corners[[0, 17, 53]])


def _synthetic_correspondences():
    config = CharucoBoardConfig(
        "DICT_4X4_1000", 10, 7, 0.08, 0.056, marker_count=35
    )
    points = config.chessboard_corners()
    rotation_vector = np.array([0.18, -0.12, 0.06])
    translation = np.array([0.10, -0.08, 1.40])
    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    camera_points = (rotation @ points.T).T + translation
    rays = camera_points / np.linalg.norm(camera_points, axis=1, keepdims=True)
    return points, rays, rotation_vector, translation


def test_ray_residual_is_zero_at_true_pose():
    points, rays, rotation_vector, translation = _synthetic_correspondences()
    parameters = np.concatenate([rotation_vector, translation])
    assert np.linalg.norm(ray_residuals(parameters, points, rays)) < 1e-12
    perturbed = parameters.copy()
    perturbed[3] += 0.1
    assert np.linalg.norm(ray_residuals(perturbed, points, rays)) > 1e-3


def test_ray_pose_optimizer_recovers_synthetic_pose():
    points, rays, rotation_vector, translation = _synthetic_correspondences()
    estimate = estimate_pose_from_rays(points, rays)
    assert estimate.valid, estimate.reason
    assert estimate.T_camera_board is not None
    expected_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    assert np.linalg.norm(estimate.T_camera_board[:3, 3] - translation) < 1e-5
    assert rotation_error_deg(
        estimate.T_camera_board[:3, :3], expected_rotation
    ) < 1e-4
    assert estimate.mean_ray_error_deg < 1e-5
    assert estimate.num_points_behind_camera == 0


def test_pose_json_record_schema_for_success_and_failure():
    estimate = BoardPoseEstimate(
        valid=True,
        reason="ok",
        charuco_corner_count=12,
        T_camera_board=np.eye(4),
        mean_ray_error_deg=0.02,
        max_ray_error_deg=0.05,
        num_points_behind_camera=0,
        optimizer_cost=1e-6,
        optimizer_nfev=5,
    )
    record = estimate.to_record("frame_000000", "camera", "unity_camera")
    assert set(record) == {
        "frame_id",
        "camera_name",
        "camera_pose_frame",
        "board_pose_frame",
        "valid",
        "charuco_corner_count",
        "T_camera_board_rowmajor",
        "t_camera_board",
        "q_camera_board_xyzw",
        "mean_ray_error_deg",
        "max_ray_error_deg",
        "num_points_behind_camera",
        "optimizer_cost",
        "optimizer_nfev",
        "reason",
    }
    assert record["board_pose_frame"] == BOARD_POSE_FRAME
    assert len(record["T_camera_board_rowmajor"]) == 16
    failed = BoardPoseEstimate.failure("insufficient_corners", 4).to_record(
        "frame_000001", "camera", "unity_camera"
    )
    assert failed["valid"] is False
    assert failed["T_camera_board_rowmajor"] is None


def test_one_frame_real_dataset_smoke_writes_jsonl(tmp_path):
    dataset = Path("dataset")
    config = CharucoBoardConfig.from_json(dataset / "charuco_board_config.json")
    detector = CharucoDetector(config)
    model, adapter, camera_pose_frame = load_camera_ray_components(dataset)
    summary = process_camera_poses(
        dataset,
        tmp_path,
        "Fisheye180_Front_Cam1",
        config,
        detector,
        model,
        adapter,
        camera_pose_frame,
        max_frames=1,
        frame_stride=1,
        save_overlays=True,
        gt_evaluator=None,
    )
    path = tmp_path / "board_poses" / "Fisheye180_Front_Cam1.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert summary["total_frames"] == 1
    assert len(records) == 1
    assert records[0]["frame_id"] == "frame_000000"
    assert records[0]["camera_pose_frame"] == "unity_camera"
    assert records[0]["reason"] in {"ok", "insufficient_corners"}
    assert (tmp_path / "pose_overlays" / "Fisheye180_Front_Cam1" / "frame_000000.jpg").is_file()
