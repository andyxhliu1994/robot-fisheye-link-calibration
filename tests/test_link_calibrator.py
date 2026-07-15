import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration_pipeline.link_calibrator import (
    BoardPoseObservation,
    board_consistency_residuals,
    rank_link_hypotheses,
)
from calibration_pipeline.run_link_calibration import main
from calibration_pipeline.se3_utils import (
    invert_T,
    mat_from_t_q,
    rotation_error_deg,
    translation_error_m,
)


def make_transform(rotation_vector, translation):
    return mat_from_t_q(
        np.asarray(translation, dtype=float),
        Rotation.from_rotvec(rotation_vector).as_quat(),
    )


def synthetic_link_problem(frame_count=16):
    T_link_camera = make_transform([0.18, -0.22, 0.09], [0.04, -0.03, 0.12])
    T_base_board = make_transform([-0.12, 0.17, 0.08], [0.3, 0.2, 1.1])
    observations = []
    correct_poses = []
    wrong_poses = []
    for index in range(frame_count):
        phase = index / max(frame_count - 1, 1)
        correct = make_transform(
            [
                0.45 * np.sin(2.2 * phase),
                0.35 * np.cos(1.7 * phase),
                0.28 * np.sin(3.1 * phase),
            ],
            [
                0.18 * np.sin(1.3 * phase),
                0.12 * np.cos(2.1 * phase),
                0.08 * np.sin(2.7 * phase),
            ],
        )
        wrong = make_transform(
            [
                -0.3 * np.cos(2.6 * phase),
                0.4 * np.sin(1.1 * phase),
                -0.25 * np.cos(3.7 * phase),
            ],
            [
                -0.12 * np.cos(2.4 * phase),
                0.16 * np.sin(1.8 * phase),
                0.1 * np.cos(1.5 * phase),
            ],
        )
        T_camera_board = invert_T(T_link_camera) @ invert_T(correct) @ T_base_board
        observations.append(
            BoardPoseObservation(
                f"frame_{index:06d}", T_camera_board, 30, 0.02
            )
        )
        correct_poses.append(correct)
        wrong_poses.append(wrong)
    candidates = [
        {"link_name": "correct_link", "link_path_rel": "robot/correct_link"},
        {"link_name": "wrong_link", "link_path_rel": "robot/wrong_link"},
    ]
    link_poses = {
        "robot/correct_link": correct_poses,
        "robot/wrong_link": wrong_poses,
    }
    return observations, candidates, link_poses, T_link_camera


def test_joint_residual_is_zero_for_known_mount_and_board():
    observations, _, links, T_link_camera = synthetic_link_problem()
    T_base_board = links["robot/correct_link"][0] @ T_link_camera @ observations[0].T_camera_board
    parameters = np.concatenate(
        [
            np.r_[
                Rotation.from_matrix(T_link_camera[:3, :3]).as_rotvec(),
                T_link_camera[:3, 3],
            ],
            np.r_[
                Rotation.from_matrix(T_base_board[:3, :3]).as_rotvec(),
                T_base_board[:3, 3],
            ],
        ]
    )
    residual = board_consistency_residuals(
        parameters,
        links["robot/correct_link"],
        [item.T_camera_board for item in observations],
    )
    assert np.linalg.norm(residual) < 1e-10


def test_correct_link_ranks_first_and_mount_is_recovered():
    observations, candidates, link_poses, expected_mount = synthetic_link_problem()
    result = rank_link_hypotheses(
        "synthetic_camera", observations, candidates, link_poses, min_valid_poses=10
    )
    record = result.to_record()
    assert record["best_link"] == "correct_link"
    assert record["second_best_link"] == "wrong_link"
    assert record["score_margin"] > 0.01
    estimated = np.asarray(record["T_link_camera_rowmajor"]).reshape(4, 4)
    assert translation_error_m(estimated[:3, 3], expected_mount[:3, 3]) < 1e-5
    assert rotation_error_deg(estimated[:3, :3], expected_mount[:3, :3]) < 1e-4
    assert record["hypotheses"][0]["rank"] == 1
    assert record["hypotheses"][0]["score"] < record["hypotheses"][1]["score"]
    assert record["mount_fully_observable"] is True


def test_single_axis_motion_uses_bounded_gauge_and_reports_rank_deficiency():
    T_link_camera = make_transform([0.0, 3.05, 0.0], [0.05, -0.02, -0.06])
    T_base_board = make_transform([0.1, -0.2, 0.05], [0.4, 0.1, 1.2])
    observations = []
    poses = []
    for index, angle in enumerate(np.linspace(-0.8, 0.9, 20)):
        link_pose = make_transform([0.0, angle, 0.0], [0.0, 0.2, 0.0])
        camera_board = invert_T(T_link_camera) @ invert_T(link_pose) @ T_base_board
        observations.append(BoardPoseObservation(f"frame_{index:06d}", camera_board))
        poses.append(link_pose)
    result = rank_link_hypotheses(
        "single_axis_camera",
        observations,
        [{"link_name": "axis_link", "link_path_rel": "robot/axis_link"}],
        {"robot/axis_link": poses},
    ).to_record()
    assert result["best_score"] < 1e-5
    assert result["mount_fully_observable"] is False
    assert result["observability_rank"] < result["observability_parameter_count"]
    assert np.linalg.norm(result["t_link_camera"]) < 1.0
    assert result["mount_estimation_note"] == (
        "minimum_norm_gauge_due_to_rank_deficient_motion"
    )


def test_camera_result_json_schema():
    observations, candidates, link_poses, _ = synthetic_link_problem()
    record = rank_link_hypotheses(
        "synthetic_camera", observations, candidates, link_poses
    ).to_record()
    required = {
        "camera_name",
        "best_link",
        "second_best_link",
        "best_score",
        "second_best_score",
        "score_margin",
        "num_valid_frames",
        "mount_fully_observable",
        "observability_rank",
        "observability_parameter_count",
        "mount_estimation_note",
        "T_link_camera_rowmajor",
        "t_link_camera",
        "q_link_camera_xyzw",
        "board_consistency_translation_m",
        "board_consistency_rotation_deg",
        "hypotheses",
        "gt_evaluation_available",
        "gt_best_link_correct",
        "gt_T_link_camera_translation_error_m",
        "gt_T_link_camera_rotation_error_deg",
    }
    assert required <= set(record)
    assert len(record["T_link_camera_rowmajor"]) == 16
    assert all("rank" in hypothesis for hypothesis in record["hypotheses"])


def _write_synthetic_cli_dataset(
    dataset: Path, board_pose_dir: Path
) -> None:
    observations, candidates, link_poses, _ = synthetic_link_problem(frame_count=12)
    (dataset / "cameras" / "synthetic_camera" / "rgb").mkdir(parents=True)
    (dataset / "link_poses").mkdir(parents=True)
    (dataset / "candidate_links.json").write_text(
        json.dumps(
            {
                "base_frame_name": "base",
                "link_count": len(candidates),
                "links": candidates,
            }
        ),
        encoding="utf-8",
    )
    board_pose_dir.mkdir(parents=True)
    with (board_pose_dir / "synthetic_camera.jsonl").open("w", encoding="utf-8") as stream:
        for index, observation in enumerate(observations):
            stream.write(
                json.dumps(
                    {
                        "frame_id": observation.frame_id,
                        "camera_name": "synthetic_camera",
                        "valid": True,
                        "charuco_corner_count": 30,
                        "mean_ray_error_deg": 0.02,
                        "T_camera_board_rowmajor": observation.T_camera_board.reshape(-1).tolist(),
                    }
                )
                + "\n"
            )
            links = []
            for candidate in candidates:
                path = candidate["link_path_rel"]
                links.append(
                    {
                        "link_name": candidate["link_name"],
                        "link_path_rel": path,
                        "valid": True,
                        "T_base_link_rowmajor": link_poses[path][index].reshape(-1).tolist(),
                    }
                )
            (dataset / "link_poses" / f"frame_{index:06d}.json").write_text(
                json.dumps({"frame_id": f"frame_{index:06d}", "links": links}),
                encoding="utf-8",
            )


def test_cli_smoke_writes_ranked_summary(tmp_path):
    dataset = tmp_path / "dataset"
    board_poses = tmp_path / "board_poses"
    output = tmp_path / "outputs"
    _write_synthetic_cli_dataset(dataset, board_poses)
    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--board-poses",
            str(board_poses),
            "--output",
            str(output),
            "--no-evaluate-gt",
        ]
    )
    assert exit_code == 0
    summary_path = output / "link_calibration" / "link_calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["ground_truth_used_for_estimation"] is False
    assert summary["cameras"][0]["best_link"] == "correct_link"
    assert len(summary["cameras"][0]["hypotheses"]) == 2
