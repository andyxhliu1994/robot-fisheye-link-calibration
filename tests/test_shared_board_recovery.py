import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_pipeline.link_calibrator import BoardPoseObservation
from calibration_pipeline.run_shared_board_recovery import main
from calibration_pipeline.se3_utils import (
    invert_T,
    mat_from_t_q,
    rotation_error_deg,
    translation_error_m,
)
from calibration_pipeline.shared_board_recovery import perform_shared_board_recovery


def make_transform(rotation_vector, translation):
    return mat_from_t_q(
        np.asarray(translation, dtype=float),
        Rotation.from_rotvec(rotation_vector).as_quat(),
    )


def independent_record(camera_name, link_path, mount, *, observable):
    return {
        "camera_name": camera_name,
        "success": True,
        "failure_reason": "ok",
        "best_link": link_path.rsplit("/", 1)[-1],
        "best_link_path_rel": link_path,
        "best_score": 0.003,
        "second_best_score": 0.4,
        "score_margin": 0.397,
        "num_valid_frames": 16,
        "mount_fully_observable": observable,
        "observability_rank": 12 if observable else 10,
        "observability_parameter_count": 12,
        "mount_estimation_note": (
            "fully_observable"
            if observable
            else "minimum_norm_gauge_due_to_rank_deficient_motion"
        ),
        "T_link_camera_rowmajor": mount.reshape(-1).tolist(),
        "t_link_camera": mount[:3, 3].tolist(),
        "q_link_camera_xyzw": Rotation.from_matrix(mount[:3, :3]).as_quat().tolist(),
        "board_consistency_translation_m": {
            "mean": 0.002,
            "median": 0.002,
            "max": 0.004,
        },
        "board_consistency_rotation_deg": {
            "mean": 0.2,
            "median": 0.2,
            "max": 0.4,
        },
        "hypotheses": [],
        "gt_evaluation_available": False,
    }


def synthetic_shared_board_problem(anchor_count=2, motion_count=1, frame_count=16):
    shared_board = make_transform([0.1, -0.16, 0.08], [0.35, -0.12, 1.15])
    records = []
    observations = {}
    link_poses = {}
    expected_mounts = {}
    for camera_index in range(anchor_count + motion_count):
        is_anchor = camera_index < anchor_count
        camera_name = f"camera_{camera_index}"
        link_path = f"robot/link_{camera_index}"
        mount = make_transform(
            [0.12 + 0.04 * camera_index, -0.2, 0.06],
            [0.03 * camera_index, -0.02, 0.08 + 0.01 * camera_index],
        )
        expected_mounts[camera_name] = mount
        independent_mount = mount if is_anchor else np.eye(4)
        records.append(
            independent_record(
                camera_name, link_path, independent_mount, observable=is_anchor
            )
        )
        camera_observations = []
        camera_links = []
        for frame_index in range(frame_count):
            phase = frame_index / max(frame_count - 1, 1)
            if is_anchor:
                link = make_transform(
                    [
                        0.35 * np.sin(2.1 * phase + camera_index),
                        0.28 * np.cos(1.7 * phase),
                        0.22 * np.sin(2.8 * phase),
                    ],
                    [
                        0.15 * np.sin(1.3 * phase),
                        0.1 * np.cos(2.0 * phase + camera_index),
                        0.06 * np.sin(2.5 * phase),
                    ],
                )
            else:
                link = make_transform(
                    [0.0, -0.7 + 1.4 * phase, 0.0],
                    [0.0, 0.18 + 0.01 * camera_index, 0.0],
                )
            camera_board = invert_T(mount) @ invert_T(link) @ shared_board
            camera_observations.append(
                BoardPoseObservation(f"frame_{frame_index:06d}", camera_board)
            )
            camera_links.append(link)
        observations[camera_name] = camera_observations
        link_poses[camera_name] = camera_links
    return (
        {"schema_version": 1, "cameras": records},
        observations,
        link_poses,
        expected_mounts,
    )


def recovered_transform(result):
    return np.asarray(result["T_link_camera_recovered_rowmajor"]).reshape(4, 4)


def test_two_anchors_recover_single_axis_mount_gauge():
    summary, observations, links, expected = synthetic_shared_board_problem()
    result = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
    )
    recovered = result["camera_results"][0]
    estimate = recovered_transform(recovered)
    truth = expected["camera_2"]
    assert result["status"] == "recovered"
    assert result["anchor_cameras"] == ["camera_0", "camera_1"]
    assert result["motion_limited_cameras"] == ["camera_2"]
    assert recovered["confidence"] == "high"
    assert translation_error_m(estimate[:3, 3], truth[:3, 3]) < 1e-6
    assert rotation_error_deg(estimate[:3, :3], truth[:3, :3]) < 1e-5


def test_one_anchor_requires_opt_in_and_reports_medium_confidence():
    summary, observations, links, _ = synthetic_shared_board_problem(anchor_count=1)
    blocked = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
    )
    assert blocked["status"] == "insufficient_anchor"
    assert blocked["camera_results"][0]["recovery_used"] is False
    allowed = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
        allow_single_anchor=True,
    )
    recovered = allowed["camera_results"][0]
    assert allowed["status"] == "recovered"
    assert recovered["confidence"] == "medium"
    assert "cannot cross-validate" in recovered["warning"]


def test_zero_anchor_does_not_emit_trusted_mount():
    summary, observations, links, _ = synthetic_shared_board_problem(
        anchor_count=0, motion_count=1
    )
    result = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
    )
    camera = result["camera_results"][0]
    assert result["status"] == "insufficient_anchor"
    assert camera["confidence"] == "low"
    assert camera["T_link_camera_recovered_rowmajor"] is None
    assert "not observable" in camera["warning"]


def test_multiple_motion_limited_cameras_are_recovered_independently():
    summary, observations, links, expected = synthetic_shared_board_problem(
        anchor_count=2, motion_count=2
    )
    result = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
    )
    assert result["motion_limited_cameras"] == ["camera_2", "camera_3"]
    assert len(result["camera_results"]) == 2
    for camera in result["camera_results"]:
        estimate = recovered_transform(camera)
        truth = expected[camera["camera_name"]]
        assert translation_error_m(estimate[:3, 3], truth[:3, 3]) < 1e-6
        assert rotation_error_deg(estimate[:3, :3], truth[:3, :3]) < 1e-5


def test_no_motion_limited_camera_skips_recovery_cleanly():
    summary, observations, links, _ = synthetic_shared_board_problem(
        anchor_count=2, motion_count=0
    )
    result = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
    )
    assert result["status"] == "no_recovery_needed"
    assert result["no_recovery_needed"] is True
    assert result["shared_board_estimated"] is False
    assert result["camera_results"] == []


def test_recovered_json_schema_contains_required_fields():
    summary, observations, links, _ = synthetic_shared_board_problem()
    result = perform_shared_board_recovery(
        summary,
        observations,
        links,
        input_summary_path="independent.json",
    )
    required_summary = {
        "input_link_calibration_summary",
        "anchor_cameras",
        "motion_limited_cameras",
        "status",
        "T_base_board_shared_rowmajor",
        "anchor_agreement",
        "camera_classifications",
        "camera_results",
        "ground_truth_used_for_estimation",
    }
    required_camera = {
        "camera_name",
        "best_link",
        "original_independent_observability_rank",
        "original_independent_observability_max_rank",
        "recovery_used",
        "recovery_method",
        "anchor_camera_count",
        "confidence",
        "T_link_camera_recovered_rowmajor",
        "t_link_camera_recovered",
        "q_link_camera_recovered_xyzw",
        "recovery_consistency_translation_m",
        "recovery_consistency_rotation_deg",
        "warning",
        "gt_evaluation_available",
        "gt_recovered_translation_error_m",
        "gt_recovered_rotation_error_deg",
    }
    assert required_summary <= set(result)
    assert required_camera <= set(result["camera_results"][0])
    assert result["ground_truth_used_for_estimation"] is False


def write_cli_fixture(
    dataset: Path, board_pose_dir: Path, link_summary_path: Path
) -> None:
    summary, observations, links, _ = synthetic_shared_board_problem(frame_count=12)
    candidates = []
    for record in summary["cameras"]:
        camera_name = record["camera_name"]
        link_path = record["best_link_path_rel"]
        candidates.append(
            {"link_name": record["best_link"], "link_path_rel": link_path}
        )
        (dataset / "cameras" / camera_name / "rgb").mkdir(parents=True)
        board_pose_dir.mkdir(parents=True, exist_ok=True)
        with (board_pose_dir / f"{camera_name}.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for observation in observations[camera_name]:
                stream.write(
                    json.dumps(
                        {
                            "frame_id": observation.frame_id,
                            "camera_name": camera_name,
                            "valid": True,
                            "T_camera_board_rowmajor": observation.T_camera_board.reshape(
                                -1
                            ).tolist(),
                        }
                    )
                    + "\n"
                )
    (dataset / "link_poses").mkdir(parents=True)
    for frame_index in range(12):
        frame_id = f"frame_{frame_index:06d}"
        frame_links = []
        for record in summary["cameras"]:
            camera_name = record["camera_name"]
            frame_links.append(
                {
                    "link_name": record["best_link"],
                    "link_path_rel": record["best_link_path_rel"],
                    "valid": True,
                    "T_base_link_rowmajor": links[camera_name][frame_index].reshape(
                        -1
                    ).tolist(),
                }
            )
        (dataset / "link_poses" / f"{frame_id}.json").write_text(
            json.dumps({"frame_id": frame_id, "links": frame_links}),
            encoding="utf-8",
        )
    (dataset / "candidate_links.json").write_text(
        json.dumps({"base_frame_name": "base", "links": candidates}),
        encoding="utf-8",
    )
    link_summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_cli_smoke_writes_shared_board_summary(tmp_path):
    dataset = tmp_path / "dataset"
    board_poses = tmp_path / "board_poses"
    link_summary = tmp_path / "link_calibration_summary.json"
    output = tmp_path / "outputs"
    write_cli_fixture(dataset, board_poses, link_summary)
    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--link-calibration",
            str(link_summary),
            "--board-poses",
            str(board_poses),
            "--output",
            str(output),
            "--no-evaluate-gt",
        ]
    )
    assert exit_code == 0
    report_path = output / "shared_board_recovery" / "shared_board_recovery_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "recovered"
    assert report["camera_results"][0]["confidence"] == "high"
