import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_pipeline.final_calibration_export import (
    build_final_calibration,
    compose_base_camera,
    evaluate_base_camera_predictions,
    make_base_camera_record,
    select_final_camera_calibrations,
)
from calibration_pipeline.run_export_final_camera_poses import main as pose_export_main
from calibration_pipeline.run_final_calibration_export import main as final_export_main
from calibration_pipeline.se3_utils import mat_from_t_q


def make_transform(rotation_vector, translation):
    return mat_from_t_q(
        np.asarray(translation, dtype=float),
        Rotation.from_rotvec(rotation_vector).as_quat(),
    )


def independent_record(camera_name, link_name, link_path, mount, *, observable):
    return {
        "camera_name": camera_name,
        "success": True,
        "best_link": link_name,
        "best_link_path_rel": link_path,
        "best_score": 0.004,
        "score_margin": 0.3,
        "mount_fully_observable": observable,
        "observability_rank": 12 if observable else 10,
        "observability_parameter_count": 12,
        "mount_estimation_note": (
            "fully_observable"
            if observable
            else "minimum_norm_gauge_due_to_rank_deficient_motion"
        ),
        "T_link_camera_rowmajor": mount.reshape(-1).tolist(),
    }


def fixture_summaries(recovery_succeeds=True):
    anchor_mount = make_transform([0.1, -0.2, 0.04], [0.02, -0.03, 0.08])
    independent_limited = np.eye(4)
    recovered_mount = make_transform([0.0, 3.0, 0.02], [0.04, -0.01, -0.07])
    candidates = {
        "base_frame_name": "base",
        "links": [
            {"link_name": "anchor_link", "link_path_rel": "robot/anchor_link"},
            {"link_name": "limited_link", "link_path_rel": "robot/limited_link"},
        ],
    }
    link_summary = {
        "cameras": [
            independent_record(
                "anchor_camera",
                "anchor_link",
                "robot/anchor_link",
                anchor_mount,
                observable=True,
            ),
            independent_record(
                "limited_camera",
                "limited_link",
                "robot/limited_link",
                independent_limited,
                observable=False,
            ),
        ]
    }
    recovery_result = {
        "camera_name": "limited_camera",
        "recovery_used": recovery_succeeds,
        "confidence": "high" if recovery_succeeds else "low",
        "T_link_camera_recovered_rowmajor": (
            recovered_mount.reshape(-1).tolist() if recovery_succeeds else None
        ),
        "warning": None if recovery_succeeds else "insufficient anchors",
        # Deliberately poor evaluation values prove selection does not consult GT.
        "gt_evaluation_available": True,
        "gt_recovered_translation_error_m": 999.0,
        "gt_recovered_rotation_error_deg": 179.0,
    }
    recovery_summary = {"camera_results": [recovery_result]}
    return link_summary, recovery_summary, candidates, anchor_mount, recovered_mount


def test_final_selection_prefers_recovery_and_preserves_anchor_independent_mount():
    link_summary, recovery_summary, candidates, anchor_mount, recovered_mount = (
        fixture_summaries()
    )
    cameras = select_final_camera_calibrations(
        link_summary, recovery_summary, candidates
    )
    by_name = {camera["camera_name"]: camera for camera in cameras}
    assert by_name["anchor_camera"]["calibration_source"] == (
        "independent_link_calibration"
    )
    assert np.allclose(
        np.asarray(by_name["anchor_camera"]["T_link_camera_rowmajor"]).reshape(4, 4),
        anchor_mount,
    )
    assert by_name["limited_camera"]["calibration_source"] == "shared_board_recovery"
    assert np.allclose(
        np.asarray(by_name["limited_camera"]["T_link_camera_rowmajor"]).reshape(4, 4),
        recovered_mount,
    )
    assert by_name["limited_camera"]["confidence"] == "high"


def test_failed_recovery_keeps_independent_mount_with_low_confidence_warning():
    link_summary, recovery_summary, candidates, _, _ = fixture_summaries(
        recovery_succeeds=False
    )
    cameras = select_final_camera_calibrations(
        link_summary, recovery_summary, candidates
    )
    limited = next(camera for camera in cameras if camera["camera_name"] == "limited_camera")
    assert limited["calibration_source"] == "independent_link_calibration"
    assert limited["confidence"] == "low"
    assert limited["observability"]["motion_limited"] is True
    assert "not fully observable" in limited["warnings"][0]


def test_transform_convention_composes_base_link_and_link_camera():
    base_link = make_transform([0.2, 0.1, -0.1], [0.3, -0.2, 0.5])
    link_camera = make_transform([-0.1, 0.3, 0.05], [0.04, 0.02, -0.08])
    camera_point = np.array([0.2, -0.1, 1.0, 1.0])
    composed = compose_base_camera(base_link, link_camera)
    assert np.allclose(composed @ camera_point, base_link @ (link_camera @ camera_point))


def write_metadata_dataset(dataset: Path, candidates):
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "candidate_links.json").write_text(
        json.dumps(candidates), encoding="utf-8"
    )
    (dataset / "camera_model_config.json").write_text(
        json.dumps(
            {
                "default_camera_model": "ocamcalib",
                "default_calibration_file": "camera_calibration/test.json",
                "ray_frame": "ocamcalib_raw",
                "pose_camera_frame": "unity_camera",
                "ray_frame_adapter": {
                    "type": "matrix_3x3",
                    "name": "flip_y",
                    "matrix": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset / "charuco_board_config.json").write_text(
        json.dumps(
            {
                "type": "charuco",
                "dictionary": "DICT_4X4_1000",
                "squares_x": 10,
                "squares_y": 7,
                "square_length_m": 0.08,
                "marker_length_m": 0.056,
            }
        ),
        encoding="utf-8",
    )


def test_final_calibration_schema_and_adapter_metadata_are_complete(tmp_path):
    link_summary, recovery_summary, candidates, _, _ = fixture_summaries()
    dataset = tmp_path / "dataset"
    write_metadata_dataset(dataset, candidates)
    calibration = build_final_calibration(
        dataset,
        link_summary,
        recovery_summary,
        link_calibration_path="outputs/link.json",
        shared_board_recovery_path="outputs/recovery.json",
    )
    assert calibration["calibration_version"] == "milestone_5_final_static_calibration"
    assert calibration["pose_semantics"]["T_A_B"] == "p_A = T_A_B @ p_B"
    assert calibration["ground_truth_used_for_selection"] is False
    assert calibration["camera_count"] == 2
    required_camera = {
        "camera_name",
        "attached_link",
        "T_link_camera_rowmajor",
        "t_link_camera",
        "q_link_camera_xyzw",
        "transform_convention",
        "camera_frame_convention",
        "calibration_source",
        "confidence",
        "warnings",
        "observability",
        "gt_validation_available",
        "gt_T_link_camera_translation_error_m",
        "gt_T_link_camera_rotation_error_deg",
    }
    assert required_camera <= set(calibration["cameras"][0])
    adapters = calibration["frame_adapters"]
    ray_adapter = adapters["camera_ray_to_camera_pose_adapter"]
    assert len(ray_adapter["matrix_rowmajor_3x3"]) == 9
    assert ray_adapter["affects_final_T_link_camera"] is True
    assert adapters["board_gt_adapter"] is None
    for name in ("camera_pose_frame", "camera_model", "board_pose_frame", "link_pose_frame"):
        assert len(adapters[name]) > 2
        assert "source" in adapters[name]
    assert adapters["board_pose_frame"]["definition"]["origin"]
    assert adapters["link_pose_frame"]["semantics"] == (
        "p_base = T_base_link @ p_link"
    )


def test_depth_model_pose_record_schema():
    link_summary, recovery_summary, candidates, anchor_mount, _ = fixture_summaries()
    camera = select_final_camera_calibrations(
        link_summary, recovery_summary, candidates
    )[0]
    base_link = make_transform([0.03, 0.2, -0.1], [0.4, 0.2, -0.3])
    record = make_base_camera_record("frame_000000", camera, base_link)
    required = {
        "frame_id",
        "camera_name",
        "attached_link",
        "T_base_cam_rowmajor",
        "t_base_cam",
        "q_base_cam_xyzw",
        "camera_frame_convention",
        "source",
        "transform_convention",
    }
    assert required <= set(record)
    assert np.allclose(
        np.asarray(record["T_base_cam_rowmajor"]).reshape(4, 4),
        base_link @ anchor_mount,
    )


def test_gt_validation_metrics_on_synthetic_predictions():
    truth = make_transform([0.1, -0.2, 0.05], [0.3, 0.2, 0.8])
    offset = make_transform([0.0, 0.01, 0.0], [0.005, 0.0, 0.0])
    predicted = truth @ offset
    records = [
        {
            "frame_id": "frame_000000",
            "T_base_cam_rowmajor": predicted.reshape(-1).tolist(),
        }
    ]
    metrics = evaluate_base_camera_predictions(
        records, {"frame_000000": truth}
    )
    assert metrics["gt_validation_available"] is True
    assert metrics["frames_evaluated"] == 1
    assert metrics["translation_error_m"]["mean"] == np.linalg.norm(
        predicted[:3, 3] - truth[:3, 3]
    )
    assert metrics["rotation_error_deg"]["mean"] > 0.0
    assert metrics["passed"] is True


def write_cli_fixture(root: Path):
    dataset = root / "dataset"
    link_summary, recovery_summary, candidates, _, _ = fixture_summaries()
    write_metadata_dataset(dataset, candidates)
    (dataset / "link_poses").mkdir(parents=True)
    for camera in link_summary["cameras"]:
        (dataset / "cameras" / camera["camera_name"]).mkdir(parents=True)
    for frame_index in range(2):
        links = []
        for candidate_index, candidate in enumerate(candidates["links"]):
            transform = make_transform(
                [0.0, 0.1 * frame_index, 0.0],
                [0.1 * candidate_index, 0.02 * frame_index, 0.0],
            )
            links.append(
                {
                    **candidate,
                    "valid": True,
                    "T_base_link_rowmajor": transform.reshape(-1).tolist(),
                }
            )
        frame_id = f"frame_{frame_index:06d}"
        (dataset / "link_poses" / f"{frame_id}.json").write_text(
            json.dumps({"frame_id": frame_id, "links": links}), encoding="utf-8"
        )
    link_path = root / "link.json"
    recovery_path = root / "recovery.json"
    link_path.write_text(json.dumps(link_summary), encoding="utf-8")
    recovery_path.write_text(json.dumps(recovery_summary), encoding="utf-8")
    return dataset, link_path, recovery_path


def test_final_export_and_pose_export_cli_smoke(tmp_path):
    dataset, link_path, recovery_path = write_cli_fixture(tmp_path)
    output = tmp_path / "outputs"
    assert (
        final_export_main(
            [
                "--dataset",
                str(dataset),
                "--link-calibration",
                str(link_path),
                "--shared-board-recovery",
                str(recovery_path),
                "--output",
                str(output),
                "--no-evaluate-gt",
            ]
        )
        == 0
    )
    calibration_path = output / "final_calibration" / "final_calibration.json"
    assert calibration_path.is_file()
    assert (output / "final_calibration" / "README.md").is_file()
    assert (
        pose_export_main(
            [
                "--dataset",
                str(dataset),
                "--calibration",
                str(calibration_path),
                "--output",
                str(output / "final_calibration"),
                "--no-evaluate-gt",
            ]
        )
        == 0
    )
    pose_path = (
        output
        / "final_calibration"
        / "camera_poses_base"
        / "limited_camera.jsonl"
    )
    records = [json.loads(line) for line in pose_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["source"] == "T_base_link @ final T_link_camera"
