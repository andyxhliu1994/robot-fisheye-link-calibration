import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_pipeline.depth_model_compat import (
    compute_relative_camera_transforms,
    evaluate_depth_model_poses,
    export_depth_model_compatibility,
    make_depth_model_transform_record,
    make_sample_manifest,
    parse_depth_model_transform,
)
from calibration_pipeline.run_depth_model_compat_export import main
from calibration_pipeline.se3_utils import invert_T, mat_from_t_q, t_q_from_mat


def make_transform(rotation_vector, translation):
    return mat_from_t_q(
        np.asarray(translation, dtype=float),
        Rotation.from_rotvec(rotation_vector).as_quat(),
    )


def camera_record(name, link_name, link_path, mount, source="independent_link_calibration"):
    translation, quaternion = t_q_from_mat(mount)
    return {
        "camera_name": name,
        "attached_link": link_path,
        "attached_link_name": link_name,
        "T_link_camera_rowmajor": mount.reshape(-1).tolist(),
        "t_link_camera": translation.tolist(),
        "q_link_camera_xyzw": quaternion.tolist(),
        "calibration_source": source,
        "confidence": "high",
        "warnings": [],
        "observability": {"rank": 12, "max_rank": 12, "motion_limited": False},
        "camera_frame_convention": {
            "handedness": "right",
            "x": "right",
            "y": "up",
            "z": "forward",
        },
    }


def final_calibration_fixture():
    mount_1 = make_transform([0.1, -0.2, 0.04], [0.02, -0.03, 0.08])
    mount_2 = make_transform([-0.05, 0.15, 0.08], [-0.04, 0.01, 0.1])
    return {
        "calibration_version": "milestone_5_final_static_calibration",
        "frame_adapters": {
            "camera_ray_to_camera_pose_adapter": {
                "name": "flip_y",
                "from_frame": "ocamcalib_raw",
                "to_frame": "unity_camera",
                "matrix_rowmajor_3x3": [1, 0, 0, 0, -1, 0, 0, 0, 1],
            }
        },
        "cameras": [
            camera_record(
                "Fisheye180_Cam1", "link_1", "robot/link_1", mount_1
            ),
            camera_record(
                "Fisheye180_Cam2",
                "link_2",
                "robot/link_2",
                mount_2,
                source="shared_board_recovery",
            ),
        ],
    }


def test_per_frame_transform_json_schema():
    camera = final_calibration_fixture()["cameras"][0]
    link = make_transform([0.0, 0.2, 0.0], [0.3, 0.1, -0.2])
    record = make_depth_model_transform_record(
        "frame_000000", camera, link, np.diag([1.0, -1.0, 1.0])
    )
    required = {
        "id",
        "frame_id",
        "camera_name",
        "attached_link",
        "t_base_cam",
        "q_base_cam_xyzw",
        "T_base_cam_rowmajor",
        "camera_frame_convention",
        "ray_to_camera_rotation_rowmajor_3x3",
        "source",
        "transform_convention",
        "calibration_source",
    }
    assert required <= set(record)
    assert len(record["T_base_cam_rowmajor"]) == 16
    assert len(record["ray_to_camera_rotation_rowmajor_3x3"]) == 9


def test_parser_prefers_T_base_cam_rowmajor():
    expected = make_transform([0.1, 0.2, -0.1], [0.2, 0.3, 0.4])
    document = {
        "T_base_cam_rowmajor": expected.reshape(-1).tolist(),
        "t_base_cam": [999, 999, 999],
        "q_base_cam_xyzw": [0, 0, 0, 1],
    }
    assert np.allclose(parse_depth_model_transform(document), expected)


def test_parser_fallback_uses_translation_and_xyzw_quaternion():
    expected = make_transform([-0.2, 0.1, 0.04], [-0.1, 0.2, 0.8])
    translation, quaternion = t_q_from_mat(expected)
    parsed = parse_depth_model_transform(
        {
            "t_base_cam": translation.tolist(),
            "q_base_cam_xyzw": quaternion.tolist(),
        }
    )
    assert np.allclose(parsed, expected)


def test_runtime_composition_matches_link_times_static_mount():
    calibration = final_calibration_fixture()
    camera = calibration["cameras"][0]
    link = make_transform([0.0, -0.3, 0.1], [0.4, 0.2, -0.1])
    record = make_depth_model_transform_record(
        "frame_000000", camera, link, np.eye(3)
    )
    mount = np.asarray(camera["T_link_camera_rowmajor"]).reshape(4, 4)
    assert np.allclose(parse_depth_model_transform(record), link @ mount)


def test_relative_pose_directions_match_reference_formulas():
    target = make_transform([0.1, -0.2, 0.0], [0.2, 0.1, 0.8])
    source = make_transform([-0.05, 0.12, 0.03], [-0.3, 0.2, 0.7])
    src_tgt, tgt_src = compute_relative_camera_transforms(target, source)
    assert np.allclose(src_tgt, invert_T(source) @ target)
    assert np.allclose(tgt_src, invert_T(target) @ source)
    assert np.allclose(src_tgt @ tgt_src, np.eye(4))


def test_sample_manifest_matches_dataset_reference_structure(tmp_path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "outputs" / "depth_model_compat"
    names = ["Fisheye180_Cam1", "Fisheye180_Cam2"]
    for name in names:
        rgb = dataset / "cameras" / name / "rgb"
        rgb.mkdir(parents=True)
        (rgb / "frame_000000.jpg").touch()
    (dataset / "session_summary.json").write_text(
        json.dumps({"setup_id": "setup_a", "session_name": "traj_a"}),
        encoding="utf-8",
    )
    samples, missing_depth = make_sample_manifest(
        dataset, output, names, ["frame_000000"]
    )
    assert samples[0]["conversion_name"] == "FisheyeConversions_2"
    assert samples[0]["setup_name"] == "setup_a"
    assert samples[0]["traj_name"] == "traj_a"
    assert len(samples[0]["views"]) == 2
    assert samples[0]["views"][0]["depth_path"] is None
    assert samples[0]["views"][0]["transform_path"].endswith(
        "transforms/Fisheye180_Cam1/frame_000000.json"
    )
    assert missing_depth == 2


def test_compatibility_report_schema(tmp_path):
    dataset, calibration_path = write_cli_dataset(tmp_path, with_gt=False)
    calibration = json.loads(calibration_path.read_text())
    report, validation = export_depth_model_compatibility(
        dataset,
        calibration,
        calibration_path,
        tmp_path / "compat",
        evaluate_gt=False,
        max_frames=1,
        frame_stride=1,
        camera_name=None,
        write_jsonl=True,
        write_per_frame_json=True,
        write_sample_manifest=True,
    )
    required = {
        "final_calibration_path",
        "camera_count",
        "frames_exported",
        "total_transform_json_files_generated",
        "all_transforms_contain_T_base_cam_rowmajor",
        "all_transforms_contain_t_and_q",
        "parser_compatibility_smoke_test_passed",
        "relative_pose_computation_smoke_test_passed",
        "per_camera_exported_frame_counts",
        "adapter_metadata_summary",
        "warnings",
    }
    assert required <= set(report)
    assert report["total_transform_json_files_generated"] == 2
    assert report["parser_compatibility_smoke_test_passed"] is True
    assert report["relative_pose_computation_smoke_test_passed"] is True
    assert validation is None


def test_gt_absolute_and_relative_validation_metrics_on_synthetic_data():
    frame_id = "frame_000000"
    gt_1 = make_transform([0.0, 0.1, 0.0], [0.1, 0.0, 0.8])
    gt_2 = make_transform([0.0, -0.2, 0.05], [-0.2, 0.1, 0.9])
    offset_1 = make_transform([0.0, 0.01, 0.0], [0.005, 0.0, 0.0])
    offset_2 = make_transform([0.0, -0.01, 0.0], [-0.004, 0.0, 0.0])
    predicted = {
        "cam1": {frame_id: gt_1 @ offset_1},
        "cam2": {frame_id: gt_2 @ offset_2},
    }
    ground_truth = {"cam1": {frame_id: gt_1}, "cam2": {frame_id: gt_2}}
    report = evaluate_depth_model_poses(predicted, ground_truth, [frame_id])
    assert report["gt_validation_available"] is True
    assert report["absolute_pose"]["frames_evaluated_total"] == 2
    assert report["relative_pose"]["T_src_tgt"]["count"] == 2
    assert report["relative_pose"]["T_tgt_src"]["count"] == 2
    assert report["relative_pose"]["T_src_tgt"]["translation_error_m"]["mean"] > 0


def write_cli_dataset(root: Path, *, with_gt: bool):
    dataset = root / "dataset"
    calibration = final_calibration_fixture()
    calibration_path = root / "final_calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    (dataset / "link_poses").mkdir(parents=True)
    (dataset / "session_summary.json").write_text(
        json.dumps({"setup_id": "setup_test", "session_name": "traj_test"}),
        encoding="utf-8",
    )
    for camera in calibration["cameras"]:
        rgb = dataset / "cameras" / camera["camera_name"] / "rgb"
        rgb.mkdir(parents=True)
        if with_gt:
            (dataset / "cameras" / camera["camera_name"] / "transform").mkdir()
    for index in range(3):
        frame_id = f"frame_{index:06d}"
        links = []
        link_transforms = {}
        for camera_index, camera in enumerate(calibration["cameras"]):
            link = make_transform(
                [0.0, 0.1 * index, 0.03 * camera_index],
                [0.15 * camera_index, 0.02 * index, -0.04 * camera_index],
            )
            link_transforms[camera["camera_name"]] = link
            links.append(
                {
                    "link_name": camera["attached_link_name"],
                    "link_path_rel": camera["attached_link"],
                    "valid": True,
                    "T_base_link_rowmajor": link.reshape(-1).tolist(),
                }
            )
            rgb_path = (
                dataset
                / "cameras"
                / camera["camera_name"]
                / "rgb"
                / f"{frame_id}.jpg"
            )
            rgb_path.touch()
        (dataset / "link_poses" / f"{frame_id}.json").write_text(
            json.dumps({"frame_id": frame_id, "links": links}), encoding="utf-8"
        )
        if with_gt:
            for camera in calibration["cameras"]:
                mount = np.asarray(camera["T_link_camera_rowmajor"]).reshape(4, 4)
                gt = link_transforms[camera["camera_name"]] @ mount
                path = (
                    dataset
                    / "cameras"
                    / camera["camera_name"]
                    / "transform"
                    / f"{frame_id}.json"
                )
                path.write_text(
                    json.dumps({"T_base_cam_rowmajor": gt.reshape(-1).tolist()}),
                    encoding="utf-8",
                )
    return dataset, calibration_path


def test_cli_smoke_writes_all_outputs_and_gt_report(tmp_path):
    dataset, calibration_path = write_cli_dataset(tmp_path, with_gt=True)
    output = tmp_path / "outputs" / "depth_model_compat"
    assert (
        main(
            [
                "--dataset",
                str(dataset),
                "--calibration",
                str(calibration_path),
                "--output",
                str(output),
                "--evaluate-gt",
            ]
        )
        == 0
    )
    report = json.loads(
        (output / "depth_model_compatibility_report.json").read_text()
    )
    validation = json.loads((output / "depth_model_pose_validation.json").read_text())
    samples = json.loads((output / "depth_model_samples.json").read_text())
    assert report["total_transform_json_files_generated"] == 6
    assert report["sample_manifest_sample_count"] == 3
    assert validation["absolute_pose"]["frames_evaluated_total"] == 6
    assert validation["relative_pose"]["T_src_tgt"]["count"] == 6
    assert len(samples) == 3
    assert (output / "camera_poses_base" / "Fisheye180_Cam1.jsonl").is_file()


def test_no_gt_mode_still_exports_compatibility_artifacts(tmp_path):
    dataset, calibration_path = write_cli_dataset(tmp_path, with_gt=False)
    output = tmp_path / "outputs" / "depth_model_compat"
    assert (
        main(
            [
                "--dataset",
                str(dataset),
                "--calibration",
                str(calibration_path),
                "--output",
                str(output),
                "--no-evaluate-gt",
                "--max-frames",
                "2",
            ]
        )
        == 0
    )
    report = json.loads(
        (output / "depth_model_compatibility_report.json").read_text()
    )
    assert report["frames_exported"] == 2
    assert report["total_transform_json_files_generated"] == 4
    assert report["gt_validation_report_written"] is False
    assert not (output / "depth_model_pose_validation.json").exists()
