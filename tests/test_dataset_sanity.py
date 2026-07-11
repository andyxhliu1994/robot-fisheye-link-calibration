import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration_pipeline.dataset_sanity import (
    diagnose_dataset,
    image_difference,
    pose_pairwise_comparisons,
    run_dataset_sanity_check,
    sha256_file,
)


def test_sha256_file(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_image_difference_metrics_and_shape_mismatch():
    first = np.zeros((2, 2, 3), np.uint8)
    second = np.full((2, 2, 3), 10, np.uint8)
    result = image_difference(first, second)
    assert result["same_shape"]
    assert result["mean_absolute_difference"] == pytest.approx(10.0)
    assert result["max_absolute_difference"] == 10
    assert result["rmse"] == pytest.approx(10.0)
    assert image_difference(first, np.zeros((3, 2, 3), np.uint8))["same_shape"] is False


def test_pose_pairwise_comparison():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.5
    second[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, np.pi / 2]))[0]
    comparisons = pose_pairwise_comparisons({"camera_a": first, "camera_b": second})
    assert len(comparisons) == 1
    assert comparisons[0]["translation_difference_m"] == pytest.approx(0.5)
    assert comparisons[0]["rotation_difference_deg"] == pytest.approx(90.0)


def test_expected_image_export_suspect_diagnosis():
    diagnosis = diagnose_dataset(
        input_complete=True,
        all_frames_all_cameras_byte_identical=True,
        affected_frame_count=5,
        transforms_different=True,
    )
    assert diagnosis["passed"] is False
    assert diagnosis["status"] == "FAIL"
    assert diagnosis["diagnosis_code"] == "DATASET_IMAGE_EXPORT_SUSPECT"


def _write_transform(path: Path, translation: list[float], link: str) -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    document = {
        "T_base_cam_rowmajor": matrix.reshape(-1).tolist(),
        "gt_link_path_rel": link,
        "gt_t_link_from_cam": translation,
        "gt_q_link_from_cam_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_report_schema_on_tiny_duplicated_stream_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "session_summary.json").parent.mkdir(parents=True)
    (root / "session_summary.json").write_text(
        json.dumps({"camera_count": 2, "frame_count": 2}), encoding="utf-8"
    )
    (root / "candidate_links.json").write_text(
        json.dumps({"link_count": 2}), encoding="utf-8"
    )
    for camera_index, camera in enumerate(("camera_a", "camera_b")):
        rgb = root / "cameras" / camera / "rgb"
        transforms = root / "cameras" / camera / "transform"
        rgb.mkdir(parents=True)
        transforms.mkdir(parents=True)
        for frame_index in range(2):
            image = np.full((16, 16, 3), 40 + frame_index, np.uint8)
            cv2.imwrite(str(rgb / f"frame_{frame_index:06d}.jpg"), image)
            _write_transform(
                transforms / f"frame_{frame_index:06d}.json",
                [float(camera_index), 0.0, 0.0],
                f"link_{camera_index}",
            )

    report = run_dataset_sanity_check(
        root, max_frames=2, frame_stride=1, output_root_for_detections=tmp_path / "outputs"
    )
    assert report["diagnosis_code"] == "DATASET_IMAGE_EXPORT_SUSPECT"
    assert report["sampled_frame_count"] == 2
    assert report["rgb_same_frame_identity"][
        "all_sampled_frames_byte_identical_across_all_cameras"
    ]
    assert report["camera_transform_distinctness"][
        "transforms_different_across_cameras"
    ]

