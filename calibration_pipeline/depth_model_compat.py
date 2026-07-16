"""Depth-model input compatibility export and evaluation-only pose validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_loader import frame_number, load_json_document
from .final_calibration_export import (
    load_camera_gt_transforms,
    make_base_camera_record,
)
from .kinematics_provider import UnityLinkPoseProvider
from .se3_utils import invert_T, mat_from_t_q, rotation_error_deg


def parse_depth_model_transform(document: Mapping[str, Any]) -> np.ndarray:
    """Mimic the depth-model parser: matrix first, then translation/quaternion."""
    matrix_values = document.get("T_base_cam_rowmajor")
    if matrix_values is not None:
        matrix = np.asarray(matrix_values, dtype=float)
        if matrix.size != 16 or not np.all(np.isfinite(matrix)):
            raise ValueError("T_base_cam_rowmajor must contain 16 finite values")
        transform = matrix.reshape(4, 4)
    else:
        transform = mat_from_t_q(
            np.asarray(document["t_base_cam"], dtype=float),
            np.asarray(document["q_base_cam_xyzw"], dtype=float),
        )
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("Parsed T_base_cam must be a homogeneous transform")
    return transform


def compute_relative_camera_transforms(
    T_base_target: np.ndarray, T_base_source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-to-source and source-to-target camera transforms.

    ``T_src_tgt`` maps target-camera points into the source camera frame.
    ``T_tgt_src`` maps source-camera points into the target camera frame.
    """
    T_src_tgt = invert_T(T_base_source) @ T_base_target
    T_tgt_src = invert_T(T_base_target) @ T_base_source
    return T_src_tgt, T_tgt_src


def make_depth_model_transform_record(
    frame_id: str,
    camera: Mapping[str, Any],
    T_base_link: np.ndarray,
    ray_adapter_matrix: np.ndarray,
) -> dict[str, Any]:
    record = make_base_camera_record(frame_id, camera, T_base_link)
    record.update(
        {
            "id": frame_id,
            "ray_to_camera_rotation_rowmajor_3x3": np.asarray(
                ray_adapter_matrix, dtype=float
            ).reshape(9).tolist(),
            "calibration_source": str(camera["calibration_source"]),
        }
    )
    record["transform_convention"]["runtime_composition"] = (
        "T_base_cam = T_base_link @ T_link_camera"
    )
    return record


def _display_path(path: Path) -> str:
    value = str(path)
    return value if path.is_absolute() or value.startswith("./") else f"./{value}"


def _selected_common_frame_ids(
    dataset_root: Path,
    camera_names: Sequence[str],
    max_frames: int | None,
    frame_stride: int,
) -> list[str]:
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    common = {
        path.stem for path in (dataset_root / "link_poses").glob("frame_*.json")
    }
    for camera_name in camera_names:
        common &= {
            path.stem
            for path in (dataset_root / "cameras" / camera_name / "rgb").glob(
                "frame_*.jpg"
            )
        }
    frame_ids = sorted(common, key=frame_number)[::frame_stride]
    return frame_ids if max_frames is None else frame_ids[:max_frames]


def _depth_model_conversion_name(camera_names: Sequence[str]) -> str:
    fovs = {
        int(match.group(1))
        for name in camera_names
        if (match := re.search(r"Fisheye(\d+)", name)) is not None
    }
    reference_keys = {
        180: "FisheyeConversions_2",
        210: "FisheyeConversions_3",
        240: "FisheyeConversions_4",
    }
    if len(fovs) == 1 and next(iter(fovs)) in reference_keys:
        return reference_keys[next(iter(fovs))]
    return "final_calibration_to_depth_model_inputs"


def _session_metadata(
    dataset_root: Path, camera_names: Sequence[str]
) -> dict[str, str]:
    conversion_name = _depth_model_conversion_name(camera_names)
    path = dataset_root / "session_summary.json"
    if not path.is_file():
        return {
            "conversion_name": conversion_name,
            "setup_name": "unknown_setup",
            "traj_name": dataset_root.name,
        }
    summary, _ = load_json_document(path)
    return {
        "conversion_name": conversion_name,
        "setup_name": str(summary.get("setup_id", "unknown_setup")),
        "traj_name": str(summary.get("session_name", dataset_root.name)),
    }


def make_sample_manifest(
    dataset_root: Path,
    output_root: Path,
    camera_names: Sequence[str],
    frame_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    session = _session_metadata(dataset_root, camera_names)
    samples = []
    missing_depth_count = 0
    for frame_id in frame_ids:
        views = []
        for camera_name in camera_names:
            rgb_path = dataset_root / "cameras" / camera_name / "rgb" / f"{frame_id}.jpg"
            depth_path = (
                dataset_root
                / "cameras"
                / camera_name
                / "depth"
                / f"{frame_id}.exr"
            )
            if not depth_path.is_file():
                depth_value = None
                missing_depth_count += 1
            else:
                depth_value = _display_path(depth_path)
            views.append(
                {
                    "cam_name": camera_name,
                    "rgb_path": _display_path(rgb_path),
                    "depth_path": depth_value,
                    "transform_path": _display_path(
                        output_root
                        / "transforms"
                        / camera_name
                        / f"{frame_id}.json"
                    ),
                }
            )
        samples.append({**session, "frame_id": frame_id, "views": views})
    return samples, missing_depth_count


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {"mean": fmean(values), "median": median(values), "max": max(values)}


def _transform_error(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> tuple[float, float]:
    return (
        float(np.linalg.norm(predicted[:3, 3] - ground_truth[:3, 3])),
        rotation_error_deg(predicted[:3, :3], ground_truth[:3, :3]),
    )


def _error_record(
    translation_errors: Sequence[float], rotation_errors: Sequence[float]
) -> dict[str, Any]:
    if not translation_errors:
        return {
            "count": 0,
            "translation_error_m": None,
            "rotation_error_deg": None,
        }
    return {
        "count": len(translation_errors),
        "translation_error_m": _metric_summary(translation_errors),
        "rotation_error_deg": _metric_summary(rotation_errors),
    }


def evaluate_depth_model_poses(
    predicted_by_camera: Mapping[str, Mapping[str, np.ndarray]],
    gt_by_camera: Mapping[str, Mapping[str, np.ndarray]],
    frame_ids: Sequence[str],
) -> dict[str, Any]:
    camera_names = list(predicted_by_camera)
    absolute_records = []
    for camera_name in camera_names:
        translations = []
        rotations = []
        predicted = predicted_by_camera[camera_name]
        ground_truth = gt_by_camera.get(camera_name, {})
        for frame_id in frame_ids:
            if frame_id not in predicted or frame_id not in ground_truth:
                continue
            translation, rotation = _transform_error(
                predicted[frame_id], ground_truth[frame_id]
            )
            translations.append(translation)
            rotations.append(rotation)
        absolute_records.append(
            {"camera_name": camera_name, **_error_record(translations, rotations)}
        )

    src_tgt_translations = []
    src_tgt_rotations = []
    tgt_src_translations = []
    tgt_src_rotations = []
    per_target = []
    for target_name in camera_names:
        target_translations = []
        target_rotations = []
        for source_name in camera_names:
            if source_name == target_name:
                continue
            for frame_id in frame_ids:
                if (
                    frame_id not in predicted_by_camera[target_name]
                    or frame_id not in predicted_by_camera[source_name]
                    or frame_id not in gt_by_camera.get(target_name, {})
                    or frame_id not in gt_by_camera.get(source_name, {})
                ):
                    continue
                pred_src_tgt, pred_tgt_src = compute_relative_camera_transforms(
                    predicted_by_camera[target_name][frame_id],
                    predicted_by_camera[source_name][frame_id],
                )
                gt_src_tgt, gt_tgt_src = compute_relative_camera_transforms(
                    gt_by_camera[target_name][frame_id],
                    gt_by_camera[source_name][frame_id],
                )
                translation, rotation = _transform_error(pred_src_tgt, gt_src_tgt)
                reverse_translation, reverse_rotation = _transform_error(
                    pred_tgt_src, gt_tgt_src
                )
                src_tgt_translations.append(translation)
                src_tgt_rotations.append(rotation)
                tgt_src_translations.append(reverse_translation)
                tgt_src_rotations.append(reverse_rotation)
                target_translations.append(translation)
                target_rotations.append(rotation)
        per_target.append(
            {
                "target_camera": target_name,
                **_error_record(target_translations, target_rotations),
            }
        )
    evaluated_absolute = sum(record["count"] for record in absolute_records)
    return {
        "schema_version": 1,
        "evaluation_only": True,
        "ground_truth_used_to_modify_calibration": False,
        "gt_validation_available": evaluated_absolute > 0,
        "absolute_pose": {
            "semantics": "p_base = T_base_cam @ p_camera",
            "frames_evaluated_total": evaluated_absolute,
            "cameras": absolute_records,
        },
        "relative_pose": {
            "T_src_tgt": {
                "semantics": (
                    "T_src_tgt = inv(T_base_src) @ T_base_tgt; maps target "
                    "camera points into the source camera frame"
                ),
                **_error_record(src_tgt_translations, src_tgt_rotations),
                "per_target_camera": per_target,
            },
            "T_tgt_src": {
                "semantics": (
                    "T_tgt_src = inv(T_base_tgt) @ T_base_src; maps source "
                    "camera points into the target camera frame"
                ),
                **_error_record(tgt_src_translations, tgt_src_rotations),
            },
        },
    }


def export_depth_model_compatibility(
    dataset_root: Path,
    final_calibration: Mapping[str, Any],
    calibration_path: Path,
    output_root: Path,
    *,
    evaluate_gt: bool,
    max_frames: int | None,
    frame_stride: int,
    camera_name: str | None,
    write_jsonl: bool,
    write_per_frame_json: bool,
    write_sample_manifest: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if write_sample_manifest and not write_per_frame_json:
        raise ValueError("Sample manifest requires --write-per-frame-json")
    cameras = list(final_calibration.get("cameras", []))
    if camera_name is not None:
        cameras = [camera for camera in cameras if camera["camera_name"] == camera_name]
        if not cameras:
            raise ValueError(f"Camera {camera_name!r} is not in final calibration")
    if not cameras:
        raise ValueError("Final calibration contains no selected cameras")
    camera_names = [str(camera["camera_name"]) for camera in cameras]
    frame_ids = _selected_common_frame_ids(
        dataset_root, camera_names, max_frames, frame_stride
    )
    if not frame_ids:
        raise ValueError("No aligned RGB/link-pose frames were found")
    adapter = final_calibration["frame_adapters"][
        "camera_ray_to_camera_pose_adapter"
    ]
    adapter_matrix = np.asarray(adapter["matrix_rowmajor_3x3"], dtype=float).reshape(
        3, 3
    )
    provider = UnityLinkPoseProvider(dataset_root)
    records_by_camera: dict[str, list[dict[str, Any]]] = {
        name: [] for name in camera_names
    }
    transforms_by_camera: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in camera_names
    }
    total_transform_jsons = 0
    for frame_id in frame_ids:
        link_poses = provider.get_candidate_link_poses(frame_id)
        for camera in cameras:
            name = str(camera["camera_name"])
            attached_link = str(camera["attached_link"])
            if attached_link not in link_poses:
                raise KeyError(f"Attached link {attached_link} missing at {frame_id}")
            record = make_depth_model_transform_record(
                frame_id, camera, link_poses[attached_link], adapter_matrix
            )
            parsed = parse_depth_model_transform(record)
            records_by_camera[name].append(record)
            transforms_by_camera[name][frame_id] = parsed
            if write_per_frame_json:
                destination = output_root / "transforms" / name / f"{frame_id}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(record, indent=2) + "\n", encoding="utf-8"
                )
                total_transform_jsons += 1
    if write_jsonl:
        jsonl_root = output_root / "camera_poses_base"
        jsonl_root.mkdir(parents=True, exist_ok=True)
        for name, records in records_by_camera.items():
            with (jsonl_root / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    samples = []
    missing_depth_count = 0
    if write_sample_manifest:
        samples, missing_depth_count = make_sample_manifest(
            dataset_root, output_root, camera_names, frame_ids
        )
        (output_root / "depth_model_samples.json").write_text(
            json.dumps(samples, indent=2) + "\n", encoding="utf-8"
        )
    parser_smoke_passed = all(
        np.all(np.isfinite(parse_depth_model_transform(record)))
        for records in records_by_camera.values()
        for record in records
    )
    warnings = []
    if missing_depth_count:
        warnings.append(
            f"Depth files were absent for {missing_depth_count} manifest views; "
            "depth_path was set to null."
        )
    if len(camera_names) >= 2:
        first_frame = frame_ids[0]
        target = transforms_by_camera[camera_names[0]][first_frame]
        source = transforms_by_camera[camera_names[1]][first_frame]
        src_tgt, tgt_src = compute_relative_camera_transforms(target, source)
        relative_smoke_passed: bool | None = bool(
            # Recorder matrices are serialized with finite precision; allow the
            # sub-micro-unit orthogonality residual observed after rigid inversion.
            np.allclose(src_tgt @ tgt_src, np.eye(4), atol=1e-6)
        )
    else:
        relative_smoke_passed = None
        warnings.append("Relative-pose smoke test skipped because one camera was selected.")

    validation = None
    if evaluate_gt:
        gt_by_camera = {
            name: load_camera_gt_transforms(dataset_root, name, frame_ids)
            for name in camera_names
        }
        validation = evaluate_depth_model_poses(
            transforms_by_camera, gt_by_camera, frame_ids
        )
        if not validation["gt_validation_available"]:
            warnings.append("Unity GT camera transforms were unavailable; validation skipped.")
            validation = None
    report = {
        "schema_version": 1,
        "final_calibration_path": _display_path(calibration_path),
        "camera_count": len(camera_names),
        "frames_exported": len(frame_ids),
        "total_transform_json_files_generated": total_transform_jsons,
        "all_transforms_contain_T_base_cam_rowmajor": all(
            "T_base_cam_rowmajor" in record
            for records in records_by_camera.values()
            for record in records
        ),
        "all_transforms_contain_t_and_q": all(
            "t_base_cam" in record and "q_base_cam_xyzw" in record
            for records in records_by_camera.values()
            for record in records
        ),
        "parser_compatibility_smoke_test_passed": parser_smoke_passed,
        "relative_pose_computation_smoke_test_passed": relative_smoke_passed,
        "per_camera_exported_frame_counts": {
            name: len(records) for name, records in records_by_camera.items()
        },
        "sample_manifest_written": write_sample_manifest,
        "sample_manifest_sample_count": len(samples),
        "jsonl_mirror_written": write_jsonl,
        "per_frame_json_written": write_per_frame_json,
        "adapter_metadata_summary": {
            "name": adapter["name"],
            "from_frame": adapter["from_frame"],
            "to_frame": adapter["to_frame"],
            "matrix_rowmajor_3x3": adapter_matrix.reshape(-1).tolist(),
            "already_applied_during_calibration": True,
        },
        "ground_truth_used_to_generate_transforms": False,
        "gt_validation_report_written": validation is not None,
        "warnings": warnings,
    }
    return report, validation
