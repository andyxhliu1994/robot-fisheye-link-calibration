"""Final static calibration selection and depth-model pose export.

Transform names follow ``T_A_B`` throughout: ``p_A = T_A_B @ p_B``.
Ground truth is not accepted by the static-calibration selection functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import numpy as np

from .camera_frame_adapter import camera_frame_adapter_from_config
from .dataset_loader import load_json_document
from .kinematics_provider import UnityLinkPoseProvider
from .se3_utils import mat_from_t_q, rotation_error_deg, t_q_from_mat


CAMERA_FRAME_CONVENTION = {
    "handedness": "right",
    "x": "right",
    "y": "up",
    "z": "forward",
}
MAX_VALIDATION_TRANSLATION_MEAN_M = 0.05
MAX_VALIDATION_ROTATION_MEAN_DEG = 3.0


def _as_transform(values: Any, field_name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.size != 16 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must contain 16 finite values")
    transform = matrix.reshape(4, 4)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{field_name} must be a homogeneous transform")
    return transform


def _transform_fields(transform: np.ndarray) -> dict[str, Any]:
    translation, quaternion = t_q_from_mat(transform)
    return {
        "T_link_camera_rowmajor": transform.reshape(-1).tolist(),
        "t_link_camera": translation.tolist(),
        "q_link_camera_xyzw": quaternion.tolist(),
    }


def _display_path(path: str | Path) -> str:
    value = str(path)
    if not Path(value).is_absolute() and not value.startswith("./"):
        return f"./{value}"
    return value


def build_frame_metadata(dataset_root: Path) -> dict[str, Any]:
    camera_config, _ = load_json_document(dataset_root / "camera_model_config.json")
    board_config, _ = load_json_document(dataset_root / "charuco_board_config.json")
    candidate_links, _ = load_json_document(dataset_root / "candidate_links.json")
    adapter_config = camera_config["ray_frame_adapter"]
    adapter = camera_frame_adapter_from_config(adapter_config)
    raw_ray_frame = str(camera_config.get("ray_frame", "camera_model_ray_frame"))
    configured_pose_frame = str(
        camera_config.get(
            "camera_pose_frame", camera_config.get("pose_camera_frame", "camera_pose")
        )
    )
    calibration_file = str(camera_config["default_calibration_file"])
    inner_corner_count = (int(board_config["squares_x"]) - 1) * (
        int(board_config["squares_y"]) - 1
    )
    return {
        "camera_ray_to_camera_pose_adapter": {
            "name": adapter.name,
            "type": str(adapter_config.get("type", "identity")),
            "from_frame": raw_ray_frame,
            "to_frame": configured_pose_frame,
            "matrix_rowmajor_3x3": adapter.matrix.reshape(-1).tolist(),
            "semantics": (
                "ray_camera_pose_frame = R_adapter @ "
                "ray_raw_camera_model_frame"
            ),
            "origin": "camera projection center; the adapter changes axes only",
            "input_units": "unit direction vector",
            "output_units": "unit direction vector",
            "source": "./dataset/camera_model_config.json",
            "affects_T_camera_board": True,
            "affects_final_T_link_camera": True,
            "application_note": (
                "Already applied during T_camera_board estimation; do not apply "
                "again to T_link_camera."
            ),
        },
        "camera_pose_frame": {
            "name": "unity_camera_pose_frame",
            "configured_name": configured_pose_frame,
            "used_for": "camera frame of T_camera_board and T_link_camera",
            "definition": {
                "origin": "camera projection center / Unity camera Transform origin",
                "handedness": "right",
                "x_axis": "camera image right",
                "y_axis": "camera image up",
                "z_axis": "camera forward",
                "units": "meters for positions; unitless orthonormal rotation",
            },
            "source": (
                "./dataset/camera_model_config.json and dataset camera transform "
                "convention"
            ),
            "affects_final_T_link_camera": True,
        },
        "camera_model": {
            "name": str(camera_config["default_camera_model"]),
            "calibration_file": f"./dataset/{calibration_file}",
            "ray_output_frame": raw_ray_frame,
            "semantics": "pixel coordinates are mapped to unit camera rays",
            "origin": "camera projection center",
            "units": "pixels at input; unit direction vector at output",
            "source": "./dataset/camera_model_config.json",
            "affects_T_camera_board": True,
            "affects_final_T_link_camera": True,
        },
        "board_pose_frame": {
            "name": "opencv_charuco_board_frame",
            "used_for": "T_camera_board estimation",
            "semantics": "p_camera = T_camera_board @ p_charuco_board",
            "definition": {
                "origin": (
                    "outer corner of the ChArUco board object-point frame used "
                    "by OpenCV and charuco_config"
                ),
                "x_axis": "increasing board column / square x index",
                "y_axis": "increasing board row / square y index",
                "z_axis": "board normal from the right-hand rule",
                "units": "meters",
                "board_corners_z": 0.0,
            },
            "board_geometry": {
                "dictionary": str(board_config["dictionary"]),
                "squares_x": int(board_config["squares_x"]),
                "squares_y": int(board_config["squares_y"]),
                "square_length_m": float(board_config["square_length_m"]),
                "marker_length_m": float(board_config["marker_length_m"]),
                "charuco_inner_corners": inner_corner_count,
            },
            "source": "./dataset/charuco_board_config.json and OpenCV ChArUco convention",
            "affects_T_camera_board": True,
            "affects_final_T_link_camera": False,
            "note": (
                "Changing this board object frame changes T_camera_board and the "
                "shared T_base_board, but not final T_link_camera when used consistently."
            ),
        },
        "link_pose_frame": {
            "name": "unity_robot_link_frame",
            "used_for": "T_base_link input",
            "semantics": "p_base = T_base_link @ p_link",
            "definition": {
                "base_frame": str(candidate_links.get("base_frame_name", "base")),
                "link_frame": "Unity Transform frame exported for each candidate link",
                "origin": "the exported Unity Transform origin of each robot link",
                "x_axis": "exported link local +X axis",
                "y_axis": "exported link local +Y axis",
                "z_axis": "exported link local +Z axis",
                "units": "meters",
                "rotation": "right-handed 3x3 orthonormal matrix",
            },
            "source": "./dataset/link_poses/*.json or ./dataset/link_poses.csv",
            "affects_final_T_link_camera": True,
        },
        "board_gt_adapter": None,
        "board_gt_adapter_note": (
            "Milestone 5 does not perform board-frame GT evaluation, so no board "
            "GT adapter is applied or recorded."
        ),
    }


def select_final_camera_calibrations(
    link_calibration_summary: Mapping[str, Any],
    shared_board_recovery_summary: Mapping[str, Any],
    candidate_links: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select final mounts without accepting or consulting ground truth."""
    link_names = {
        str(candidate["link_path_rel"]): str(candidate["link_name"])
        for candidate in candidate_links.get("links", [])
    }
    recovered_by_camera = {
        str(result["camera_name"]): result
        for result in shared_board_recovery_summary.get("camera_results", [])
        if result.get("recovery_used")
        and result.get("confidence") in {"high", "medium"}
        and result.get("T_link_camera_recovered_rowmajor") is not None
    }
    final_cameras = []
    for independent in link_calibration_summary.get("cameras", []):
        camera_name = str(independent["camera_name"])
        attached_link = str(independent.get("best_link_path_rel") or "")
        if attached_link not in link_names:
            raise ValueError(f"Unknown attached link for {camera_name}: {attached_link}")
        rank = independent.get("observability_rank")
        max_rank = independent.get("observability_parameter_count")
        motion_limited = not bool(independent.get("mount_fully_observable")) or (
            rank is not None and max_rank is not None and int(rank) < int(max_rank)
        )
        warnings = []
        recovery = recovered_by_camera.get(camera_name)
        if recovery is not None:
            transform = _as_transform(
                recovery["T_link_camera_recovered_rowmajor"],
                "T_link_camera_recovered_rowmajor",
            )
            source = "shared_board_recovery"
            confidence = str(recovery["confidence"])
            if recovery.get("warning"):
                warnings.append(str(recovery["warning"]))
        else:
            transform = _as_transform(
                independent.get("T_link_camera_rowmajor"),
                "T_link_camera_rowmajor",
            )
            source = "independent_link_calibration"
            if motion_limited:
                confidence = "low"
                warnings.append(
                    "Full T_link_camera is not fully observable from this dataset; "
                    "no trusted shared-board recovery was available."
                )
            else:
                confidence = "high"
        final_cameras.append(
            {
                "camera_name": camera_name,
                "attached_link": attached_link,
                "attached_link_name": link_names[attached_link],
                **_transform_fields(transform),
                "transform_convention": {
                    "T_link_camera": "p_link = T_link_camera @ p_camera",
                    "runtime_pose": "T_base_cam = T_base_link @ T_link_camera",
                },
                "camera_frame_convention": dict(CAMERA_FRAME_CONVENTION),
                "calibration_source": source,
                "confidence": confidence,
                "warnings": warnings,
                "observability": {
                    "rank": rank,
                    "max_rank": max_rank,
                    "motion_limited": motion_limited,
                    "independent_note": independent.get("mount_estimation_note"),
                },
                "selection_diagnostics": {
                    "independent_best_score": independent.get("best_score"),
                    "independent_score_margin": independent.get("score_margin"),
                    "shared_board_recovery_used": recovery is not None,
                },
            }
        )
    return final_cameras


def build_final_calibration(
    dataset_root: Path,
    link_calibration_summary: Mapping[str, Any],
    shared_board_recovery_summary: Mapping[str, Any],
    *,
    link_calibration_path: str | Path,
    shared_board_recovery_path: str | Path,
) -> dict[str, Any]:
    candidate_links, _ = load_json_document(dataset_root / "candidate_links.json")
    cameras = select_final_camera_calibrations(
        link_calibration_summary, shared_board_recovery_summary, candidate_links
    )
    return {
        "calibration_version": "milestone_5_final_static_calibration",
        "created_from": {
            "link_calibration_summary": _display_path(link_calibration_path),
            "shared_board_recovery_summary": _display_path(
                shared_board_recovery_path
            ),
        },
        "ground_truth_used_for_selection": False,
        "pose_semantics": {
            "T_A_B": "p_A = T_A_B @ p_B",
            "T_base_cam": "p_base = T_base_cam @ p_camera",
            "T_link_camera": "p_link = T_link_camera @ p_camera",
            "runtime_composition": "T_base_cam = T_base_link @ T_link_camera",
        },
        "frame_adapters": build_frame_metadata(dataset_root),
        "validation": {
            "gt_validation_available": False,
            "evaluation_only": True,
            "static_validation_report": None,
            "camera_pose_validation_report": None,
        },
        "camera_count": len(cameras),
        "cameras": cameras,
    }


def compose_base_camera(T_base_link: np.ndarray, T_link_camera: np.ndarray) -> np.ndarray:
    """Return T_base_cam where p_base = T_base_cam @ p_camera."""
    return np.asarray(T_base_link, dtype=float) @ np.asarray(T_link_camera, dtype=float)


def make_base_camera_record(
    frame_id: str,
    camera: Mapping[str, Any],
    T_base_link: np.ndarray,
) -> dict[str, Any]:
    mount = _as_transform(camera["T_link_camera_rowmajor"], "T_link_camera_rowmajor")
    T_base_cam = compose_base_camera(T_base_link, mount)
    translation, quaternion = t_q_from_mat(T_base_cam)
    return {
        "frame_id": frame_id,
        "camera_name": str(camera["camera_name"]),
        "attached_link": str(camera["attached_link"]),
        "T_base_cam_rowmajor": T_base_cam.reshape(-1).tolist(),
        "t_base_cam": translation.tolist(),
        "q_base_cam_xyzw": quaternion.tolist(),
        "camera_frame_convention": dict(CAMERA_FRAME_CONVENTION),
        "source": "T_base_link @ final T_link_camera",
        "transform_convention": {
            "T_base_cam": "p_base = T_base_cam @ p_camera"
        },
    }


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {"mean": fmean(values), "median": median(values), "max": max(values)}


def evaluate_base_camera_predictions(
    predicted_records: Sequence[Mapping[str, Any]],
    gt_by_frame: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    translation_errors = []
    rotation_errors = []
    for record in predicted_records:
        frame_id = str(record["frame_id"])
        if frame_id not in gt_by_frame:
            continue
        predicted = _as_transform(record["T_base_cam_rowmajor"], "T_base_cam_rowmajor")
        ground_truth = np.asarray(gt_by_frame[frame_id], dtype=float).reshape(4, 4)
        translation_errors.append(
            float(np.linalg.norm(predicted[:3, 3] - ground_truth[:3, 3]))
        )
        rotation_errors.append(
            rotation_error_deg(predicted[:3, :3], ground_truth[:3, :3])
        )
    if not translation_errors:
        return {
            "gt_validation_available": False,
            "frames_evaluated": 0,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "passed": False,
            "reason": "no_matching_gt_frames",
        }
    translation = _metric_summary(translation_errors)
    rotation = _metric_summary(rotation_errors)
    passed = (
        translation["mean"] <= MAX_VALIDATION_TRANSLATION_MEAN_M
        and rotation["mean"] <= MAX_VALIDATION_ROTATION_MEAN_DEG
    )
    return {
        "gt_validation_available": True,
        "frames_evaluated": len(translation_errors),
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "thresholds": {
            "max_translation_mean_m": MAX_VALIDATION_TRANSLATION_MEAN_M,
            "max_rotation_mean_deg": MAX_VALIDATION_ROTATION_MEAN_DEG,
        },
        "passed": passed,
        "reason": "ok" if passed else "mean_error_threshold_exceeded",
    }


def load_camera_gt_transforms(
    dataset_root: Path, camera_name: str, frame_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    transforms = {}
    transform_dir = dataset_root / "cameras" / camera_name / "transform"
    for frame_id in frame_ids:
        path = transform_dir / f"{frame_id}.json"
        if not path.is_file():
            continue
        document, _ = load_json_document(path)
        if document.get("T_base_cam_rowmajor") is not None:
            transform = _as_transform(
                document["T_base_cam_rowmajor"], "T_base_cam_rowmajor"
            )
        else:
            transform = mat_from_t_q(
                np.asarray(document["t_base_cam"], dtype=float),
                np.asarray(document["q_base_cam_xyzw"], dtype=float),
            )
        transforms[frame_id] = transform
    return transforms


def export_base_camera_pose_jsonls(
    dataset_root: Path,
    final_calibration: Mapping[str, Any],
    output_dir: Path,
    *,
    evaluate_gt: bool,
) -> dict[str, Any]:
    link_pose_paths = sorted((dataset_root / "link_poses").glob("frame_*.json"))
    frame_ids = [path.stem for path in link_pose_paths]
    if not frame_ids:
        raise ValueError("No dataset link-pose JSON files were found")
    provider = UnityLinkPoseProvider(dataset_root)
    records_by_camera = {
        str(camera["camera_name"]): [] for camera in final_calibration["cameras"]
    }
    for frame_id in frame_ids:
        link_poses = provider.get_candidate_link_poses(frame_id)
        for camera in final_calibration["cameras"]:
            camera_name = str(camera["camera_name"])
            attached_link = str(camera["attached_link"])
            if attached_link not in link_poses:
                raise KeyError(f"Attached link {attached_link} missing at {frame_id}")
            records_by_camera[camera_name].append(
                make_base_camera_record(frame_id, camera, link_poses[attached_link])
            )
    pose_dir = output_dir / "camera_poses_base"
    pose_dir.mkdir(parents=True, exist_ok=True)
    validation_records = []
    for camera in final_calibration["cameras"]:
        camera_name = str(camera["camera_name"])
        records = records_by_camera[camera_name]
        with (pose_dir / f"{camera_name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        if evaluate_gt:
            gt = load_camera_gt_transforms(dataset_root, camera_name, frame_ids)
            metrics = evaluate_base_camera_predictions(records, gt)
        else:
            metrics = {
                "gt_validation_available": False,
                "frames_evaluated": 0,
                "translation_error_m": None,
                "rotation_error_deg": None,
                "passed": None,
                "reason": "evaluation_disabled",
            }
        validation_records.append({"camera_name": camera_name, **metrics})
    evaluated = [
        record for record in validation_records if record["gt_validation_available"]
    ]
    return {
        "schema_version": 1,
        "pose_semantics": "p_base = T_base_cam @ p_camera",
        "runtime_composition": "T_base_cam = T_base_link @ T_link_camera",
        "ground_truth_used_to_modify_calibration": False,
        "frame_count": len(frame_ids),
        "camera_count": len(records_by_camera),
        "gt_validation_available": bool(evaluated),
        "passed": bool(evaluated) and all(record["passed"] for record in evaluated),
        "cameras": validation_records,
    }
