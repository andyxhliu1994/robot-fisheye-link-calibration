"""Recover motion-limited camera mounts from a shared fixed board.

This module consumes only independent link-calibration results, estimated
camera-to-board poses, and candidate-link kinematics. Ground truth is excluded
from classification, estimation, recovery, and confidence assignment.
"""

from __future__ import annotations

from copy import deepcopy
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .link_calibrator import (
    BoardPoseObservation,
    average_transforms,
    parameters_from_transform,
    transform_from_parameters,
)
from .se3_utils import invert_T, rotation_error_deg, t_q_from_mat


MAX_INDEPENDENT_SCORE = 0.1
MIN_LINK_SCORE_MARGIN = 0.02
MAX_BOUNDED_MOUNT_TRANSLATION_M = 5.0
GOOD_TRANSLATION_CONSISTENCY_M = 0.05
GOOD_ROTATION_CONSISTENCY_DEG = 3.0


def _transform_from_record(record: Mapping[str, Any], key: str) -> np.ndarray:
    values = np.asarray(record.get(key), dtype=float)
    if values.size != 16 or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain 16 finite values")
    transform = values.reshape(4, 4)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{key} must be a homogeneous transform")
    return transform


def _is_bounded_mount(record: Mapping[str, Any]) -> bool:
    try:
        transform = _transform_from_record(record, "T_link_camera_rowmajor")
    except (TypeError, ValueError):
        return False
    return float(np.linalg.norm(transform[:3, 3])) <= MAX_BOUNDED_MOUNT_TRANSLATION_M


def _has_strong_link_association(record: Mapping[str, Any]) -> bool:
    score = record.get("best_score")
    margin = record.get("score_margin")
    return (
        bool(record.get("success"))
        and record.get("best_link_path_rel") is not None
        and score is not None
        and float(score) <= MAX_INDEPENDENT_SCORE
        and margin is not None
        and float(margin) >= MIN_LINK_SCORE_MARGIN
    )


def classify_camera_records(
    camera_records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Classify records by observability diagnostics, never by link name."""
    anchors = []
    motion_limited = []
    excluded = []
    for record in camera_records:
        rank = record.get("observability_rank")
        maximum_rank = record.get("observability_parameter_count")
        note = str(record.get("mount_estimation_note") or "").lower()
        fully_observable = (
            record.get("mount_fully_observable") is True
            and rank is not None
            and maximum_rank is not None
            and int(rank) >= int(maximum_rank)
            and "gauge" not in note
            and "motion" not in note
        )
        low_observability = (
            record.get("mount_fully_observable") is False
            or (
                rank is not None
                and maximum_rank is not None
                and int(rank) < int(maximum_rank)
            )
            or "gauge" in note
            or "motion" in note
        )
        strong_association = _has_strong_link_association(record)
        if fully_observable and strong_association and _is_bounded_mount(record):
            anchors.append(record)
        elif low_observability and strong_association:
            motion_limited.append(record)
        else:
            excluded.append(record)
    return anchors, motion_limited, excluded


def _mean_residuals(parameters: np.ndarray, transforms: Sequence[np.ndarray]) -> np.ndarray:
    mean_transform = transform_from_parameters(parameters)
    residuals = []
    for transform in transforms:
        delta = invert_T(mean_transform) @ transform
        residuals.extend(delta[:3, 3])
        residuals.extend(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
    return np.asarray(residuals, dtype=float)


def robust_average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Robustly average rigid transforms using an SE(3)-style residual."""
    values = [np.asarray(transform, dtype=float).reshape(4, 4) for transform in transforms]
    if not values:
        raise ValueError("At least one transform is required")
    initial = parameters_from_transform(average_transforms(values))
    optimization = least_squares(
        _mean_residuals,
        initial,
        args=(values,),
        method="trf",
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=400,
    )
    if not optimization.success or not np.all(np.isfinite(optimization.x)):
        raise ValueError("Robust transform average did not converge")
    return transform_from_parameters(optimization.x)


def transform_consistency_metrics(
    transforms: Sequence[np.ndarray], reference: np.ndarray
) -> dict[str, dict[str, float]]:
    translations = [
        float(np.linalg.norm(transform[:3, 3] - reference[:3, 3]))
        for transform in transforms
    ]
    rotations = [
        rotation_error_deg(transform[:3, :3], reference[:3, :3])
        for transform in transforms
    ]
    if not translations:
        raise ValueError("At least one transform is required for consistency metrics")
    return {
        "translation_m": {
            "mean": fmean(translations),
            "median": median(translations),
            "max": max(translations),
        },
        "rotation_deg": {
            "mean": fmean(rotations),
            "median": median(rotations),
            "max": max(rotations),
        },
    }


def _transform_fields(name: str, transform: np.ndarray) -> dict[str, Any]:
    translation, quaternion = t_q_from_mat(transform)
    return {
        f"T_{name}_rowmajor": transform.reshape(-1).tolist(),
        f"t_{name}": translation.tolist(),
        f"q_{name}_xyzw": quaternion.tolist(),
    }


def estimate_shared_board(
    anchor_records: Sequence[Mapping[str, Any]],
    observations_by_camera: Mapping[str, Sequence[BoardPoseObservation]],
    link_poses_by_camera: Mapping[str, Sequence[np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = []
    contribution_counts: dict[str, int] = {}
    for record in anchor_records:
        camera_name = str(record["camera_name"])
        observations = list(observations_by_camera.get(camera_name, []))
        link_poses = list(link_poses_by_camera.get(camera_name, []))
        if len(observations) != len(link_poses):
            raise ValueError(f"Pose count mismatch for anchor {camera_name}")
        mount = _transform_from_record(record, "T_link_camera_rowmajor")
        camera_candidates = [
            link @ mount @ observation.T_camera_board
            for link, observation in zip(link_poses, observations)
        ]
        contribution_counts[camera_name] = len(camera_candidates)
        candidates.extend(camera_candidates)
    if not candidates:
        raise ValueError("Anchor cameras supplied no valid board-pose observations")
    shared = robust_average_transforms(candidates)
    metrics = transform_consistency_metrics(candidates, shared)
    return shared, {
        "sample_count": len(candidates),
        "per_anchor_contribution_counts": contribution_counts,
        "outlier_rejection_count": 0,
        **metrics,
    }


def recover_camera_mount(
    record: Mapping[str, Any],
    observations: Sequence[BoardPoseObservation],
    link_poses: Sequence[np.ndarray],
    shared_board: np.ndarray,
    *,
    anchor_camera_count: int,
    single_anchor: bool,
    anchor_agreement: Mapping[str, Any],
) -> dict[str, Any]:
    camera_name = str(record["camera_name"])
    if len(observations) != len(link_poses):
        raise ValueError(f"Pose count mismatch for motion-limited camera {camera_name}")
    candidates = [
        invert_T(link) @ shared_board @ invert_T(observation.T_camera_board)
        for link, observation in zip(link_poses, observations)
    ]
    if not candidates:
        raise ValueError(f"No valid board-pose observations for {camera_name}")
    recovered = robust_average_transforms(candidates)
    metrics = transform_consistency_metrics(candidates, recovered)
    agreement_good = (
        float(anchor_agreement["translation_m"]["mean"])
        <= GOOD_TRANSLATION_CONSISTENCY_M
        and float(anchor_agreement["rotation_deg"]["mean"])
        <= GOOD_ROTATION_CONSISTENCY_DEG
    )
    recovery_good = (
        metrics["translation_m"]["mean"] <= GOOD_TRANSLATION_CONSISTENCY_M
        and metrics["rotation_deg"]["mean"] <= GOOD_ROTATION_CONSISTENCY_DEG
    )
    if agreement_good and recovery_good:
        confidence = "medium" if single_anchor else "high"
    else:
        confidence = "low"
    if single_anchor:
        warning = "Single-anchor recovery cannot cross-validate the shared board pose."
    elif confidence == "low":
        warning = "Shared-board or recovered-mount residuals exceed consistency thresholds."
    else:
        warning = None
    result = {
        "camera_name": camera_name,
        "classification": "motion_limited",
        "best_link": record.get("best_link"),
        "best_link_path_rel": record.get("best_link_path_rel"),
        "original_independent_observability_rank": record.get("observability_rank"),
        "original_independent_observability_max_rank": record.get(
            "observability_parameter_count"
        ),
        "original_independent_result": deepcopy(dict(record)),
        "recovery_used": True,
        "recovery_method": "shared_fixed_board",
        "anchor_camera_count": anchor_camera_count,
        "confidence": confidence,
        **_transform_fields("link_camera_recovered", recovered),
        "recovery_consistency_translation_m": metrics["translation_m"],
        "recovery_consistency_rotation_deg": metrics["rotation_deg"],
        "num_valid_frames": len(candidates),
        "num_rejected_frames": 0,
        "warning": warning,
        "gt_evaluation_available": False,
        "gt_recovered_translation_error_m": None,
        "gt_recovered_rotation_error_deg": None,
    }
    return result


def _unrecovered_camera_record(
    record: Mapping[str, Any], warning: str, anchor_count: int
) -> dict[str, Any]:
    return {
        "camera_name": str(record["camera_name"]),
        "classification": "motion_limited",
        "best_link": record.get("best_link"),
        "best_link_path_rel": record.get("best_link_path_rel"),
        "original_independent_observability_rank": record.get("observability_rank"),
        "original_independent_observability_max_rank": record.get(
            "observability_parameter_count"
        ),
        "original_independent_result": deepcopy(dict(record)),
        "recovery_used": False,
        "recovery_method": None,
        "anchor_camera_count": anchor_count,
        "confidence": "low",
        "T_link_camera_recovered_rowmajor": None,
        "t_link_camera_recovered": None,
        "q_link_camera_recovered_xyzw": None,
        "recovery_consistency_translation_m": None,
        "recovery_consistency_rotation_deg": None,
        "num_valid_frames": 0,
        "num_rejected_frames": 0,
        "warning": warning,
        "gt_evaluation_available": False,
        "gt_recovered_translation_error_m": None,
        "gt_recovered_rotation_error_deg": None,
    }


def perform_shared_board_recovery(
    link_calibration_summary: Mapping[str, Any],
    observations_by_camera: Mapping[str, Sequence[BoardPoseObservation]],
    link_poses_by_camera: Mapping[str, Sequence[np.ndarray]],
    *,
    input_summary_path: str,
    min_anchor_cameras: int = 2,
    allow_single_anchor: bool = False,
    target_camera: str | None = None,
) -> dict[str, Any]:
    records = list(link_calibration_summary.get("cameras", []))
    anchors, motion_limited, excluded = classify_camera_records(records)
    if target_camera is not None:
        motion_limited = [
            record
            for record in motion_limited
            if str(record.get("camera_name")) == target_camera
        ]
    anchor_names = [str(record["camera_name"]) for record in anchors]
    motion_names = [str(record["camera_name"]) for record in motion_limited]
    excluded_names = [str(record["camera_name"]) for record in excluded]
    classifications = {
        **{str(record["camera_name"]): "anchor" for record in anchors},
        **{
            str(record["camera_name"]): "motion_limited"
            for record in motion_limited
        },
        **{str(record["camera_name"]): "excluded" for record in excluded},
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "shared_board_recovery",
        "input_link_calibration_summary": input_summary_path,
        "ground_truth_used_for_estimation": False,
        "anchor_cameras": anchor_names,
        "motion_limited_cameras": motion_names,
        "excluded_cameras": excluded_names,
        "camera_classifications": [
            {
                "camera_name": str(record["camera_name"]),
                "classification": classifications.get(
                    str(record["camera_name"]), "not_targeted"
                ),
                "original_independent_result": deepcopy(dict(record)),
            }
            for record in records
        ],
        "min_anchor_cameras": min_anchor_cameras,
        "allow_single_anchor": allow_single_anchor,
        "classification_thresholds": {
            "max_independent_score": MAX_INDEPENDENT_SCORE,
            "min_link_score_margin": MIN_LINK_SCORE_MARGIN,
            "max_bounded_mount_translation_m": MAX_BOUNDED_MOUNT_TRANSLATION_M,
            "good_translation_consistency_m": GOOD_TRANSLATION_CONSISTENCY_M,
            "good_rotation_consistency_deg": GOOD_ROTATION_CONSISTENCY_DEG,
        },
        "status": None,
        "no_recovery_needed": False,
        "shared_board_estimated": False,
        "T_base_board_shared_rowmajor": None,
        "t_base_board_shared": None,
        "q_base_board_shared_xyzw": None,
        "anchor_agreement": None,
        "camera_results": [],
        "warnings": [],
    }
    if not motion_limited:
        summary["status"] = "no_recovery_needed"
        summary["no_recovery_needed"] = True
        return summary

    anchor_count = len(anchors)
    single_anchor = anchor_count == 1
    enough_anchors = anchor_count >= min_anchor_cameras
    single_anchor_allowed = single_anchor and allow_single_anchor
    if not enough_anchors and not single_anchor_allowed:
        if anchor_count == 0:
            warning = "No anchor cameras are available; full camera mounts are not observable."
        else:
            warning = (
                f"Only {anchor_count} anchor camera is available; enable "
                "--allow-single-anchor to permit medium-confidence recovery."
            )
        summary["status"] = "insufficient_anchor"
        summary["warnings"].append(warning)
        summary["camera_results"] = [
            _unrecovered_camera_record(record, warning, anchor_count)
            for record in motion_limited
        ]
        return summary

    shared_board, agreement = estimate_shared_board(
        anchors, observations_by_camera, link_poses_by_camera
    )
    summary["status"] = "recovered"
    summary["shared_board_estimated"] = True
    summary.update(_transform_fields("base_board_shared", shared_board))
    summary["anchor_agreement"] = agreement
    if single_anchor:
        summary["warnings"].append(
            "Single-anchor recovery cannot cross-validate the shared board pose."
        )
    summary["camera_results"] = [
        recover_camera_mount(
            record,
            observations_by_camera.get(str(record["camera_name"]), []),
            link_poses_by_camera.get(str(record["camera_name"]), []),
            shared_board,
            anchor_camera_count=anchor_count,
            single_anchor=single_anchor,
            anchor_agreement=agreement,
        )
        for record in motion_limited
    ]
    return summary
