"""Link association and static T_link_camera estimation.

The estimation functions in this module consume only camera-to-board estimates
and candidate base-to-link kinematics. Ground truth is intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .se3_utils import invert_T, mat_from_t_q, rotation_error_deg, t_q_from_mat


ROTATION_LENGTH_SCALE_M = 0.25
CANONICAL_GAUGE_STRENGTH = 1e-3


@dataclass(frozen=True)
class BoardPoseObservation:
    frame_id: str
    T_camera_board: np.ndarray
    charuco_corner_count: int = 0
    mean_ray_error_deg: float | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_camera_board, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_camera_board must be a finite 4x4 transform")
        object.__setattr__(self, "T_camera_board", transform)


def transform_from_parameters(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (6,):
        raise ValueError("SE(3) parameters must be [rotation_vector, translation]")
    return mat_from_t_q(values[3:], Rotation.from_rotvec(values[:3]).as_quat())


def parameters_from_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=float).reshape(4, 4)
    return np.concatenate(
        [Rotation.from_matrix(value[:3, :3]).as_rotvec(), value[:3, 3]]
    )


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("At least one transform is required")
    values = [np.asarray(transform, dtype=float).reshape(4, 4) for transform in transforms]
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_matrix(
        np.asarray([value[:3, :3] for value in values])
    ).mean().as_matrix()
    result[:3, 3] = np.mean([value[:3, 3] for value in values], axis=0)
    return result


def board_consistency_residuals(
    parameters: np.ndarray,
    T_base_links: Sequence[np.ndarray],
    T_camera_boards: Sequence[np.ndarray],
) -> np.ndarray:
    """Joint residual for unknown mount X and fixed base-to-board transform."""
    values = np.asarray(parameters, dtype=float)
    if values.shape != (12,):
        raise ValueError("Joint parameters must contain mount and board SE(3) values")
    if len(T_base_links) != len(T_camera_boards):
        raise ValueError("Link and board pose sequences must have equal length")
    T_link_camera = transform_from_parameters(values[:6])
    T_base_board_mean = transform_from_parameters(values[6:])
    residuals = []
    for T_base_link, T_camera_board in zip(T_base_links, T_camera_boards):
        predicted = (
            np.asarray(T_base_link, dtype=float)
            @ T_link_camera
            @ np.asarray(T_camera_board, dtype=float)
        )
        delta = invert_T(T_base_board_mean) @ predicted
        residuals.extend(delta[:3, 3])
        residuals.extend(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
    return np.asarray(residuals, dtype=float)


def _gauge_stabilized_residuals(
    parameters: np.ndarray,
    T_base_links: Sequence[np.ndarray],
    T_camera_boards: Sequence[np.ndarray],
) -> np.ndarray:
    """Add a negligible minimum-norm gauge choice for unobservable mounts.

    A link undergoing only one-axis motion cannot determine mount translation
    along, or mount rotation about, that axis.  The weak prior prevents those
    null-space components from diverging without materially affecting the
    board-consistency fit in observable directions.
    """
    data_residuals = board_consistency_residuals(
        parameters, T_base_links, T_camera_boards
    )
    mount_prior = CANONICAL_GAUGE_STRENGTH * np.asarray(parameters[:6], dtype=float)
    return np.concatenate([data_residuals, mount_prior])


def _hand_eye_initializations(
    T_base_links: Sequence[np.ndarray], T_camera_boards: Sequence[np.ndarray]
) -> list[tuple[str, np.ndarray]]:
    rotations_gripper_to_base = [value[:3, :3] for value in T_base_links]
    translations_gripper_to_base = [value[:3, 3].reshape(3, 1) for value in T_base_links]
    rotations_target_to_camera = [value[:3, :3] for value in T_camera_boards]
    translations_target_to_camera = [value[:3, 3].reshape(3, 1) for value in T_camera_boards]
    methods = [
        ("tsai", cv2.CALIB_HAND_EYE_TSAI),
        ("park", cv2.CALIB_HAND_EYE_PARK),
        ("horaud", cv2.CALIB_HAND_EYE_HORAUD),
        ("andreff", cv2.CALIB_HAND_EYE_ANDREFF),
        ("daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]
    initializations: list[tuple[str, np.ndarray]] = [("identity_fallback", np.eye(4))]
    for name, method in methods:
        try:
            rotation, translation = cv2.calibrateHandEye(
                rotations_gripper_to_base,
                translations_gripper_to_base,
                rotations_target_to_camera,
                translations_target_to_camera,
                method=method,
            )
        except cv2.error:
            continue
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = np.asarray(rotation, dtype=float)
        transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
        if (
            np.all(np.isfinite(transform))
            and abs(np.linalg.det(transform[:3, :3]) - 1.0) < 1e-3
        ):
            initializations.append((name, transform))
    return initializations


def _consistency_metrics(
    T_link_camera: np.ndarray,
    T_base_board_mean: np.ndarray,
    T_base_links: Sequence[np.ndarray],
    T_camera_boards: Sequence[np.ndarray],
) -> dict[str, float]:
    translation_errors = []
    rotation_errors_deg = []
    for T_base_link, T_camera_board in zip(T_base_links, T_camera_boards):
        board_pose = T_base_link @ T_link_camera @ T_camera_board
        translation_errors.append(
            float(np.linalg.norm(board_pose[:3, 3] - T_base_board_mean[:3, 3]))
        )
        rotation_errors_deg.append(
            rotation_error_deg(board_pose[:3, :3], T_base_board_mean[:3, :3])
        )
    translation_array = np.asarray(translation_errors)
    rotation_array_deg = np.asarray(rotation_errors_deg)
    rotation_array_rad = np.radians(rotation_array_deg)
    score = float(
        np.sqrt(
            np.mean(
                translation_array**2
                + (ROTATION_LENGTH_SCALE_M * rotation_array_rad) ** 2
            )
        )
    )
    return {
        "score": score,
        "translation_mean_m": fmean(translation_errors),
        "translation_median_m": median(translation_errors),
        "translation_max_m": max(translation_errors),
        "rotation_mean_deg": fmean(rotation_errors_deg),
        "rotation_median_deg": median(rotation_errors_deg),
        "rotation_max_deg": max(rotation_errors_deg),
    }


def motion_diversity_score(T_base_links: Sequence[np.ndarray]) -> float:
    """RMS pose spread in meter-equivalent units; diagnostic only."""
    mean_pose = average_transforms(T_base_links)
    squared = []
    for transform in T_base_links:
        translation = np.linalg.norm(transform[:3, 3] - mean_pose[:3, 3])
        rotation = np.radians(
            rotation_error_deg(transform[:3, :3], mean_pose[:3, :3])
        )
        squared.append(translation**2 + (ROTATION_LENGTH_SCALE_M * rotation) ** 2)
    return float(np.sqrt(np.mean(squared)))


@dataclass
class LinkHypothesisResult:
    link_name: str
    link_path_rel: str
    success: bool
    failure_reason: str
    num_valid_frames: int
    score: float | None = None
    rank: int | None = None
    translation_mean_m: float | None = None
    translation_median_m: float | None = None
    translation_max_m: float | None = None
    rotation_mean_deg: float | None = None
    rotation_median_deg: float | None = None
    rotation_max_deg: float | None = None
    motion_diversity_score: float | None = None
    initialization_method: str | None = None
    optimizer_success: bool | None = None
    optimizer_nfev: int | None = None
    observability_rank: int | None = None
    observability_parameter_count: int = 12
    mount_fully_observable: bool | None = None
    observability_condition_number: float | None = None
    T_link_camera: np.ndarray | None = field(default=None, repr=False)
    T_base_board_mean: np.ndarray | None = field(default=None, repr=False)

    def to_record(self) -> dict[str, Any]:
        return {
            "link_name": self.link_name,
            "link_path_rel": self.link_path_rel,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "score": self.score,
            "se3_residual_score": self.score,
            "rank": self.rank,
            "translation_mean_m": self.translation_mean_m,
            "translation_median_m": self.translation_median_m,
            "translation_max_m": self.translation_max_m,
            "rotation_mean_deg": self.rotation_mean_deg,
            "rotation_median_deg": self.rotation_median_deg,
            "rotation_max_deg": self.rotation_max_deg,
            "num_valid_frames": self.num_valid_frames,
            "motion_diversity_score": self.motion_diversity_score,
            "initialization_method": self.initialization_method,
            "optimizer_success": self.optimizer_success,
            "optimizer_nfev": self.optimizer_nfev,
            "observability_rank": self.observability_rank,
            "observability_parameter_count": self.observability_parameter_count,
            "mount_fully_observable": self.mount_fully_observable,
            "observability_condition_number": self.observability_condition_number,
        }


def estimate_link_hypothesis(
    link_name: str,
    link_path_rel: str,
    T_base_links: Sequence[np.ndarray],
    T_camera_boards: Sequence[np.ndarray],
    *,
    min_valid_poses: int = 10,
) -> LinkHypothesisResult:
    count = len(T_camera_boards)
    if count != len(T_base_links):
        return LinkHypothesisResult(
            link_name, link_path_rel, False, "pose_count_mismatch", count
        )
    if count < min_valid_poses:
        return LinkHypothesisResult(
            link_name, link_path_rel, False, "insufficient_valid_poses", count
        )
    links = [np.asarray(value, dtype=float).reshape(4, 4) for value in T_base_links]
    boards = [np.asarray(value, dtype=float).reshape(4, 4) for value in T_camera_boards]
    diversity = motion_diversity_score(links)
    initializations = _hand_eye_initializations(links, boards)
    scored_seeds = []
    for method, T_link_camera in initializations:
        T_base_board_mean = average_transforms(
            [link @ T_link_camera @ board for link, board in zip(links, boards)]
        )
        metrics = _consistency_metrics(
            T_link_camera, T_base_board_mean, links, boards
        )
        mount_parameters = parameters_from_transform(T_link_camera)
        canonical_seed_score = metrics["score"] + CANONICAL_GAUGE_STRENGTH * float(
            np.linalg.norm(mount_parameters)
        )
        scored_seeds.append(
            (
                canonical_seed_score,
                method,
                T_link_camera,
                T_base_board_mean,
            )
        )
    if not scored_seeds:
        return LinkHypothesisResult(
            link_name,
            link_path_rel,
            False,
            "hand_eye_initialization_failed",
            count,
            motion_diversity_score=diversity,
        )
    _, method, initial_mount, initial_board = min(scored_seeds, key=lambda item: item[0])
    initial_parameters = np.concatenate(
        [
            parameters_from_transform(initial_mount),
            parameters_from_transform(initial_board),
        ]
    )
    try:
        optimization = least_squares(
            _gauge_stabilized_residuals,
            initial_parameters,
            args=(links, boards),
            method="trf",
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=800,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return LinkHypothesisResult(
            link_name,
            link_path_rel,
            False,
            f"optimization_failed:{type(error).__name__}",
            count,
            motion_diversity_score=diversity,
            initialization_method=method,
        )
    if not np.all(np.isfinite(optimization.x)):
        return LinkHypothesisResult(
            link_name,
            link_path_rel,
            False,
            "optimization_produced_nonfinite_pose",
            count,
            motion_diversity_score=diversity,
            initialization_method=method,
            optimizer_success=bool(optimization.success),
            optimizer_nfev=int(optimization.nfev),
        )
    T_link_camera = transform_from_parameters(optimization.x[:6])
    T_base_board_mean = transform_from_parameters(optimization.x[6:])
    data_jacobian = np.asarray(optimization.jac[: 6 * count], dtype=float)
    singular_values = np.linalg.svd(data_jacobian, compute_uv=False)
    if singular_values.size and singular_values[0] > 0.0:
        rank_threshold = singular_values[0] * 1e-6
        observability_rank = int(np.count_nonzero(singular_values > rank_threshold))
        smallest = singular_values[-1]
        condition_number = (
            float(singular_values[0] / smallest)
            if smallest > np.finfo(float).eps
            else None
        )
    else:
        observability_rank = 0
        condition_number = None
    metrics = _consistency_metrics(
        T_link_camera, T_base_board_mean, links, boards
    )
    return LinkHypothesisResult(
        link_name=link_name,
        link_path_rel=link_path_rel,
        success=True,
        failure_reason="ok",
        num_valid_frames=count,
        motion_diversity_score=diversity,
        initialization_method=method,
        optimizer_success=bool(optimization.success),
        optimizer_nfev=int(optimization.nfev),
        observability_rank=observability_rank,
        mount_fully_observable=observability_rank == 12,
        observability_condition_number=condition_number,
        T_link_camera=T_link_camera,
        T_base_board_mean=T_base_board_mean,
        **metrics,
    )


@dataclass
class CameraLinkCalibrationResult:
    camera_name: str
    num_valid_frames: int
    hypotheses: list[LinkHypothesisResult]

    @property
    def successful_hypotheses(self) -> list[LinkHypothesisResult]:
        return sorted(
            (item for item in self.hypotheses if item.success and item.score is not None),
            key=lambda item: item.score,
        )

    def to_record(self) -> dict[str, Any]:
        successful = self.successful_hypotheses
        best = successful[0] if successful else None
        second = successful[1] if len(successful) > 1 else None
        if best is not None and best.T_link_camera is not None:
            translation, quaternion = t_q_from_mat(best.T_link_camera)
            matrix = best.T_link_camera.reshape(-1).tolist()
            translation_record = translation.tolist()
            quaternion_record = quaternion.tolist()
        else:
            matrix = translation_record = quaternion_record = None
        return {
            "camera_name": self.camera_name,
            "success": best is not None,
            "failure_reason": "ok" if best is not None else "no_successful_hypothesis",
            "best_link": best.link_name if best else None,
            "best_link_path_rel": best.link_path_rel if best else None,
            "second_best_link": second.link_name if second else None,
            "second_best_link_path_rel": second.link_path_rel if second else None,
            "best_score": best.score if best else None,
            "second_best_score": second.score if second else None,
            "score_margin": (
                second.score - best.score if best and second else None
            ),
            "num_valid_frames": self.num_valid_frames,
            "mount_fully_observable": (
                best.mount_fully_observable if best else None
            ),
            "observability_rank": best.observability_rank if best else None,
            "observability_parameter_count": (
                best.observability_parameter_count if best else 12
            ),
            "mount_estimation_note": (
                "fully_observable"
                if best and best.mount_fully_observable
                else (
                    "minimum_norm_gauge_due_to_rank_deficient_motion"
                    if best
                    else None
                )
            ),
            "T_link_camera_rowmajor": matrix,
            "t_link_camera": translation_record,
            "q_link_camera_xyzw": quaternion_record,
            "board_consistency_translation_m": {
                "mean": best.translation_mean_m if best else None,
                "median": best.translation_median_m if best else None,
                "max": best.translation_max_m if best else None,
            },
            "board_consistency_rotation_deg": {
                "mean": best.rotation_mean_deg if best else None,
                "median": best.rotation_median_deg if best else None,
                "max": best.rotation_max_deg if best else None,
            },
            "hypotheses": [item.to_record() for item in self.hypotheses],
            "gt_evaluation_available": False,
            "gt_best_link_correct": None,
            "gt_T_link_camera_translation_error_m": None,
            "gt_T_link_camera_rotation_error_deg": None,
        }


def rank_link_hypotheses(
    camera_name: str,
    observations: Sequence[BoardPoseObservation],
    candidate_links: Sequence[Mapping[str, Any]],
    link_poses_by_path: Mapping[str, Sequence[np.ndarray]],
    *,
    min_valid_poses: int = 10,
) -> CameraLinkCalibrationResult:
    board_poses = [item.T_camera_board for item in observations]
    hypotheses = []
    for candidate in candidate_links:
        path = str(candidate["link_path_rel"])
        link_poses = link_poses_by_path.get(path, [])
        hypotheses.append(
            estimate_link_hypothesis(
                str(candidate["link_name"]),
                path,
                link_poses,
                board_poses,
                min_valid_poses=min_valid_poses,
            )
        )
    ordered = sorted(
        hypotheses,
        key=lambda item: (
            not item.success,
            item.score if item.score is not None else float("inf"),
            item.link_name,
        ),
    )
    for rank, hypothesis in enumerate(ordered, start=1):
        hypothesis.rank = rank
    return CameraLinkCalibrationResult(camera_name, len(observations), ordered)
