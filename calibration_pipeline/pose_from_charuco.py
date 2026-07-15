"""Bearing-only ChArUco board pose estimation.

The optimizer consumes corresponding board points and unit camera rays. It has
no dependency on robot kinematics or exported ground-truth transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .se3_utils import mat_from_t_q, t_q_from_mat


BOARD_POSE_FRAME = "charuco_outer_corner_x_right_y_down_z_board_normal"


def _validated_correspondences(
    board_points: np.ndarray, observed_rays: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(board_points, dtype=float)
    rays = np.asarray(observed_rays, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("board_points must have shape (N, 3)")
    if rays.shape != points.shape:
        raise ValueError("observed_rays must have the same (N, 3) shape")
    if len(points) < 4:
        raise ValueError("At least four point/ray correspondences are required")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(rays)):
        raise ValueError("Point/ray correspondences must be finite")
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Observed rays must be nonzero")
    return points, rays / norms[:, None]


def transform_points(parameters: np.ndarray, board_points: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (6,):
        raise ValueError("Pose parameters must be [rotation_vector, translation]")
    rotation = Rotation.from_rotvec(values[:3]).as_matrix()
    return (rotation @ np.asarray(board_points, dtype=float).T).T + values[3:]


def ray_residuals(
    parameters: np.ndarray, board_points: np.ndarray, observed_rays: np.ndarray
) -> np.ndarray:
    """Vector-direction residuals, equivalent to angular error near optimum.

    Direction-vector subtraction is used instead of only a cross product so an
    antipodal (behind-camera) ray cannot have a zero residual.
    """
    points, rays = _validated_correspondences(board_points, observed_rays)
    camera_points = transform_points(parameters, points)
    norms = np.linalg.norm(camera_points, axis=1)
    norms = np.maximum(norms, np.finfo(float).eps)
    predicted_rays = camera_points / norms[:, None]
    return (predicted_rays - rays).reshape(-1)


def ray_errors_deg(
    T_camera_board: np.ndarray,
    board_points: np.ndarray,
    observed_rays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, rays = _validated_correspondences(board_points, observed_rays)
    transform = np.asarray(T_camera_board, dtype=float).reshape(4, 4)
    camera_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    predicted = camera_points / np.maximum(
        np.linalg.norm(camera_points, axis=1, keepdims=True), np.finfo(float).eps
    )
    cosines = np.clip(np.sum(predicted * rays, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosines)), camera_points


def _pnp_initializations(
    board_points: np.ndarray, observed_rays: np.ndarray
) -> list[np.ndarray]:
    """Obtain pose seeds from normalized ray coordinates, never from GT."""
    forward = observed_rays[:, 2] > 1e-3
    if np.count_nonzero(forward) < 4:
        return []
    points = np.ascontiguousarray(board_points[forward], dtype=np.float64)
    normalized_pixels = np.ascontiguousarray(
        observed_rays[forward, :2] / observed_rays[forward, 2, None],
        dtype=np.float64,
    )
    candidates: list[np.ndarray] = []
    flags = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(cv2.SOLVEPNP_IPPE)
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags.append(cv2.SOLVEPNP_SQPNP)
    if not flags:
        flags.append(cv2.SOLVEPNP_EPNP)
    for flag in flags:
        try:
            result = cv2.solvePnPGeneric(
                points,
                normalized_pixels,
                np.eye(3, dtype=np.float64),
                None,
                flags=flag,
            )
        except cv2.error:
            continue
        if not result[0]:
            continue
        for rotation_vector, translation in zip(result[1], result[2]):
            candidate = np.concatenate(
                [
                    np.asarray(rotation_vector, dtype=float).reshape(3),
                    np.asarray(translation, dtype=float).reshape(3),
                ]
            )
            if np.all(np.isfinite(candidate)):
                candidates.append(candidate)
    return candidates


@dataclass
class BoardPoseEstimate:
    valid: bool
    reason: str
    charuco_corner_count: int
    T_camera_board: np.ndarray | None = field(default=None, repr=False)
    mean_ray_error_deg: float | None = None
    max_ray_error_deg: float | None = None
    num_points_behind_camera: int | None = None
    optimizer_cost: float | None = None
    optimizer_nfev: int | None = None
    per_point_ray_error_deg: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def failure(cls, reason: str, corner_count: int) -> "BoardPoseEstimate":
        return cls(False, reason, corner_count)

    def to_record(
        self, frame_id: str, camera_name: str, camera_pose_frame: str
    ) -> dict[str, Any]:
        if self.T_camera_board is None:
            translation = None
            quaternion = None
            matrix = None
        else:
            translation_array, quaternion_array = t_q_from_mat(self.T_camera_board)
            translation = translation_array.tolist()
            quaternion = quaternion_array.tolist()
            matrix = self.T_camera_board.reshape(-1).tolist()
        return {
            "frame_id": frame_id,
            "camera_name": camera_name,
            "camera_pose_frame": camera_pose_frame,
            "board_pose_frame": BOARD_POSE_FRAME,
            "valid": self.valid,
            "charuco_corner_count": self.charuco_corner_count,
            "T_camera_board_rowmajor": matrix,
            "t_camera_board": translation,
            "q_camera_board_xyzw": quaternion,
            "mean_ray_error_deg": self.mean_ray_error_deg,
            "max_ray_error_deg": self.max_ray_error_deg,
            "num_points_behind_camera": self.num_points_behind_camera,
            "optimizer_cost": self.optimizer_cost,
            "optimizer_nfev": self.optimizer_nfev,
            "reason": self.reason,
        }


def estimate_pose_from_rays(
    board_points: np.ndarray,
    observed_rays: np.ndarray,
    *,
    min_points: int = 8,
    max_mean_ray_error_deg: float = 1.0,
) -> BoardPoseEstimate:
    points = np.asarray(board_points, dtype=float)
    rays = np.asarray(observed_rays, dtype=float)
    corner_count = len(points) if points.ndim else 0
    if corner_count < min_points:
        return BoardPoseEstimate.failure("insufficient_corners", corner_count)
    try:
        points, rays = _validated_correspondences(points, rays)
    except ValueError:
        return BoardPoseEstimate.failure("invalid_correspondences", corner_count)
    candidates = _pnp_initializations(points, rays)
    if not candidates:
        return BoardPoseEstimate.failure("pose_initialization_failed", corner_count)

    solutions: list[tuple[tuple[bool, float], Any, np.ndarray, np.ndarray]] = []
    for initial in candidates:
        try:
            optimization = least_squares(
                ray_residuals,
                initial,
                args=(points, rays),
                method="trf",
                loss="soft_l1",
                f_scale=np.radians(1.0),
                max_nfev=400,
            )
            if not np.all(np.isfinite(optimization.x)):
                continue
            T_camera_board = mat_from_t_q(
                optimization.x[3:], Rotation.from_rotvec(optimization.x[:3]).as_quat()
            )
            errors, camera_points = ray_errors_deg(
                T_camera_board, points, rays
            )
            behind_count = int(np.count_nonzero(camera_points[:, 2] <= 0.0))
            ranking = (behind_count > 0, float(np.mean(errors)))
            solutions.append((ranking, optimization, T_camera_board, errors))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if not solutions:
        return BoardPoseEstimate.failure("pose_optimization_failed", corner_count)

    _, best, transform, errors = min(solutions, key=lambda item: item[0])
    _, camera_points = ray_errors_deg(transform, points, rays)
    behind_count = int(np.count_nonzero(camera_points[:, 2] <= 0.0))
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    if behind_count:
        valid = False
        reason = "points_behind_camera"
    elif not best.success:
        valid = False
        reason = "pose_optimization_failed"
    elif mean_error > max_mean_ray_error_deg:
        valid = False
        reason = "mean_ray_error_too_high"
    else:
        valid = True
        reason = "ok"
    return BoardPoseEstimate(
        valid=valid,
        reason=reason,
        charuco_corner_count=corner_count,
        T_camera_board=transform,
        mean_ray_error_deg=mean_error,
        max_ray_error_deg=max_error,
        num_points_behind_camera=behind_count,
        optimizer_cost=float(best.cost),
        optimizer_nfev=int(best.nfev),
        per_point_ray_error_deg=errors,
    )
