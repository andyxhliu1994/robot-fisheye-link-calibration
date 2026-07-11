"""Small, convention-explicit SE(3) helpers.

All matrices follow ``T_A_B``: they map coordinates from frame B to frame A.
Quaternions are ordered ``[x, y, z, w]``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _as_transform(T: np.ndarray) -> np.ndarray:
    value = np.asarray(T, dtype=float)
    if value.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {value.shape}")
    return value


def mat_from_t_q(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    translation = np.asarray(t, dtype=float)
    quaternion = np.asarray(q, dtype=float)
    if translation.shape != (3,):
        raise ValueError("Translation must contain exactly three values")
    if quaternion.shape != (4,):
        raise ValueError("Quaternion must contain exactly four xyzw values")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Transform values must be finite")
    norm = np.linalg.norm(quaternion)
    if norm <= np.finfo(float).eps:
        raise ValueError("Quaternion norm must be nonzero")
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    T[:3, 3] = translation
    return T


def t_q_from_mat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = _as_transform(T)
    return value[:3, 3].copy(), Rotation.from_matrix(value[:3, :3]).as_quat()


def invert_T(T: np.ndarray) -> np.ndarray:
    value = _as_transform(T)
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = value[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ value[:3, 3]
    return inverse


def compose_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _as_transform(A) @ _as_transform(B)


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    first = np.asarray(R1, dtype=float)
    second = np.asarray(R2, dtype=float)
    if first.shape != (3, 3) or second.shape != (3, 3):
        raise ValueError("Rotation matrices must be 3x3")
    relative = Rotation.from_matrix(first).inv() * Rotation.from_matrix(second)
    return float(np.degrees(relative.magnitude()))


def translation_error_m(t1: np.ndarray, t2: np.ndarray) -> float:
    first = np.asarray(t1, dtype=float)
    second = np.asarray(t2, dtype=float)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("Translations must contain exactly three values")
    return float(np.linalg.norm(first - second))

