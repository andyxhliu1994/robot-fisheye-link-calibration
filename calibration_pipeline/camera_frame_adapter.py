"""Adapters between a camera model's ray frame and its pose frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraFrameAdapter:
    matrix: np.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("Camera frame adapter matrix must be 3x3")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Camera frame adapter must contain finite values")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-9):
            raise ValueError("Camera frame adapter must be orthogonal")
        object.__setattr__(self, "matrix", matrix)

    def adapt_ray(self, ray: np.ndarray) -> np.ndarray:
        value = np.asarray(ray, dtype=float)
        if value.shape != (3,):
            raise ValueError("Ray must contain exactly three values")
        adapted = self.matrix @ value
        norm = np.linalg.norm(adapted)
        if norm <= np.finfo(float).eps:
            raise ValueError("Ray norm must be nonzero")
        return adapted / norm

    __call__ = adapt_ray


def camera_frame_adapter_from_config(config: Mapping[str, Any]) -> CameraFrameAdapter:
    adapter_type = config.get("type", "identity")
    name = str(config.get("name", adapter_type))
    if adapter_type == "identity":
        return CameraFrameAdapter(np.eye(3), name)
    if adapter_type in {"flip_y_to_unity", "flip_y_to_unity_camera_frame"}:
        return CameraFrameAdapter(np.diag([1.0, -1.0, 1.0]), name)
    if adapter_type == "matrix_3x3":
        if "matrix" not in config:
            raise ValueError("matrix_3x3 adapter requires a matrix")
        return CameraFrameAdapter(np.asarray(config["matrix"], dtype=float), name)
    raise ValueError(f"Unsupported camera frame adapter type: {adapter_type}")

