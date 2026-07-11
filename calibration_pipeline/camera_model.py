"""Camera models which turn pixels into unit rays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


class CameraModel(Protocol):
    def pixel_to_ray(
        self, u: float, v: float, image_width: int, image_height: int
    ) -> np.ndarray:
        """Return a unit ray in the model's calibration ray frame."""


@dataclass(frozen=True)
class OCamCalibCameraModel:
    distortion_center: np.ndarray
    stretch_matrix: np.ndarray
    taylor_coefficient: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.distortion_center, dtype=float)
        stretch = np.asarray(self.stretch_matrix, dtype=float)
        coefficients = np.asarray(self.taylor_coefficient, dtype=float)
        if center.shape != (2,) or stretch.shape != (2, 2):
            raise ValueError("OCamCalib center and stretch must be 2-vector and 2x2")
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("OCamCalib Taylor polynomial must be nonempty")
        if abs(np.linalg.det(stretch)) <= np.finfo(float).eps:
            raise ValueError("OCamCalib stretch matrix must be invertible")
        object.__setattr__(self, "distortion_center", center)
        object.__setattr__(self, "stretch_matrix", stretch)
        object.__setattr__(self, "taylor_coefficient", coefficients)

    @classmethod
    def from_mapping(cls, calibration: Mapping[str, Any]) -> "OCamCalibCameraModel":
        return cls(
            calibration["distortion_center"],
            calibration["stretch_matrix"],
            calibration["taylor_coefficient"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "OCamCalibCameraModel":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def pixel_to_ray(
        self, u: float, v: float, image_width: int, image_height: int
    ) -> np.ndarray:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        calibration_width = round(2.0 * self.distortion_center[0])
        calibration_height = round(2.0 * self.distortion_center[1])
        if calibration_width <= 0 or calibration_height <= 0:
            raise ValueError("Calibration image dimensions must be positive")
        sx = image_width / calibration_width
        sy = image_height / calibration_height
        center = np.array([sx, sy]) * self.distortion_center
        scaled_stretch = np.diag([sx, sy]) @ self.stretch_matrix
        inverse_transpose = np.linalg.inv(scaled_stretch).T
        delta = np.array([float(u), float(v)]) - center
        # This expanded form intentionally matches the validated OCamCalib export.
        X = delta[0] * inverse_transpose[0, 0] + delta[1] * inverse_transpose[1, 0]
        Y = delta[0] * inverse_transpose[0, 1] + delta[1] * inverse_transpose[1, 1]
        rho = float(np.hypot(X, Y))
        z = float(np.polyval(self.taylor_coefficient[::-1], rho))
        ray = np.array([X, Y, z], dtype=float)
        norm = np.linalg.norm(ray)
        if norm <= np.finfo(float).eps or not np.isfinite(norm):
            raise ValueError("Pixel produced an invalid OCamCalib ray")
        return ray / norm

