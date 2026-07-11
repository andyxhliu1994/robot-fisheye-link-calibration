"""ChArUco detection and visualization, deliberately without pose estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .charuco_config import CharucoBoardConfig


@dataclass
class CharucoDetection:
    frame_id: str
    camera_name: str
    valid: bool
    marker_count: int
    charuco_corner_count: int
    board_bbox_area_px: float | None
    reason: str
    marker_corners: list[np.ndarray] = field(default_factory=list, repr=False)
    marker_ids: np.ndarray | None = field(default=None, repr=False)
    charuco_corners: np.ndarray | None = field(default=None, repr=False)
    charuco_ids: np.ndarray | None = field(default=None, repr=False)

    def to_record(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            "valid": self.valid,
            "marker_count": self.marker_count,
            "charuco_corner_count": self.charuco_corner_count,
            "board_bbox_area_px": self.board_bbox_area_px,
            "reason": self.reason,
        }


class CharucoDetector:
    def __init__(
        self,
        config: CharucoBoardConfig,
        min_markers: int = 4,
        min_charuco_corners: int = 8,
    ) -> None:
        if min_markers < 1 or min_charuco_corners < 1:
            raise ValueError("Detection thresholds must be positive")
        self.config = config
        self.dictionary = config.create_dictionary()
        self.board = config.create_board()
        self.min_markers = min_markers
        self.min_charuco_corners = min_charuco_corners
        self.parameters = self._create_detector_parameters()
        self._aruco_detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

    @staticmethod
    def _create_detector_parameters():
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            return cv2.aruco.DetectorParameters_create()
        raise RuntimeError("This OpenCV version cannot create ArUco detector parameters")

    def _detect_markers(self, gray: np.ndarray):
        if self._aruco_detector is not None:
            return self._aruco_detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

    @staticmethod
    def _bbox_area(
        marker_corners: list[np.ndarray], charuco_corners: np.ndarray | None
    ) -> float | None:
        point_groups = [np.asarray(corners).reshape(-1, 2) for corners in marker_corners]
        if charuco_corners is not None and len(charuco_corners):
            point_groups.append(np.asarray(charuco_corners).reshape(-1, 2))
        if not point_groups:
            return None
        points = np.concatenate(point_groups, axis=0)
        width, height = np.ptp(points, axis=0)
        return float(width * height)

    def detect(
        self, image: np.ndarray | None, frame_id: str, camera_name: str
    ) -> CharucoDetection:
        if image is None or image.size == 0:
            return CharucoDetection(
                frame_id, camera_name, False, 0, 0, None, "image_read_failed"
            )
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, conversion)
        else:
            return CharucoDetection(
                frame_id, camera_name, False, 0, 0, None, "unsupported_image_shape"
            )

        marker_corners, marker_ids, _ = self._detect_markers(gray)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        charuco_corners = None
        charuco_ids = None
        interpolation_failed = False
        if marker_count:
            try:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
            except cv2.error:
                interpolation_failed = True

        charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
        valid = marker_count >= self.min_markers and charuco_count >= self.min_charuco_corners
        if valid:
            reason = "ok"
        elif marker_count == 0:
            reason = "no_markers_detected"
        elif marker_count < self.min_markers:
            reason = "insufficient_markers"
        elif interpolation_failed:
            reason = "charuco_interpolation_failed"
        elif charuco_count == 0:
            reason = "no_charuco_corners"
        else:
            reason = "insufficient_charuco_corners"
        return CharucoDetection(
            frame_id=frame_id,
            camera_name=camera_name,
            valid=valid,
            marker_count=marker_count,
            charuco_corner_count=charuco_count,
            board_bbox_area_px=self._bbox_area(marker_corners, charuco_corners),
            reason=reason,
            marker_corners=marker_corners,
            marker_ids=marker_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
        )

    @staticmethod
    def draw_overlay(image: np.ndarray, detection: CharucoDetection) -> np.ndarray:
        if image.ndim == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            overlay = image.copy()
        if detection.marker_ids is not None and len(detection.marker_ids):
            cv2.aruco.drawDetectedMarkers(
                overlay, detection.marker_corners, detection.marker_ids
            )
        if detection.charuco_ids is not None and len(detection.charuco_ids):
            cv2.aruco.drawDetectedCornersCharuco(
                overlay, detection.charuco_corners, detection.charuco_ids, (255, 0, 255)
            )

        status = "VALID" if detection.valid else f"INVALID: {detection.reason}"
        lines = [
            detection.frame_id,
            f"markers={detection.marker_count} charuco={detection.charuco_corner_count}",
            status,
        ]
        color = (40, 220, 40) if detection.valid else (40, 40, 240)
        for index, text in enumerate(lines):
            origin = (20, 36 + index * 34)
            cv2.putText(
                overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
            )
        return overlay

