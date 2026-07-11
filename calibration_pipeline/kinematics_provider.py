"""Kinematic pose sources used independently of the calibration core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .dataset_loader import load_json_document, normalize_frame_id


class KinematicsProvider(ABC):
    @abstractmethod
    def get_candidate_link_poses(self, frame_id: str | int) -> dict[str, np.ndarray]:
        """Return ``{link_path_or_name: T_base_link}`` for one frame."""


class UnityLinkPoseProvider(KinematicsProvider):
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        self.link_pose_dir = self.dataset_root / "link_poses"

    def frame_path(self, frame_id: str | int) -> Path:
        return self.link_pose_dir / f"{normalize_frame_id(frame_id)}.json"

    def get_candidate_link_poses(self, frame_id: str | int) -> dict[str, np.ndarray]:
        path = self.frame_path(frame_id)
        document, _ = load_json_document(path)
        poses: dict[str, np.ndarray] = {}
        for link in document.get("links", []):
            if not link.get("valid", True):
                continue
            key = link.get("link_path_rel") or link.get("link_name")
            if not key:
                raise ValueError(f"Link without name or path in {path}")
            matrix = np.asarray(link["T_base_link_rowmajor"], dtype=float)
            if matrix.size != 16:
                raise ValueError(f"Invalid link transform for {key} in {path}")
            poses[str(key)] = matrix.reshape(4, 4)
        return poses

