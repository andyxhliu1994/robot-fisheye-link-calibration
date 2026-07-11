"""Ground-truth-only integrity evaluation (never used for estimation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dataset_loader import DatasetLoader, load_json_document, normalize_frame_id
from .kinematics_provider import UnityLinkPoseProvider
from .se3_utils import mat_from_t_q, rotation_error_deg, translation_error_m


DEFAULT_THRESHOLDS = {
    "translation_error_m": 1e-5,
    "rotation_error_deg": 1e-2,
    "max_matrix_element_error": 1e-4,
}


def evaluate_gt_consistency(
    dataset_root: str | Path,
    frame_ids: Iterable[str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Check ``T_base_link_gt @ T_link_camera_gt == T_base_camera_gt``."""
    root = Path(dataset_root)
    loader = DatasetLoader(root)
    provider = UnityLinkPoseProvider(root)
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for frame_id_value in frame_ids:
        frame_id = normalize_frame_id(frame_id_value)
        try:
            link_poses = provider.get_candidate_link_poses(frame_id)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{frame_id}: unavailable link poses ({error})")
            continue
        for camera in loader.camera_names():
            path = root / "cameras" / camera / "transform" / f"{frame_id}.json"
            if not path.is_file():
                warnings.append(f"{camera}/{frame_id}: missing transform file; skipped")
                continue
            try:
                gt, _ = load_json_document(path)
                link_path = gt["gt_link_path_rel"]
                if link_path not in link_poses:
                    raise KeyError(f"GT link is unavailable: {link_path}")
                T_link_camera_gt = mat_from_t_q(
                    gt["gt_t_link_from_cam"], gt["gt_q_link_from_cam_xyzw"]
                )
                predicted = link_poses[link_path] @ T_link_camera_gt
                exported = np.asarray(gt["T_base_cam_rowmajor"], dtype=float).reshape(4, 4)
                translation_error = translation_error_m(predicted[:3, 3], exported[:3, 3])
                rotation_error = rotation_error_deg(predicted[:3, :3], exported[:3, :3])
                matrix_error = float(np.max(np.abs(predicted - exported)))
                passed = (
                    translation_error <= limits["translation_error_m"]
                    and rotation_error <= limits["rotation_error_deg"]
                    and matrix_error <= limits["max_matrix_element_error"]
                )
                records.append(
                    {
                        "camera_name": camera,
                        "frame_id": frame_id,
                        "gt_link_path_rel": link_path,
                        "translation_error_m": translation_error,
                        "rotation_error_deg": rotation_error,
                        "max_matrix_element_error": matrix_error,
                        "passed": passed,
                    }
                )
            except (KeyError, TypeError, ValueError, OSError) as error:
                warnings.append(f"{camera}/{frame_id}: invalid GT data; skipped ({error})")

    def metric_summary(key: str) -> dict[str, float | None]:
        values = np.asarray([record[key] for record in records], dtype=float)
        if not len(values):
            return {"max": None, "mean": None, "median": None}
        return {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }

    failed = [record for record in records if not record["passed"]]
    return {
        "equation": "T_base_camera_pred = T_base_link_gt @ T_link_camera_gt",
        "mount_field_interpretation": "gt_*_link_from_cam is treated as T_link_camera_gt without inversion",
        "thresholds": limits,
        "requested_frame_count": len(list(frame_ids)) if not isinstance(frame_ids, list) else len(frame_ids),
        "evaluated_composition_count": len(records),
        "failed_composition_count": len(failed),
        "translation_error_m": metric_summary("translation_error_m"),
        "rotation_error_deg": metric_summary("rotation_error_deg"),
        "max_matrix_element_error": metric_summary("max_matrix_element_error"),
        "warnings": warnings,
        "failures": failed[:100],
        "passed": bool(records) and not failed,
    }

