"""Camera-stream and exported-transform sanity checks.

This module is evaluation-only. It diagnoses recording/export problems and does
not estimate any camera or board pose.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import cv2
import numpy as np

from .dataset_loader import DatasetLoader, frame_number, load_json_document
from .se3_utils import mat_from_t_q, rotation_error_deg, translation_error_m


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_difference(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Return basic pixel differences without adding image-analysis dependencies."""
    if first.shape != second.shape:
        return {
            "same_shape": False,
            "mean_absolute_difference": None,
            "max_absolute_difference": None,
            "rmse": None,
        }
    difference = cv2.absdiff(first, second)
    channel_means = cv2.mean(difference)
    active_channels = 1 if difference.ndim == 2 else difference.shape[2]
    mean_absolute = float(sum(channel_means[:active_channels]) / active_channels)
    rmse = float(cv2.norm(first, second, cv2.NORM_L2) / np.sqrt(first.size))
    return {
        "same_shape": True,
        "mean_absolute_difference": mean_absolute,
        "max_absolute_difference": int(np.max(difference)),
        "rmse": rmse,
    }


def pose_pairwise_comparisons(
    transforms: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    comparisons = []
    for first_name, second_name in combinations(sorted(transforms), 2):
        first = np.asarray(transforms[first_name], dtype=float).reshape(4, 4)
        second = np.asarray(transforms[second_name], dtype=float).reshape(4, 4)
        comparisons.append(
            {
                "camera_a": first_name,
                "camera_b": second_name,
                "translation_difference_m": translation_error_m(
                    first[:3, 3], second[:3, 3]
                ),
                "rotation_difference_deg": rotation_error_deg(
                    first[:3, :3], second[:3, :3]
                ),
            }
        )
    return comparisons


def diagnose_dataset(
    *,
    input_complete: bool,
    all_frames_all_cameras_byte_identical: bool,
    affected_frame_count: int,
    transforms_different: bool,
) -> dict[str, Any]:
    if not input_complete:
        return {
            "passed": False,
            "status": "FAIL",
            "diagnosis_code": "DATASET_INCOMPLETE",
            "diagnosis": "Required sampled camera RGB or transform files are missing or unreadable.",
            "recommendation": "Repair or re-export the incomplete recording before pose estimation.",
        }
    if all_frames_all_cameras_byte_identical and transforms_different:
        return {
            "passed": False,
            "status": "FAIL",
            "diagnosis_code": "DATASET_IMAGE_EXPORT_SUSPECT",
            "diagnosis": (
                "The camera RGB streams are byte-identical across cameras for every "
                "sampled frame, but the exported camera transforms are different."
            ),
            "recommendation": (
                "The Unity recorder may have saved the same camera or render texture "
                "to every camera folder. Do not proceed to pose estimation until the "
                "image export pipeline is fixed or independently verified."
            ),
        }
    if all_frames_all_cameras_byte_identical:
        return {
            "passed": False,
            "status": "FAIL",
            "diagnosis_code": "CAMERA_SETUP_DUPLICATED",
            "diagnosis": "Camera RGB streams and exported camera setup both appear duplicated.",
            "recommendation": "Verify camera creation, mounting, and recording before pose estimation.",
        }
    if affected_frame_count:
        return {
            "passed": False,
            "status": "WARN",
            "diagnosis_code": "PARTIAL_CAMERA_STREAM_DUPLICATION",
            "diagnosis": "Some sampled same-frame RGB files are duplicated across cameras.",
            "recommendation": "Investigate the affected frames and recorder routing before pose estimation.",
        }
    if not transforms_different:
        return {
            "passed": False,
            "status": "WARN",
            "diagnosis_code": "CAMERA_TRANSFORMS_NOT_DISTINCT",
            "diagnosis": "RGB streams differ, but exported camera transforms are not distinct.",
            "recommendation": "Verify the camera mounting/export metadata before pose estimation.",
        }
    return {
        "passed": True,
        "status": "PASS",
        "diagnosis_code": "CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT",
        "diagnosis": "Sampled camera RGB streams and camera transforms are distinct.",
        "recommendation": "The distinctness checks do not block the next milestone.",
    }


def _select_frame_ids(rgb_dir: Path, max_frames: int, stride: int) -> list[str]:
    if max_frames <= 0 or stride <= 0:
        raise ValueError("max_frames and frame_stride must be positive")
    frames: list[str] = []
    for path in rgb_dir.glob("frame_*.jpg"):
        try:
            frame_number(path.stem)
        except ValueError:
            continue
        frames.append(path.stem)
    frames.sort(key=frame_number)
    return frames[::stride][:max_frames]


def _summarize_metric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": fmean(values)}


def _aggregate_pose_comparisons(
    comparisons: Mapping[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    translations: dict[str, Any] = {}
    rotations: dict[str, Any] = {}
    different = False
    for pair, values in sorted(comparisons.items()):
        translation_values = [item["translation_difference_m"] for item in values]
        rotation_values = [item["rotation_difference_deg"] for item in values]
        translations[pair] = {
            **_summarize_metric(translation_values),
            "compared_frame_count": len(values),
        }
        rotations[pair] = {
            **_summarize_metric(rotation_values),
            "compared_frame_count": len(values),
        }
        if any(value > 1e-9 for value in translation_values) or any(
            value > 1e-6 for value in rotation_values
        ):
            different = True
    return translations, rotations, different


def inspect_detection_outputs(output_root: Path, camera_names: list[str]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    canonical_contents: dict[str, list[dict[str, Any]]] = {}
    missing = []
    for camera in camera_names:
        path = output_root / "detections" / f"{camera}.jsonl"
        if not path.is_file():
            missing.append(camera)
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        summaries[camera] = {
            "record_count": len(records),
            "valid_count": sum(bool(record.get("valid")) for record in records),
            "mean_marker_count": (
                fmean(float(record.get("marker_count", 0)) for record in records)
                if records
                else 0.0
            ),
            "mean_charuco_corner_count": (
                fmean(float(record.get("charuco_corner_count", 0)) for record in records)
                if records
                else 0.0
            ),
        }
        canonical_contents[camera] = [
            {key: value for key, value in record.items() if key != "camera_name"}
            for record in records
        ]
    values = list(canonical_contents.values())
    return {
        "available": bool(values),
        "missing_cameras": missing,
        "per_camera_summary": summaries,
        "contents_identical_ignoring_camera_name": (
            all(value == values[0] for value in values[1:]) if values else None
        ),
    }


def run_dataset_sanity_check(
    dataset_root: str | Path,
    max_frames: int = 50,
    frame_stride: int = 10,
    output_root_for_detections: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    loader = DatasetLoader(root)
    camera_names = loader.camera_names()
    if len(camera_names) < 2:
        raise ValueError("At least two camera directories are required")
    frame_ids = _select_frame_ids(
        root / "cameras" / camera_names[0] / "rgb", max_frames, frame_stride
    )
    if not frame_ids:
        raise ValueError("No RGB frames were selected")

    session_summary, _ = load_json_document(root / "session_summary.json")
    candidate_links, _ = load_json_document(root / "candidate_links.json")
    hash_history: dict[str, list[tuple[str, str]]] = defaultdict(list)
    previous_images: dict[str, np.ndarray] = {}
    temporal_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pixel_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_frames: list[dict[str, Any]] = []
    missing_rgb: list[str] = []

    for frame_id in frame_ids:
        hashes: dict[str, str] = {}
        file_sizes: dict[str, int] = {}
        images: dict[str, np.ndarray] = {}
        for camera in camera_names:
            path = root / "cameras" / camera / "rgb" / f"{frame_id}.jpg"
            if not path.is_file():
                missing_rgb.append(f"{camera}/{frame_id}")
                continue
            digest = sha256_file(path)
            hashes[camera] = digest
            file_sizes[camera] = path.stat().st_size
            hash_history[camera].append((frame_id, digest))
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                missing_rgb.append(f"{camera}/{frame_id}:unreadable")
                continue
            images[camera] = image
            if camera in previous_images:
                temporal_metrics[camera].append(image_difference(previous_images[camera], image))
            previous_images[camera] = image

        grouped: dict[str, list[str]] = defaultdict(list)
        for camera, digest in hashes.items():
            grouped[digest].append(camera)
        duplicate_groups = [sorted(group) for group in grouped.values() if len(group) > 1]
        all_identical = len(hashes) == len(camera_names) and len(grouped) == 1
        if duplicate_groups:
            identity_frames.append(
                {
                    "frame_id": frame_id,
                    "all_cameras_byte_identical": all_identical,
                    "sha256_by_camera": hashes,
                    "file_size_by_camera": file_sizes,
                    "identical_camera_groups": duplicate_groups,
                }
            )

        for camera_a, camera_b in combinations(camera_names, 2):
            if camera_a not in images or camera_b not in images:
                continue
            pair = f"{camera_a}__vs__{camera_b}"
            if hashes.get(camera_a) == hashes.get(camera_b):
                metric = {
                    "same_shape": images[camera_a].shape == images[camera_b].shape,
                    "mean_absolute_difference": 0.0,
                    "max_absolute_difference": 0,
                    "rmse": 0.0,
                }
            else:
                metric = image_difference(images[camera_a], images[camera_b])
            pixel_metrics[pair].append(metric)

    pairwise_pixel_summary = []
    for pair, metrics in sorted(pixel_metrics.items()):
        comparable = [item for item in metrics if item["same_shape"]]
        pairwise_pixel_summary.append(
            {
                "camera_pair": pair,
                "compared_frame_count": len(metrics),
                "shape_mismatch_count": len(metrics) - len(comparable),
                "mean_absolute_difference": (
                    fmean(item["mean_absolute_difference"] for item in comparable)
                    if comparable
                    else None
                ),
                "max_absolute_difference": (
                    max(item["max_absolute_difference"] for item in comparable)
                    if comparable
                    else None
                ),
                "rmse": (
                    fmean(item["rmse"] for item in comparable) if comparable else None
                ),
                "all_compared_pixels_identical": bool(comparable)
                and all(item["max_absolute_difference"] == 0 for item in comparable),
            }
        )

    temporal_summaries = []
    for camera in camera_names:
        entries = hash_history[camera]
        digest_groups: dict[str, list[str]] = defaultdict(list)
        for frame_id, digest in entries:
            digest_groups[digest].append(frame_id)
        repeated_groups = [frames for frames in digest_groups.values() if len(frames) > 1]
        metrics = [item for item in temporal_metrics[camera] if item["same_shape"]]
        adjacent_identical = sum(item["max_absolute_difference"] == 0 for item in metrics)
        temporal_summaries.append(
            {
                "camera_name": camera,
                "sampled_frame_count": len(entries),
                "unique_sha256_count": len(digest_groups),
                "adjacent_comparison_count": len(temporal_metrics[camera]),
                "adjacent_identical_count": adjacent_identical,
                "changed_adjacent_count": len(metrics) - adjacent_identical,
                "repeated_hash_group_count": len(repeated_groups),
                "repeated_hash_examples": repeated_groups[:10],
                "mean_adjacent_absolute_difference": (
                    fmean(item["mean_absolute_difference"] for item in metrics)
                    if metrics
                    else None
                ),
                "mean_adjacent_rmse": (
                    fmean(item["rmse"] for item in metrics) if metrics else None
                ),
            }
        )

    base_pose_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    local_pose_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gt_links: dict[str, set[str]] = defaultdict(set)
    missing_transforms: list[str] = []
    for frame_id in frame_ids:
        base_transforms: dict[str, np.ndarray] = {}
        local_transforms: dict[str, np.ndarray] = {}
        for camera in camera_names:
            path = root / "cameras" / camera / "transform" / f"{frame_id}.json"
            if not path.is_file():
                missing_transforms.append(f"{camera}/{frame_id}")
                continue
            document, _ = load_json_document(path)
            try:
                base_transforms[camera] = np.asarray(
                    document["T_base_cam_rowmajor"], dtype=float
                ).reshape(4, 4)
                local_transforms[camera] = mat_from_t_q(
                    document["gt_t_link_from_cam"],
                    document["gt_q_link_from_cam_xyzw"],
                )
                gt_links[camera].add(str(document["gt_link_path_rel"]))
            except (KeyError, TypeError, ValueError) as error:
                missing_transforms.append(f"{camera}/{frame_id}:invalid:{error}")
        for item in pose_pairwise_comparisons(base_transforms):
            pair = f"{item['camera_a']}__vs__{item['camera_b']}"
            base_pose_values[pair].append(item)
        for item in pose_pairwise_comparisons(local_transforms):
            pair = f"{item['camera_a']}__vs__{item['camera_b']}"
            local_pose_values[pair].append(item)

    base_translation, base_rotation, base_different = _aggregate_pose_comparisons(
        base_pose_values
    )
    local_translation, local_rotation, local_different = _aggregate_pose_comparisons(
        local_pose_values
    )
    link_sets = {camera: sorted(values) for camera, values in gt_links.items()}
    links_different = len({tuple(values) for values in link_sets.values()}) > 1
    transforms_different = base_different or local_different or links_different
    all_identical_count = sum(
        item["all_cameras_byte_identical"] for item in identity_frames
    )
    all_frames_identical = all_identical_count == len(frame_ids) and bool(frame_ids)
    diagnosis = diagnose_dataset(
        input_complete=not missing_rgb and not missing_transforms,
        all_frames_all_cameras_byte_identical=all_frames_identical,
        affected_frame_count=len(identity_frames),
        transforms_different=transforms_different,
    )
    detection_root = (
        Path(output_root_for_detections)
        if output_root_for_detections is not None
        else root.parent / "outputs"
    )
    return {
        **diagnosis,
        "sampled_frame_count": len(frame_ids),
        "sampled_frame_ids": frame_ids,
        "camera_names": camera_names,
        "dataset_metadata": {
            "session_camera_count": session_summary.get("camera_count"),
            "session_frame_count": session_summary.get(
                "frame_count", session_summary.get("frame_count_so_far")
            ),
            "candidate_link_count": candidate_links.get("link_count"),
        },
        "input_completeness": {
            "complete": not missing_rgb and not missing_transforms,
            "missing_or_unreadable_rgb": missing_rgb,
            "missing_or_invalid_transforms": missing_transforms,
        },
        "rgb_same_frame_identity": {
            "all_sampled_frames_byte_identical_across_all_cameras": all_frames_identical,
            "all_cameras_identical_frame_count": all_identical_count,
            "affected_frame_count": len(identity_frames),
            "examples": identity_frames[:10],
        },
        "rgb_same_frame_pixel_differences": {
            "pairwise_summary": pairwise_pixel_summary
        },
        "rgb_across_frame_changes": {"per_camera_summary": temporal_summaries},
        "camera_transform_distinctness": {
            "transforms_different_across_cameras": transforms_different,
            "pairwise_translation_difference_m": base_translation,
            "pairwise_rotation_difference_deg": base_rotation,
            "gt_links": link_sets,
            "t_link_camera_gt_pairwise_translation_difference_m": local_translation,
            "t_link_camera_gt_pairwise_rotation_difference_deg": local_rotation,
        },
        "detection_output_consistency": inspect_detection_outputs(
            detection_root, camera_names
        ),
    }

