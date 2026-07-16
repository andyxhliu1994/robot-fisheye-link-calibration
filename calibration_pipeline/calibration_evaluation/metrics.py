"""Metric collection for one completed calibration experiment."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import numpy as np

from ..depth_model_compat import (
    compute_relative_camera_transforms,
    parse_depth_model_transform,
)
from ..dataset_loader import load_json_document
from ..se3_utils import rotation_error_deg


PER_CAMERA_COLUMNS = [
    "camera_name",
    "attached_link",
    "attached_link_name",
    "calibration_source",
    "confidence",
    "warnings",
    "detection_total_frames",
    "detection_valid_frames",
    "detection_valid_ratio",
    "detection_mean_markers",
    "detection_median_markers",
    "detection_mean_charuco_corners",
    "detection_median_charuco_corners",
    "detection_failure_reasons",
    "board_pose_total_frames",
    "board_pose_valid_frames",
    "board_pose_valid_ratio",
    "board_pose_mean_ray_error_deg",
    "board_pose_median_ray_error_deg",
    "board_pose_max_ray_error_deg",
    "board_pose_mean_corner_count",
    "board_pose_median_corner_count",
    "board_pose_failure_reasons",
    "link_best_score",
    "link_second_best_score",
    "link_score_margin",
    "link_board_consistency_translation_mean_m",
    "link_board_consistency_translation_median_m",
    "link_board_consistency_translation_max_m",
    "link_board_consistency_rotation_mean_deg",
    "link_board_consistency_rotation_median_deg",
    "link_board_consistency_rotation_max_deg",
    "observability_rank",
    "observability_max_rank",
    "motion_limited",
    "shared_board_recovery_used",
    "shared_board_recovery_confidence",
    "recovery_consistency_translation_mean_m",
    "recovery_consistency_rotation_mean_deg",
    "recovery_warning",
    "gt_attached_link_correct",
    "gt_T_link_camera_translation_error_m",
    "gt_T_link_camera_rotation_error_deg",
    "gt_T_base_cam_translation_error_mean_m",
    "gt_T_base_cam_translation_error_median_m",
    "gt_T_base_cam_translation_error_max_m",
    "gt_T_base_cam_rotation_error_mean_deg",
    "gt_T_base_cam_rotation_error_median_deg",
    "gt_T_base_cam_rotation_error_max_deg",
]

GT_FREE_COLUMNS = [
    column for column in PER_CAMERA_COLUMNS if not column.startswith("gt_")
]

LINK_RANKING_COLUMNS = [
    "camera_name",
    "rank",
    "link_name",
    "link_path_rel",
    "success",
    "score",
    "translation_mean_m",
    "translation_median_m",
    "translation_max_m",
    "rotation_mean_deg",
    "rotation_median_deg",
    "rotation_max_deg",
    "num_valid_frames",
]

PAIRWISE_COLUMNS = [
    "target_camera",
    "source_camera",
    "pair_count",
    "relative_translation_error_mean_m",
    "relative_translation_error_median_m",
    "relative_translation_error_max_m",
    "relative_rotation_error_mean_deg",
    "relative_rotation_error_median_deg",
    "relative_rotation_error_max_deg",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else row.get(column)
                    for column in columns
                }
            )


def _load_json(
    path: Path,
    warnings: list[str],
    *,
    strict: bool,
    label: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        message = f"Missing {label}: {path}"
        if strict:
            raise FileNotFoundError(message)
        warnings.append(message)
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Could not read {label} at {path}: {error}"
        if strict:
            raise ValueError(message) from error
        warnings.append(message)
        return None
    if not isinstance(document, dict):
        message = f"Expected a JSON object for {label}: {path}"
        if strict:
            raise ValueError(message)
        warnings.append(message)
        return None
    return document


def _load_dataset_json(
    path: Path, warnings: list[str], *, label: str
) -> dict[str, Any] | None:
    if not path.is_file():
        warnings.append(f"Missing {label}: {path}")
        return None
    try:
        document, _ = load_json_document(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        warnings.append(f"Could not read {label} at {path}: {error}")
        return None
    if not isinstance(document, dict):
        warnings.append(f"Expected a JSON object for {label}: {path}")
        return None
    return document


def _load_jsonl(path: Path, warnings: list[str], *, strict: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        message = f"Missing JSONL input: {path}"
        if strict:
            raise FileNotFoundError(message)
        warnings.append(message)
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"Invalid JSONL record at {path}:{line_number}: {error}"
            if strict:
                raise ValueError(message) from error
            warnings.append(message)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": float(fmean(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(fmean(present)) if present else None


def _metric_value(record: Mapping[str, Any] | None, metric: str, field: str) -> Any:
    if not record:
        return None
    value = record.get(metric)
    return value.get(field) if isinstance(value, Mapping) else None


def _json_counter(records: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(
        str(record.get("reason", "unknown"))
        for record in records
        if not record.get("valid", False)
    )
    return json.dumps(dict(counts), sort_keys=True)


def _detection_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid", False)]
    markers = [float(record.get("marker_count", 0)) for record in records]
    corners = [float(record.get("charuco_corner_count", 0)) for record in records]
    return {
        "detection_total_frames": len(records),
        "detection_valid_frames": len(valid),
        "detection_valid_ratio": len(valid) / len(records) if records else None,
        "detection_mean_markers": _mean(markers),
        "detection_median_markers": float(median(markers)) if markers else None,
        "detection_mean_charuco_corners": _mean(corners),
        "detection_median_charuco_corners": (
            float(median(corners)) if corners else None
        ),
        "detection_failure_reasons": _json_counter(records),
    }


def _board_pose_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid", False)]
    ray_errors = [
        float(record["mean_ray_error_deg"])
        for record in valid
        if record.get("mean_ray_error_deg") is not None
    ]
    corner_counts = [
        float(record.get("charuco_corner_count", 0)) for record in valid
    ]
    ray = _stats(ray_errors)
    return {
        "board_pose_total_frames": len(records),
        "board_pose_valid_frames": len(valid),
        "board_pose_valid_ratio": len(valid) / len(records) if records else None,
        "board_pose_mean_ray_error_deg": ray["mean"],
        "board_pose_median_ray_error_deg": ray["median"],
        "board_pose_max_ray_error_deg": ray["max"],
        "board_pose_mean_corner_count": _mean(corner_counts),
        "board_pose_median_corner_count": (
            float(median(corner_counts)) if corner_counts else None
        ),
        "board_pose_failure_reasons": _json_counter(records),
    }


def _record_map(document: Mapping[str, Any] | None, key: str) -> dict[str, dict[str, Any]]:
    if not document:
        return {}
    return {
        str(record["camera_name"]): dict(record)
        for record in document.get(key, [])
        if isinstance(record, Mapping) and record.get("camera_name") is not None
    }


def _infer_fov(camera_names: Sequence[str], compatibility: Mapping[str, Any] | None) -> Any:
    if compatibility:
        conversion = str(compatibility.get("conversion_name", ""))
        reverse = {
            "FisheyeConversions_2": 180,
            "FisheyeConversions_3": 210,
            "FisheyeConversions_4": 240,
        }
        if conversion in reverse:
            return reverse[conversion]
    values = {
        int(match.group(1))
        for name in camera_names
        if (match := re.search(r"Fisheye(\d+)", name)) is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def _load_transform(path: Path) -> np.ndarray:
    document = json.loads(path.read_text(encoding="utf-8"))
    return parse_depth_model_transform(document)


def _pairwise_relative_metrics(
    dataset_root: Path,
    outputs_root: Path,
    camera_names: Sequence[str],
    warnings: list[str],
    *,
    max_pairs: int | None,
    strict: bool,
) -> list[dict[str, Any]]:
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    predicted_root = outputs_root / "depth_model_compat" / "transforms"
    gt_root = dataset_root / "cameras"
    if not predicted_root.is_dir():
        message = f"Missing calibrated per-frame transforms: {predicted_root}"
        if strict:
            raise FileNotFoundError(message)
        warnings.append(message)
        return []
    rows = []
    for target_name in camera_names:
        for source_name in camera_names:
            if target_name == source_name:
                continue
            target_pred = predicted_root / target_name
            source_pred = predicted_root / source_name
            target_gt = gt_root / target_name / "transform"
            source_gt = gt_root / source_name / "transform"
            directories = (target_pred, source_pred, target_gt, source_gt)
            if not all(path.is_dir() for path in directories):
                message = (
                    "Missing predicted/GT transform directory for pair "
                    f"target={target_name}, source={source_name}"
                )
                if strict:
                    raise FileNotFoundError(message)
                warnings.append(message)
                continue
            frame_ids = set(path.stem for path in target_pred.glob("frame_*.json"))
            for directory in (source_pred, target_gt, source_gt):
                frame_ids &= {path.stem for path in directory.glob("frame_*.json")}
            selected = sorted(frame_ids)
            if max_pairs is not None:
                selected = selected[:max_pairs]
            translation_errors = []
            rotation_errors = []
            for frame_id in selected:
                pred_target = _load_transform(target_pred / f"{frame_id}.json")
                pred_source = _load_transform(source_pred / f"{frame_id}.json")
                gt_target = _load_transform(target_gt / f"{frame_id}.json")
                gt_source = _load_transform(source_gt / f"{frame_id}.json")
                pred_src_tgt, _ = compute_relative_camera_transforms(
                    pred_target, pred_source
                )
                gt_src_tgt, _ = compute_relative_camera_transforms(
                    gt_target, gt_source
                )
                translation_errors.append(
                    float(np.linalg.norm(pred_src_tgt[:3, 3] - gt_src_tgt[:3, 3]))
                )
                rotation_errors.append(
                    rotation_error_deg(pred_src_tgt[:3, :3], gt_src_tgt[:3, :3])
                )
            translation = _stats(translation_errors)
            rotation = _stats(rotation_errors)
            rows.append(
                {
                    "target_camera": target_name,
                    "source_camera": source_name,
                    "pair_count": len(selected),
                    "relative_translation_error_mean_m": translation["mean"],
                    "relative_translation_error_median_m": translation["median"],
                    "relative_translation_error_max_m": translation["max"],
                    "relative_rotation_error_mean_deg": rotation["mean"],
                    "relative_rotation_error_median_deg": rotation["median"],
                    "relative_rotation_error_max_deg": rotation["max"],
                }
            )
    return rows


def _weighted_pair_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    weighted = [
        (float(row[field]), int(row["pair_count"]))
        for row in rows
        if row.get(field) is not None and int(row.get("pair_count", 0)) > 0
    ]
    total = sum(count for _, count in weighted)
    return (
        sum(value * count for value, count in weighted) / total if total else None
    )


def collect_calibration_metrics(
    dataset_root: Path,
    outputs_root: Path,
    *,
    evaluate_gt: bool,
    camera_name: str | None = None,
    max_pairs: int | None = None,
    strict: bool = False,
    experiment_id: str | None = None,
    setup_name: str | None = None,
    fov: str | int | None = None,
) -> dict[str, Any]:
    """Collect one experiment's GT-free and optional evaluation-only metrics."""
    warnings: list[str] = []
    paths = {
        "detections": outputs_root / "detections",
        "board_poses": outputs_root / "board_poses",
        "link_calibration": (
            outputs_root / "link_calibration" / "link_calibration_summary.json"
        ),
        "shared_board_recovery": (
            outputs_root
            / "shared_board_recovery"
            / "shared_board_recovery_summary.json"
        ),
        "final_calibration": (
            outputs_root / "final_calibration" / "final_calibration.json"
        ),
        "static_validation": (
            outputs_root
            / "final_calibration"
            / "final_static_calibration_validation.json"
        ),
        "camera_pose_validation": (
            outputs_root
            / "final_calibration"
            / "final_camera_pose_validation.json"
        ),
        "depth_pose_validation": (
            outputs_root
            / "depth_model_compat"
            / "depth_model_pose_validation.json"
        ),
        "compatibility_report": (
            outputs_root
            / "depth_model_compat"
            / "depth_model_compatibility_report.json"
        ),
    }
    final = _load_json(
        paths["final_calibration"], warnings, strict=strict, label="final calibration"
    )
    link = _load_json(
        paths["link_calibration"], warnings, strict=strict, label="link calibration"
    )
    recovery = _load_json(
        paths["shared_board_recovery"],
        warnings,
        strict=strict,
        label="shared-board recovery",
    )
    compatibility = _load_json(
        paths["compatibility_report"],
        warnings,
        strict=strict,
        label="depth compatibility report",
    )
    static_validation = None
    pose_validation = None
    depth_validation = None
    if evaluate_gt:
        static_validation = _load_json(
            paths["static_validation"],
            warnings,
            strict=strict,
            label="static GT validation",
        )
        pose_validation = _load_json(
            paths["camera_pose_validation"],
            warnings,
            strict=strict,
            label="runtime camera-pose GT validation",
        )
        depth_validation = _load_json(
            paths["depth_pose_validation"],
            warnings,
            strict=strict,
            label="depth-model pose GT validation",
        )

    final_by_camera = _record_map(final, "cameras")
    link_by_camera = _record_map(link, "cameras")
    recovery_by_camera = _record_map(recovery, "camera_results")
    static_by_camera = _record_map(static_validation, "cameras")
    runtime_by_camera = _record_map(pose_validation, "cameras")
    if depth_validation:
        depth_absolute = _record_map(depth_validation.get("absolute_pose"), "cameras")
    else:
        depth_absolute = {}
    names = sorted(
        set(final_by_camera)
        | set(link_by_camera)
        | {path.stem for path in paths["detections"].glob("*.jsonl")}
        | {path.stem for path in paths["board_poses"].glob("*.jsonl")}
    )
    if camera_name is not None:
        if camera_name not in names:
            raise ValueError(f"Camera {camera_name!r} was not found in evaluation inputs")
        names = [camera_name]
    if not names:
        warnings.append("No cameras were found in the available calibration outputs.")

    per_camera = []
    for name in names:
        final_record = final_by_camera.get(name, {})
        link_record = link_by_camera.get(name, {})
        recovery_record = recovery_by_camera.get(name, {})
        static_record = static_by_camera.get(name, {})
        runtime_record = runtime_by_camera.get(name) or depth_absolute.get(name, {})
        detection_records = _load_jsonl(
            paths["detections"] / f"{name}.jsonl", warnings, strict=strict
        )
        board_records = _load_jsonl(
            paths["board_poses"] / f"{name}.jsonl", warnings, strict=strict
        )
        observability = final_record.get("observability", {})
        translation_metric = runtime_record.get("translation_error_m", {})
        rotation_metric = runtime_record.get("rotation_error_deg", {})
        final_warnings = [str(item) for item in final_record.get("warnings", [])]
        row = {
            "camera_name": name,
            "attached_link": final_record.get("attached_link")
            or link_record.get("best_link_path_rel"),
            "attached_link_name": final_record.get("attached_link_name")
            or link_record.get("best_link"),
            "calibration_source": final_record.get("calibration_source"),
            "confidence": final_record.get("confidence"),
            "warnings": " | ".join(final_warnings),
            **_detection_metrics(detection_records),
            **_board_pose_metrics(board_records),
            "link_best_score": link_record.get("best_score"),
            "link_second_best_score": link_record.get("second_best_score"),
            "link_score_margin": link_record.get("score_margin"),
            "link_board_consistency_translation_mean_m": _metric_value(
                link_record, "board_consistency_translation_m", "mean"
            ),
            "link_board_consistency_translation_median_m": _metric_value(
                link_record, "board_consistency_translation_m", "median"
            ),
            "link_board_consistency_translation_max_m": _metric_value(
                link_record, "board_consistency_translation_m", "max"
            ),
            "link_board_consistency_rotation_mean_deg": _metric_value(
                link_record, "board_consistency_rotation_deg", "mean"
            ),
            "link_board_consistency_rotation_median_deg": _metric_value(
                link_record, "board_consistency_rotation_deg", "median"
            ),
            "link_board_consistency_rotation_max_deg": _metric_value(
                link_record, "board_consistency_rotation_deg", "max"
            ),
            "observability_rank": observability.get("rank")
            if observability
            else link_record.get("observability_rank"),
            "observability_max_rank": observability.get("max_rank")
            if observability
            else link_record.get("observability_parameter_count"),
            "motion_limited": observability.get("motion_limited")
            if observability
            else (
                not link_record.get("mount_fully_observable")
                if link_record
                else None
            ),
            "shared_board_recovery_used": (
                final_record.get("calibration_source") == "shared_board_recovery"
            ),
            "shared_board_recovery_confidence": recovery_record.get("confidence"),
            "recovery_consistency_translation_mean_m": _metric_value(
                recovery_record, "recovery_consistency_translation_m", "mean"
            ),
            "recovery_consistency_rotation_mean_deg": _metric_value(
                recovery_record, "recovery_consistency_rotation_deg", "mean"
            ),
            "recovery_warning": recovery_record.get("warning"),
            "gt_attached_link_correct": static_record.get("gt_attached_link_correct")
            if evaluate_gt
            else None,
            "gt_T_link_camera_translation_error_m": static_record.get(
                "gt_T_link_camera_translation_error_m"
            )
            if evaluate_gt
            else None,
            "gt_T_link_camera_rotation_error_deg": static_record.get(
                "gt_T_link_camera_rotation_error_deg"
            )
            if evaluate_gt
            else None,
            "gt_T_base_cam_translation_error_mean_m": translation_metric.get("mean")
            if evaluate_gt
            else None,
            "gt_T_base_cam_translation_error_median_m": translation_metric.get("median")
            if evaluate_gt
            else None,
            "gt_T_base_cam_translation_error_max_m": translation_metric.get("max")
            if evaluate_gt
            else None,
            "gt_T_base_cam_rotation_error_mean_deg": rotation_metric.get("mean")
            if evaluate_gt
            else None,
            "gt_T_base_cam_rotation_error_median_deg": rotation_metric.get("median")
            if evaluate_gt
            else None,
            "gt_T_base_cam_rotation_error_max_deg": rotation_metric.get("max")
            if evaluate_gt
            else None,
        }
        per_camera.append(row)

    link_ranking = []
    for name in names:
        for hypothesis in link_by_camera.get(name, {}).get("hypotheses", []):
            link_ranking.append(
                {
                    "camera_name": name,
                    **{column: hypothesis.get(column) for column in LINK_RANKING_COLUMNS[1:]},
                }
            )
    link_ranking.sort(key=lambda row: (row["camera_name"], row.get("rank") or 9999))

    pairwise = []
    if evaluate_gt and len(names) >= 2:
        pairwise = _pairwise_relative_metrics(
            dataset_root,
            outputs_root,
            names,
            warnings,
            max_pairs=max_pairs,
            strict=strict,
        )

    confidence_counts = Counter(str(row.get("confidence") or "unknown") for row in per_camera)
    anchor_agreement = recovery.get("anchor_agreement", {}) if recovery else {}
    global_gt_free = {
        "camera_count": len(per_camera),
        "final_calibration_available": final is not None,
        "confidence_counts": dict(confidence_counts),
        "motion_limited_camera_count": sum(bool(row.get("motion_limited")) for row in per_camera),
        "recovered_camera_count": sum(
            bool(row.get("shared_board_recovery_used")) for row in per_camera
        ),
        "warning_count": sum(bool(row.get("warnings")) for row in per_camera)
        + len(recovery.get("warnings", []) if recovery else []),
        "mean_detection_valid_ratio": _mean(
            [row.get("detection_valid_ratio") for row in per_camera]
        ),
        "mean_board_pose_ray_error_deg": _mean(
            [row.get("board_pose_mean_ray_error_deg") for row in per_camera]
        ),
        "mean_link_score_margin": _mean(
            [row.get("link_score_margin") for row in per_camera]
        ),
        "shared_board_anchor_camera_count": len(
            recovery.get("anchor_cameras", []) if recovery else []
        ),
        "shared_board_motion_limited_camera_count": len(
            recovery.get("motion_limited_cameras", []) if recovery else []
        ),
        "shared_board_anchor_translation_mean_m": _metric_value(
            anchor_agreement, "translation_m", "mean"
        ),
        "shared_board_anchor_rotation_mean_deg": _metric_value(
            anchor_agreement, "rotation_deg", "mean"
        ),
        "compatibility_parser_smoke_test_passed": (
            compatibility.get("parser_compatibility_smoke_test_passed")
            if compatibility
            else None
        ),
        "relative_pose_smoke_test_passed": (
            compatibility.get("relative_pose_computation_smoke_test_passed")
            if compatibility
            else None
        ),
        "compatibility_exported_transform_count": (
            compatibility.get("total_transform_json_files_generated")
            if compatibility
            else None
        ),
        "compatibility_manifest_sample_count": (
            compatibility.get("sample_manifest_sample_count")
            if compatibility
            else None
        ),
    }
    known_links = [
        row["gt_attached_link_correct"]
        for row in per_camera
        if row.get("gt_attached_link_correct") is not None
    ]
    global_gt = None
    if evaluate_gt:
        global_gt = {
            "link_association_top1_accuracy": (
                sum(bool(value) for value in known_links) / len(known_links)
                if known_links
                else None
            ),
            "mean_static_translation_error_m": _mean(
                [row.get("gt_T_link_camera_translation_error_m") for row in per_camera]
            ),
            "mean_static_rotation_error_deg": _mean(
                [row.get("gt_T_link_camera_rotation_error_deg") for row in per_camera]
            ),
            "mean_runtime_pose_translation_error_m": _mean(
                [row.get("gt_T_base_cam_translation_error_mean_m") for row in per_camera]
            ),
            "mean_runtime_pose_rotation_error_deg": _mean(
                [row.get("gt_T_base_cam_rotation_error_mean_deg") for row in per_camera]
            ),
            "mean_relative_translation_error_m": _weighted_pair_mean(
                pairwise, "relative_translation_error_mean_m"
            ),
            "mean_relative_rotation_error_deg": _weighted_pair_mean(
                pairwise, "relative_rotation_error_mean_deg"
            ),
            "relative_pair_count": sum(int(row.get("pair_count", 0)) for row in pairwise),
        }
    session = _load_dataset_json(
        dataset_root / "session_summary.json", warnings, label="session summary"
    )
    camera_config = _load_dataset_json(
        dataset_root / "camera_model_config.json",
        warnings,
        label="camera model config",
    )
    board_config = _load_dataset_json(
        dataset_root / "charuco_board_config.json", warnings, label="board config"
    )
    evaluated_frame_count = None
    if pose_validation:
        evaluated_frame_count = pose_validation.get("frame_count")
    elif depth_validation:
        evaluated_frame_count = depth_validation.get("absolute_pose", {}).get(
            "frames_evaluated_total"
        )
    return {
        "schema_version": 1,
        "report_type": "single_experiment_calibration_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "dataset": str(dataset_root),
            "outputs_root": str(outputs_root),
            **{key: str(value) for key, value in paths.items()},
        },
        "output_paths": {},
        "gt_evaluation_enabled": evaluate_gt,
        "gt_metrics_available": bool(
            global_gt
            and any(value is not None for key, value in global_gt.items() if key != "relative_pair_count")
        ),
        "experiment_id": experiment_id
        or (str(session.get("session_name")) if session and session.get("session_name") else None),
        "setup_name": setup_name
        or (str(session.get("setup_id")) if session and session.get("setup_id") else None),
        "fov": fov if fov is not None else _infer_fov(names, compatibility),
        "camera_model": camera_config.get("default_camera_model") if camera_config else None,
        "adapter_metadata": (
            final.get("frame_adapters", {}).get("camera_ray_to_camera_pose_adapter")
            if final
            else None
        ),
        "board_config": {
            key: board_config.get(key)
            for key in (
                "dictionary",
                "squares_x",
                "squares_y",
                "square_length_m",
                "marker_length_m",
            )
        }
        if board_config
        else None,
        "dataset_frame_count": session.get("frame_count_so_far") if session else None,
        "evaluated_frame_count": evaluated_frame_count,
        "camera_count": len(per_camera),
        "per_camera": per_camera,
        "link_ranking": link_ranking,
        "pairwise_relative_metrics": pairwise,
        "global_gt_free_metrics": global_gt_free,
        "global_gt_based_metrics": global_gt,
        "warnings": warnings,
        "plot_paths": [],
        "csv_paths": {},
        "report_path": None,
    }
