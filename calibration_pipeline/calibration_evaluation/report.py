"""Human-readable Markdown report for one calibration experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def _percentage(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def build_report(summary: Mapping[str, Any]) -> str:
    cameras = list(summary.get("per_camera", []))
    gt_free = summary.get("global_gt_free_metrics", {})
    gt = summary.get("global_gt_based_metrics") or {}
    confidence = gt_free.get("confidence_counts", {})
    lines = [
        "# Single-experiment calibration evaluation",
        "",
        "## Run metadata",
        "",
        f"- Dataset: `{summary.get('input_paths', {}).get('dataset', 'n/a')}`",
        f"- Calibration outputs: `{summary.get('input_paths', {}).get('outputs_root', 'n/a')}`",
        f"- Evaluation mode: {'GT enabled' if summary.get('gt_evaluation_enabled') else 'GT disabled'}",
        f"- GT metrics available: {_fmt(summary.get('gt_metrics_available'))}",
        f"- Generated UTC: `{summary.get('generated_utc', 'n/a')}`",
        f"- Experiment ID: `{summary.get('experiment_id') or 'n/a'}`",
        f"- Setup: `{summary.get('setup_name') or 'n/a'}`",
        f"- FOV: `{summary.get('fov') if summary.get('fov') is not None else 'n/a'}`",
        f"- Camera count: {summary.get('camera_count', 0)}",
        "",
        "## Executive summary",
        "",
        f"- Final calibration available: {_fmt(gt_free.get('final_calibration_available'))}",
        f"- Confidence: high={confidence.get('high', 0)}, medium={confidence.get('medium', 0)}, low={confidence.get('low', 0)}, unknown={confidence.get('unknown', 0)}",
        f"- Motion-limited cameras: {gt_free.get('motion_limited_camera_count', 0)}",
        f"- Cameras using shared-board recovery: {gt_free.get('recovered_camera_count', 0)}",
        f"- Reported calibration warnings: {gt_free.get('warning_count', 0)}",
    ]
    warnings = list(summary.get("warnings", []))
    if warnings:
        lines.extend(["", "Major/reporting warnings:", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.extend(["", "No reporting-input warnings were recorded."])

    lines.extend(
        [
            "",
            "## Per-camera summary",
            "",
            "| Camera | Final link | Source | Confidence | Detection valid | Ray error (deg) | Score margin | Observability | Static GT (m / deg) | Runtime GT (m / deg) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cameras:
        static = (
            f"{_fmt(row.get('gt_T_link_camera_translation_error_m'), 5)} / "
            f"{_fmt(row.get('gt_T_link_camera_rotation_error_deg'), 3)}"
        )
        runtime = (
            f"{_fmt(row.get('gt_T_base_cam_translation_error_mean_m'), 5)} / "
            f"{_fmt(row.get('gt_T_base_cam_rotation_error_mean_deg'), 3)}"
        )
        observability = (
            f"{_fmt(row.get('observability_rank'), 0)}/"
            f"{_fmt(row.get('observability_max_rank'), 0)}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("camera_name")),
                    _fmt(row.get("attached_link_name") or row.get("attached_link")),
                    _fmt(row.get("calibration_source")),
                    _fmt(row.get("confidence")),
                    _percentage(row.get("detection_valid_ratio")),
                    _fmt(row.get("board_pose_mean_ray_error_deg"), 4),
                    _fmt(row.get("link_score_margin"), 4),
                    observability,
                    static,
                    runtime,
                ]
            )
            + " |"
        )

    low_detection = [
        row["camera_name"]
        for row in cameras
        if row.get("detection_valid_ratio") is not None
        and float(row["detection_valid_ratio"]) < 0.5
    ]
    high_ray = [
        row["camera_name"]
        for row in cameras
        if row.get("board_pose_mean_ray_error_deg") is not None
        and float(row["board_pose_mean_ray_error_deg"]) > 0.1
    ]
    lines.extend(
        [
            "",
            "## Detection and board pose quality",
            "",
            f"Mean detection valid ratio across cameras: {_percentage(gt_free.get('mean_detection_valid_ratio'))}.",
            f"Mean per-camera board-pose ray error: {_fmt(gt_free.get('mean_board_pose_ray_error_deg'), 5)} deg.",
            f"Cameras below the 50% visibility review heuristic: {', '.join(low_detection) if low_detection else 'none'}.",
            f"Cameras above the 0.1 deg ray-error review heuristic: {', '.join(high_ray) if high_ray else 'none'}.",
            "These heuristics flag review candidates; they are not calibration pass/fail thresholds.",
            "",
            "## Link association and observability",
            "",
            f"Mean best-vs-second link score margin: {_fmt(gt_free.get('mean_link_score_margin'), 5)}.",
            f"Shared-board anchors: {gt_free.get('shared_board_anchor_camera_count', 0)}; motion-limited cameras: {gt_free.get('shared_board_motion_limited_camera_count', 0)}.",
            f"Shared-anchor agreement: {_fmt(gt_free.get('shared_board_anchor_translation_mean_m'), 6)} m and {_fmt(gt_free.get('shared_board_anchor_rotation_mean_deg'), 4)} deg mean.",
        ]
    )
    for row in cameras:
        lines.append(
            f"- {_fmt(row.get('camera_name'))}: link={_fmt(row.get('attached_link_name') or row.get('attached_link'))}, "
            f"margin={_fmt(row.get('link_score_margin'), 5)}, "
            f"rank={_fmt(row.get('observability_rank'), 0)}/{_fmt(row.get('observability_max_rank'), 0)}, "
            f"motion_limited={_fmt(row.get('motion_limited'))}, "
            f"source={_fmt(row.get('calibration_source'))}."
        )

    lines.extend(["", "## Static mount calibration accuracy", ""])
    if summary.get("gt_metrics_available") and gt.get("mean_static_translation_error_m") is not None:
        lines.extend(
            [
                f"Evaluation-only mean `T_link_camera` translation error: {_fmt(gt.get('mean_static_translation_error_m'), 6)} m.",
                f"Evaluation-only mean `T_link_camera` rotation error: {_fmt(gt.get('mean_static_rotation_error_deg'), 4)} deg.",
                f"Attached-link top-1 accuracy: {_percentage(gt.get('link_association_top1_accuracy'))}.",
            ]
        )
    else:
        lines.append("Static mount GT was not available or GT evaluation was disabled.")

    lines.extend(["", "## Runtime `T_base_cam` validation", ""])
    if summary.get("gt_metrics_available") and gt.get("mean_runtime_pose_translation_error_m") is not None:
        lines.extend(
            [
                f"Evaluation-only mean runtime translation error: {_fmt(gt.get('mean_runtime_pose_translation_error_m'), 6)} m.",
                f"Evaluation-only mean runtime rotation error: {_fmt(gt.get('mean_runtime_pose_rotation_error_deg'), 4)} deg.",
            ]
        )
    else:
        lines.append("Runtime camera-pose GT was not available or GT evaluation was disabled.")

    lines.extend(["", "## Relative pose validation for depth-model warping", ""])
    if summary.get("gt_metrics_available") and gt.get("mean_relative_translation_error_m") is not None:
        lines.extend(
            [
                f"Evaluated ordered source-target samples: {gt.get('relative_pair_count', 0)}.",
                f"Mean target-to-source translation error: {_fmt(gt.get('mean_relative_translation_error_m'), 6)} m.",
                f"Mean target-to-source rotation error: {_fmt(gt.get('mean_relative_rotation_error_deg'), 4)} deg.",
            ]
        )
    else:
        lines.append("Pairwise relative GT was not available or GT evaluation was disabled.")

    lines.extend(
        [
            "",
            "## GT-free quality indicators",
            "",
            "For real-robot runs, review detection coverage, ray error, link score margin, independent observability, board consistency, shared-anchor agreement, confidence, and warnings. These internal indicators do not become GT errors and must not be described as ground-truth accuracy.",
            f"Parser compatibility smoke test: {_fmt(gt_free.get('compatibility_parser_smoke_test_passed'))}.",
            f"Relative-pose computation smoke test: {_fmt(gt_free.get('relative_pose_smoke_test_passed'))}.",
            "",
            "## Output files",
            "",
        ]
    )
    output_paths = summary.get("output_paths", {})
    for name, path in output_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append(f"- plots generated: {len(summary.get('plot_paths', []))}")

    lines.extend(
        [
            "",
            "## Notes for interpretation",
            "",
            "- GT metrics are Unity/offline evaluation-only and do not affect transform selection or confidence.",
            "- `outputs/final_calibration/final_calibration.json` remains the deployment artifact.",
            "- The depth model consumes per-frame `T_base_cam`, not static `T_link_camera`.",
            "- Real-robot datasets may have no GT reports; GT-free metrics and warnings still remain useful.",
            "- This report summarizes one experiment only; it does not aggregate across setups, FOVs, or runs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(summary), encoding="utf-8")
