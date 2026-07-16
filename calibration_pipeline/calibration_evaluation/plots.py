"""Publication-friendly matplotlib plots for one calibration experiment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _short_name(name: str) -> str:
    match = re.search(r"Cam(?:era)?[_-]?(\d+)$", name, re.IGNORECASE)
    return f"Cam{match.group(1)}" if match else name


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _bar_plot(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    path: Path,
    *,
    title: str,
    ylabel: str,
    color: str = "#4472C4",
) -> str | None:
    selected = [row for row in rows if row.get(field) is not None]
    if not selected:
        return None
    labels = [_short_name(str(row["camera_name"])) for row in selected]
    values = [float(row[field]) for row in selected]
    fig, axis = plt.subplots(figsize=(max(6.0, len(labels) * 1.25), 4.2))
    axis.bar(labels, values, color=color)
    axis.set_title(title)
    axis.set_xlabel("Camera")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    return _save(fig, path)


def _grouped_bar(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[tuple[str, str]],
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> str | None:
    selected = [
        row for row in rows if any(row.get(field) is not None for field, _ in fields)
    ]
    if not selected:
        return None
    labels = [_short_name(str(row["camera_name"])) for row in selected]
    x = np.arange(len(labels))
    width = 0.8 / len(fields)
    fig, axis = plt.subplots(figsize=(max(6.0, len(labels) * 1.3), 4.2))
    for index, (field, legend) in enumerate(fields):
        values = [
            np.nan if row.get(field) is None else float(row[field]) for row in selected
        ]
        axis.bar(x + (index - (len(fields) - 1) / 2) * width, values, width, label=legend)
    axis.set_xticks(x, labels, rotation=25)
    axis.set_title(title)
    axis.set_xlabel("Camera")
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def _heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    path: Path,
    *,
    title: str,
    colorbar_label: str,
) -> str | None:
    if matrix.size == 0 or np.all(np.isnan(matrix)):
        return None
    fig, axis = plt.subplots(
        figsize=(max(6.0, len(column_labels) * 1.1), max(4.5, len(row_labels) * 0.8))
    )
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="viridis", aspect="auto")
    axis.set_title(title)
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    color="white" if value > np.nanmean(matrix) else "black",
                    fontsize=8,
                )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    return _save(fig, path)


def _link_score_heatmap(
    ranking: Sequence[Mapping[str, Any]], path: Path
) -> str | None:
    cameras = sorted({str(row["camera_name"]) for row in ranking})
    links = sorted({str(row.get("link_name", "")) for row in ranking})
    matrix = np.full((len(cameras), len(links)), np.nan)
    camera_index = {name: index for index, name in enumerate(cameras)}
    link_index = {name: index for index, name in enumerate(links)}
    for row in ranking:
        if row.get("score") is not None:
            matrix[camera_index[str(row["camera_name"])], link_index[str(row["link_name"])]] = float(row["score"])
    return _heatmap(
        matrix,
        [_short_name(name) for name in cameras],
        links,
        path,
        title="Candidate link scores (lower is better)",
        colorbar_label="SE(3) residual score",
    )


def _relative_heatmap(
    rows: Sequence[Mapping[str, Any]], field: str, path: Path, *, title: str, unit: str
) -> str | None:
    cameras = sorted(
        {str(row["target_camera"]) for row in rows}
        | {str(row["source_camera"]) for row in rows}
    )
    matrix = np.full((len(cameras), len(cameras)), np.nan)
    index = {name: value for value, name in enumerate(cameras)}
    for row in rows:
        if row.get(field) is not None:
            matrix[index[str(row["target_camera"])], index[str(row["source_camera"])]] = float(row[field])
    return _heatmap(
        matrix,
        [_short_name(name) for name in cameras],
        [_short_name(name) for name in cameras],
        path,
        title=title,
        colorbar_label=unit,
    )


def generate_plots(
    summary: Mapping[str, Any], plot_dir: Path, warnings: list[str]
) -> list[str]:
    """Generate every plot supported by the available single-run metrics."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    cameras = list(summary.get("per_camera", []))
    ranking = list(summary.get("link_ranking", []))
    pairwise = list(summary.get("pairwise_relative_metrics", []))
    specifications = [
        (
            "detection_valid_ratio.png",
            lambda path: _bar_plot(
                cameras,
                "detection_valid_ratio",
                path,
                title="ChArUco detection valid ratio",
                ylabel="Valid ratio",
            ),
        ),
        (
            "charuco_corner_count.png",
            lambda path: _grouped_bar(
                cameras,
                [
                    ("detection_mean_charuco_corners", "Mean"),
                    ("detection_median_charuco_corners", "Median"),
                ],
                path,
                title="Detected ChArUco corner count",
                ylabel="Corners per frame",
            ),
        ),
        (
            "board_pose_ray_error.png",
            lambda path: _bar_plot(
                cameras,
                "board_pose_mean_ray_error_deg",
                path,
                title="Board-pose mean ray error",
                ylabel="Angular error (deg)",
            ),
        ),
        (
            "link_score_margin.png",
            lambda path: _bar_plot(
                cameras,
                "link_score_margin",
                path,
                title="Best-link score margin",
                ylabel="Score margin",
            ),
        ),
        (
            "observability_rank.png",
            lambda path: _grouped_bar(
                cameras,
                [("observability_rank", "Rank"), ("observability_max_rank", "Maximum")],
                path,
                title="Independent mount observability",
                ylabel="Jacobian rank",
            ),
        ),
        (
            "link_score_heatmap.png",
            lambda path: _link_score_heatmap(ranking, path),
        ),
        (
            "t_link_camera_translation_error.png",
            lambda path: _bar_plot(
                cameras,
                "gt_T_link_camera_translation_error_m",
                path,
                title="Static mount translation error",
                ylabel="Translation error (m)",
                color="#C55A11",
            ),
        ),
        (
            "t_link_camera_rotation_error.png",
            lambda path: _bar_plot(
                cameras,
                "gt_T_link_camera_rotation_error_deg",
                path,
                title="Static mount rotation error",
                ylabel="Rotation error (deg)",
                color="#C55A11",
            ),
        ),
        (
            "t_base_cam_translation_error.png",
            lambda path: _bar_plot(
                cameras,
                "gt_T_base_cam_translation_error_mean_m",
                path,
                title="Runtime absolute camera-pose translation error",
                ylabel="Mean translation error (m)",
                color="#70AD47",
            ),
        ),
        (
            "t_base_cam_rotation_error.png",
            lambda path: _bar_plot(
                cameras,
                "gt_T_base_cam_rotation_error_mean_deg",
                path,
                title="Runtime absolute camera-pose rotation error",
                ylabel="Mean rotation error (deg)",
                color="#70AD47",
            ),
        ),
        (
            "relative_pose_translation_heatmap.png",
            lambda path: _relative_heatmap(
                pairwise,
                "relative_translation_error_mean_m",
                path,
                title="Relative target-to-source translation error",
                unit="Mean error (m)",
            ),
        ),
        (
            "relative_pose_rotation_heatmap.png",
            lambda path: _relative_heatmap(
                pairwise,
                "relative_rotation_error_mean_deg",
                path,
                title="Relative target-to-source rotation error",
                unit="Mean error (deg)",
            ),
        ),
        (
            "recovery_source_by_camera.png",
            lambda path: _bar_plot(
                [
                    {
                        **row,
                        "recovery_source_numeric": (
                            1.0 if row.get("shared_board_recovery_used") else 0.0
                        ),
                    }
                    for row in cameras
                ],
                "recovery_source_numeric",
                path,
                title="Final calibration source (1 = shared recovery)",
                ylabel="Source category",
                color="#8064A2",
            ),
        ),
    ]
    generated = []
    for filename, generator in specifications:
        path = plot_dir / filename
        try:
            result = generator(path)
        except Exception as error:  # plotting must not prevent the report
            warnings.append(f"Plot failed for {filename}: {type(error).__name__}: {error}")
            plt.close("all")
            continue
        if result is None:
            warnings.append(f"Plot skipped because metrics were unavailable: {filename}")
        else:
            generated.append(result)
    return generated
