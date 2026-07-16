"""Generate a JSON/CSV/plot/Markdown evaluation for one calibration run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration_evaluation.metrics import (
    GT_FREE_COLUMNS,
    LINK_RANKING_COLUMNS,
    PAIRWISE_COLUMNS,
    PER_CAMERA_COLUMNS,
    collect_calibration_metrics,
    write_csv,
)
from .calibration_evaluation.report import write_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--fov")
    parser.add_argument("--setup-name")
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include evaluation-only GT reports and pairwise pose errors",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write available matplotlib plots (default: enabled)",
    )
    return parser.parse_args(argv)


def print_summary(summary: Mapping[str, Any]) -> None:
    output_paths = summary["output_paths"]
    print("Single-experiment calibration evaluation complete.")
    print(f"- Summary JSON: {output_paths['summary_json']}")
    print(f"- Per-camera CSV: {output_paths['per_camera_csv']}")
    print(f"- Link ranking CSV: {output_paths['link_ranking_csv']}")
    print(f"- Pairwise relative CSV: {output_paths['pairwise_relative_csv']}")
    print(f"- GT-free quality CSV: {output_paths['gt_free_quality_csv']}")
    print(f"- Report: {output_paths['report']}")
    print(f"- Plot directory: {output_paths['plot_directory']}")
    print(f"- Plots generated: {len(summary['plot_paths'])}")
    gt_free = summary["global_gt_free_metrics"]
    print(f"- Cameras: {summary['camera_count']}")
    print(
        "- Mean detection valid ratio: "
        f"{gt_free['mean_detection_valid_ratio']:.4f}"
        if gt_free.get("mean_detection_valid_ratio") is not None
        else "- Mean detection valid ratio: unavailable"
    )
    print(
        "- Mean board-pose ray error: "
        f"{gt_free['mean_board_pose_ray_error_deg']:.6f} deg"
        if gt_free.get("mean_board_pose_ray_error_deg") is not None
        else "- Mean board-pose ray error: unavailable"
    )
    print(
        f"- Motion-limited/recovered cameras: "
        f"{gt_free['motion_limited_camera_count']}/"
        f"{gt_free['recovered_camera_count']}"
    )
    gt = summary.get("global_gt_based_metrics") or {}
    if summary.get("gt_metrics_available"):
        if gt.get("mean_static_translation_error_m") is not None:
            print(
                "- Mean static error: "
                f"{gt['mean_static_translation_error_m']:.6f} m, "
                f"{gt['mean_static_rotation_error_deg']:.4f} deg"
            )
        if gt.get("mean_runtime_pose_translation_error_m") is not None:
            print(
                "- Mean runtime error: "
                f"{gt['mean_runtime_pose_translation_error_m']:.6f} m, "
                f"{gt['mean_runtime_pose_rotation_error_deg']:.4f} deg"
            )
        if gt.get("mean_relative_translation_error_m") is not None:
            print(
                "- Mean relative error: "
                f"{gt['mean_relative_translation_error_m']:.6f} m, "
                f"{gt['mean_relative_rotation_error_deg']:.4f} deg; "
                f"samples={gt['relative_pair_count']}"
            )
    for warning in summary.get("warnings", []):
        print(f"- Warning: {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = collect_calibration_metrics(
        args.dataset,
        args.outputs,
        evaluate_gt=args.evaluate_gt,
        camera_name=args.camera,
        max_pairs=args.max_pairs,
        strict=args.strict,
        experiment_id=args.experiment_id,
        setup_name=args.setup_name,
        fov=args.fov,
    )
    paths = {
        "summary_json": args.output / "calibration_metrics_summary.json",
        "per_camera_csv": args.output / "per_camera_metrics.csv",
        "link_ranking_csv": args.output / "link_ranking_table.csv",
        "pairwise_relative_csv": args.output / "pairwise_relative_metrics.csv",
        "gt_free_quality_csv": args.output / "gt_free_quality_metrics.csv",
        "report": args.output / "report.md",
        "plot_directory": args.output / "plots",
    }
    write_csv(paths["per_camera_csv"], summary["per_camera"], PER_CAMERA_COLUMNS)
    write_csv(
        paths["link_ranking_csv"], summary["link_ranking"], LINK_RANKING_COLUMNS
    )
    write_csv(
        paths["pairwise_relative_csv"],
        summary["pairwise_relative_metrics"],
        PAIRWISE_COLUMNS,
    )
    write_csv(paths["gt_free_quality_csv"], summary["per_camera"], GT_FREE_COLUMNS)
    if args.save_plots:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(args.output / ".matplotlib")
        )
        from .calibration_evaluation.plots import generate_plots

        summary["plot_paths"] = generate_plots(
            summary, paths["plot_directory"], summary["warnings"]
        )
    summary["output_paths"] = {key: str(value) for key, value in paths.items()}
    summary["csv_paths"] = {
        key: str(paths[key])
        for key in (
            "per_camera_csv",
            "link_ranking_csv",
            "pairwise_relative_csv",
            "gt_free_quality_csv",
        )
    }
    summary["report_path"] = str(paths["report"])
    write_report(paths["report"], summary)
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
