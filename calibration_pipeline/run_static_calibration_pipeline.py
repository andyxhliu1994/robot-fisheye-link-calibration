"""Run the existing static camera-mount calibration stages in sequence."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    run_final_calibration_export,
    run_link_calibration,
    run_shared_board_recovery,
)


MISSING_BOARD_POSES_MESSAGE = (
    "Board poses not found. Run run_pose_from_charuco first."
)


@dataclass(frozen=True)
class StaticCalibrationPaths:
    link_calibration: Path
    shared_board_recovery: Path
    final_calibration: Path


def standard_output_paths(output_root: Path) -> StaticCalibrationPaths:
    """Return the standard outputs produced by the three wrapped CLIs."""
    return StaticCalibrationPaths(
        link_calibration=(
            output_root / "link_calibration" / "link_calibration_summary.json"
        ),
        shared_board_recovery=(
            output_root
            / "shared_board_recovery"
            / "shared_board_recovery_summary.json"
        ),
        final_calibration=(
            output_root / "final_calibration" / "final_calibration.json"
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--board-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera")
    parser.add_argument("--min-valid-poses", type=int, default=10)
    parser.add_argument("--min-anchor-cameras", type=int, default=2)
    parser.add_argument("--allow-single-anchor", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass optional Unity GT evaluation through every wrapped stage",
    )
    return parser.parse_args(argv)


def _validate_board_poses(board_pose_dir: Path, camera_name: str | None) -> None:
    if not board_pose_dir.is_dir():
        raise SystemExit(MISSING_BOARD_POSES_MESSAGE)
    if camera_name is not None:
        found = (board_pose_dir / f"{camera_name}.jsonl").is_file()
    else:
        found = any(board_pose_dir.glob("*.jsonl"))
    if not found:
        raise SystemExit(MISSING_BOARD_POSES_MESSAGE)


def build_stage_arguments(
    args: argparse.Namespace, paths: StaticCalibrationPaths
) -> dict[str, list[str]]:
    """Build argv lists for the existing lower-level CLIs."""
    evaluation_flag = "--evaluate-gt" if args.evaluate_gt else "--no-evaluate-gt"
    link_arguments = [
        "--dataset",
        str(args.dataset),
        "--board-poses",
        str(args.board_poses),
        "--output",
        str(args.output),
        "--min-valid-poses",
        str(args.min_valid_poses),
        evaluation_flag,
    ]
    recovery_arguments = [
        "--dataset",
        str(args.dataset),
        "--link-calibration",
        str(paths.link_calibration),
        "--board-poses",
        str(args.board_poses),
        "--output",
        str(args.output),
        "--min-anchor-cameras",
        str(args.min_anchor_cameras),
        evaluation_flag,
    ]
    if args.camera is not None:
        link_arguments.extend(["--camera", args.camera])
        recovery_arguments.extend(["--camera", args.camera])
    if args.allow_single_anchor:
        recovery_arguments.append("--allow-single-anchor")
    final_arguments = [
        "--dataset",
        str(args.dataset),
        "--link-calibration",
        str(paths.link_calibration),
        "--shared-board-recovery",
        str(paths.shared_board_recovery),
        "--output",
        str(args.output),
        evaluation_flag,
    ]
    return {
        "link_calibration": link_arguments,
        "shared_board_recovery": recovery_arguments,
        "final_calibration_export": final_arguments,
    }


def _print_dry_run(arguments: Mapping[str, Sequence[str]]) -> None:
    modules = {
        "link_calibration": "calibration_pipeline.run_link_calibration",
        "shared_board_recovery": (
            "calibration_pipeline.run_shared_board_recovery"
        ),
        "final_calibration_export": (
            "calibration_pipeline.run_final_calibration_export"
        ),
    }
    print("Static calibration pipeline dry run.")
    for name, stage_arguments in arguments.items():
        command = ["python", "-m", modules[name], *stage_arguments]
        print(f"- {shlex.join(command)}")


def _run_stage(
    stage_name: str,
    entrypoint,
    arguments: Sequence[str],
    expected_output: Path,
) -> None:
    exit_code = entrypoint(arguments)
    if exit_code not in (None, 0):
        raise SystemExit(f"{stage_name} failed with exit code {exit_code}")
    if not expected_output.is_file():
        raise SystemExit(
            f"{stage_name} did not produce expected output: {expected_output}"
        )


def print_summary(
    paths: StaticCalibrationPaths,
    calibration: Mapping[str, Any],
    *,
    evaluate_gt: bool,
) -> None:
    cameras = list(calibration.get("cameras", []))
    recovery_used = any(
        camera.get("calibration_source") == "shared_board_recovery"
        for camera in cameras
    )
    print("Static calibration pipeline complete.")
    print(f"- link calibration: {paths.link_calibration}")
    print(f"- shared recovery: {paths.shared_board_recovery}")
    print(f"- final calibration: {paths.final_calibration}")
    print(f"- cameras: {len(cameras)}")
    print(f"- GT evaluation: {'enabled' if evaluate_gt else 'disabled'}")
    print(
        "- shared-board recovery used: "
        f"{'yes' if recovery_used else 'no'}"
    )
    for camera in cameras:
        warnings = "; ".join(str(item) for item in camera.get("warnings", []))
        print(
            f"- {camera['camera_name']}: "
            f"attached_link={camera['attached_link']}, "
            f"source={camera['calibration_source']}, "
            f"confidence={camera['confidence']}, "
            f"warnings={warnings or 'none'}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_board_poses(args.board_poses, args.camera)
    paths = standard_output_paths(args.output)
    stage_arguments = build_stage_arguments(args, paths)
    if args.dry_run:
        _print_dry_run(stage_arguments)
        return 0

    _run_stage(
        "Link calibration",
        run_link_calibration.main,
        stage_arguments["link_calibration"],
        paths.link_calibration,
    )
    _run_stage(
        "Shared-board recovery",
        run_shared_board_recovery.main,
        stage_arguments["shared_board_recovery"],
        paths.shared_board_recovery,
    )
    _run_stage(
        "Final calibration export",
        run_final_calibration_export.main,
        stage_arguments["final_calibration_export"],
        paths.final_calibration,
    )
    calibration = json.loads(paths.final_calibration.read_text(encoding="utf-8"))
    print_summary(paths, calibration, evaluate_gt=args.evaluate_gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
