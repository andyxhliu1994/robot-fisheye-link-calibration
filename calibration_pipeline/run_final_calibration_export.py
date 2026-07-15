"""Build the final deployment-oriented static camera calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .final_calibration_export import build_final_calibration
from .run_link_calibration import GroundTruthMountEvaluator


DEPLOYMENT_README = """# Final static calibration

`final_calibration.json` contains static camera-to-link calibration only.

Runtime code must obtain `T_base_link` from forward kinematics or recorded link
poses, then compute `T_base_cam = T_base_link @ T_link_camera`. Do not feed
`T_link_camera` directly to the depth model: it expects an absolute
`T_base_cam` for every camera and frame.

The camera-ray adapter was already applied while estimating `T_camera_board`.
Do not apply that adapter again to `T_link_camera`.
"""


def add_mount_gt_validation(calibration: dict[str, Any], dataset_root: Path) -> None:
    """Validate finalized selections; this never changes selected transforms."""
    evaluator = GroundTruthMountEvaluator(dataset_root)
    for camera in calibration["cameras"]:
        result = evaluator.evaluate(
            {
                "camera_name": camera["camera_name"],
                "best_link_path_rel": camera["attached_link"],
                "T_link_camera_rowmajor": camera["T_link_camera_rowmajor"],
            }
        )
        camera["gt_validation_available"] = result.get(
            "gt_evaluation_available", False
        )
        camera["gt_attached_link_correct"] = result.get("gt_best_link_correct")
        camera["gt_T_link_camera_translation_error_m"] = result.get(
            "gt_T_link_camera_translation_error_m"
        )
        camera["gt_T_link_camera_rotation_error_deg"] = result.get(
            "gt_T_link_camera_rotation_error_deg"
        )


def print_summary(calibration: dict[str, Any], output_path: Path) -> None:
    print("Final static calibration summary")
    print(f"- Final camera count: {calibration['camera_count']}")
    for camera in calibration["cameras"]:
        warnings = "; ".join(camera["warnings"]) or "none"
        print(
            f"- {camera['camera_name']}: attached_link={camera['attached_link_name']}, "
            f"source={camera['calibration_source']}, "
            f"confidence={camera['confidence']}, warnings={warnings}"
        )
        if camera.get("gt_validation_available"):
            print(
                f"  GT mount: translation_error_m="
                f"{camera['gt_T_link_camera_translation_error_m']:.6f}, "
                f"rotation_error_deg="
                f"{camera['gt_T_link_camera_rotation_error_deg']:.4f}"
            )
    adapter = calibration["frame_adapters"]["camera_ray_to_camera_pose_adapter"]
    print(
        f"- Ray adapter: {adapter['name']} ({adapter['from_frame']} -> "
        f"{adapter['to_frame']}), matrix included"
    )
    print(f"- Output: {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--link-calibration", type=Path, required=True)
    parser.add_argument("--shared-board-recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate finalized static mounts against Unity GT",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    link_summary = json.loads(args.link_calibration.read_text(encoding="utf-8"))
    recovery_summary = json.loads(
        args.shared_board_recovery.read_text(encoding="utf-8")
    )
    calibration = build_final_calibration(
        args.dataset,
        link_summary,
        recovery_summary,
        link_calibration_path=args.link_calibration,
        shared_board_recovery_path=args.shared_board_recovery,
    )
    # GT is first loaded after transform selection is complete.
    if args.evaluate_gt:
        add_mount_gt_validation(calibration, args.dataset)
    output_dir = args.output / "final_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_calibration.json"
    output_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(DEPLOYMENT_README, encoding="utf-8")
    print_summary(calibration, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
