"""Export depth-model-compatible absolute T_base_cam poses per frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .final_calibration_export import export_base_camera_pose_jsonls


def print_summary(validation: Mapping[str, Any], output_dir: Path) -> None:
    print("Final absolute camera-pose export summary")
    print(
        f"- Exported {validation['frame_count']} frames for "
        f"{validation['camera_count']} cameras"
    )
    for camera in validation["cameras"]:
        if camera["gt_validation_available"]:
            translation = camera["translation_error_m"]
            rotation = camera["rotation_error_deg"]
            print(
                f"- {camera['camera_name']}: evaluated={camera['frames_evaluated']}, "
                f"translation_mean_m={translation['mean']:.6f}, "
                f"translation_max_m={translation['max']:.6f}, "
                f"rotation_mean_deg={rotation['mean']:.4f}, "
                f"rotation_max_deg={rotation['max']:.4f}, "
                f"passed={camera['passed']}"
            )
        else:
            print(f"- {camera['camera_name']}: GT validation unavailable")
    print(f"- Camera poses: {output_dir / 'camera_poses_base'}")
    if validation["gt_validation_available"]:
        print(f"- Validation: {output_dir / 'final_camera_pose_validation.json'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate exported T_base_cam records against dataset camera GT",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    validation = export_base_camera_pose_jsonls(
        args.dataset,
        calibration,
        args.output,
        evaluate_gt=args.evaluate_gt,
    )
    if args.evaluate_gt and validation["gt_validation_available"]:
        validation_path = args.output / "final_camera_pose_validation.json"
        validation_path.write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
    print_summary(validation, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
