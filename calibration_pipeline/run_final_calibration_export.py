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


def _report_reference(path: Path) -> str:
    value = str(path)
    return value if path.is_absolute() or value.startswith("./") else f"./{value}"


def build_mount_gt_validation(
    calibration: dict[str, Any], dataset_root: Path
) -> dict[str, Any]:
    """Build an evaluation-only report without mutating deployment cameras."""
    setup_path = dataset_root / "setup_used.json"
    if not setup_path.is_file():
        return {
            "schema_version": 1,
            "evaluation_only": True,
            "gt_validation_available": False,
            "ground_truth_used_to_modify_calibration": False,
            "reason": "setup_used_json_not_found",
            "camera_count": len(calibration["cameras"]),
            "cameras_evaluated": 0,
            "cameras": [],
        }
    evaluator = GroundTruthMountEvaluator(dataset_root)
    records = []
    for camera in calibration["cameras"]:
        result = evaluator.evaluate(
            {
                "camera_name": camera["camera_name"],
                "best_link_path_rel": camera["attached_link"],
                "T_link_camera_rowmajor": camera["T_link_camera_rowmajor"],
            }
        )
        records.append(
            {
                "camera_name": camera["camera_name"],
                "attached_link": camera["attached_link"],
                "gt_validation_available": result.get(
                    "gt_evaluation_available", False
                ),
                "gt_attached_link_correct": result.get("gt_best_link_correct"),
                "gt_T_link_camera_translation_error_m": result.get(
                    "gt_T_link_camera_translation_error_m"
                ),
                "gt_T_link_camera_rotation_error_deg": result.get(
                    "gt_T_link_camera_rotation_error_deg"
                ),
            }
        )
    evaluated = [record for record in records if record["gt_validation_available"]]
    return {
        "schema_version": 1,
        "evaluation_only": True,
        "gt_validation_available": bool(evaluated),
        "ground_truth_used_to_modify_calibration": False,
        "reason": "ok" if evaluated else "no_matching_gt_camera_mounts",
        "camera_count": len(records),
        "cameras_evaluated": len(evaluated),
        "cameras": records,
    }


def print_summary(
    calibration: dict[str, Any],
    output_path: Path,
    static_validation: dict[str, Any] | None,
) -> None:
    print("Final static calibration summary")
    print(f"- Final camera count: {calibration['camera_count']}")
    for camera in calibration["cameras"]:
        warnings = "; ".join(camera["warnings"]) or "none"
        print(
            f"- {camera['camera_name']}: attached_link={camera['attached_link_name']}, "
            f"source={camera['calibration_source']}, "
            f"confidence={camera['confidence']}, warnings={warnings}"
        )
    if static_validation is not None:
        for result in static_validation["cameras"]:
            if not result["gt_validation_available"]:
                continue
            print(
                f"  GT mount {result['camera_name']}: translation_error_m="
                f"{result['gt_T_link_camera_translation_error_m']:.6f}, "
                f"rotation_error_deg="
                f"{result['gt_T_link_camera_rotation_error_deg']:.4f}"
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
    output_dir = args.output / "final_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_calibration.json"
    static_validation = None
    # GT is first loaded after transform selection is complete.
    if args.evaluate_gt:
        static_validation = build_mount_gt_validation(calibration, args.dataset)
        if static_validation["gt_validation_available"]:
            static_path = output_dir / "final_static_calibration_validation.json"
            static_path.write_text(
                json.dumps(static_validation, indent=2) + "\n", encoding="utf-8"
            )
            calibration["validation"] = {
                "gt_validation_available": True,
                "evaluation_only": True,
                "static_validation_report": _report_reference(static_path),
                "camera_pose_validation_report": _report_reference(
                    output_dir / "final_camera_pose_validation.json"
                ),
            }
    output_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(DEPLOYMENT_README, encoding="utf-8")
    print_summary(calibration, output_path, static_validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
