"""Export and validate depth-model-compatible calibrated camera poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .depth_model_compat import export_depth_model_compatibility


def print_summary(
    report: Mapping[str, Any],
    validation: Mapping[str, Any] | None,
    output_root: Path,
) -> None:
    print("Depth-model pose compatibility summary")
    print(f"- Exported cameras: {report['camera_count']}")
    for camera_name, count in report["per_camera_exported_frame_counts"].items():
        print(f"- {camera_name}: frames={count}")
    print(
        f"- Per-frame transform JSONs: "
        f"{report['total_transform_json_files_generated']}"
    )
    print(f"- Sample manifest samples: {report['sample_manifest_sample_count']}")
    print(
        f"- Parser compatibility: "
        f"{report['parser_compatibility_smoke_test_passed']}"
    )
    print(
        f"- Relative-pose smoke test: "
        f"{report['relative_pose_computation_smoke_test_passed']}"
    )
    if validation is not None:
        for camera in validation["absolute_pose"]["cameras"]:
            if camera["count"] == 0:
                continue
            print(
                f"- Absolute {camera['camera_name']}: n={camera['count']}, "
                f"translation_mean_m={camera['translation_error_m']['mean']:.6f}, "
                f"rotation_mean_deg={camera['rotation_error_deg']['mean']:.4f}"
            )
        relative = validation["relative_pose"]["T_src_tgt"]
        if relative["count"]:
            print(
                f"- Relative T_src_tgt: pairs={relative['count']}, "
                f"translation_mean_m={relative['translation_error_m']['mean']:.6f}, "
                f"rotation_mean_deg={relative['rotation_error_deg']['mean']:.4f}"
            )
    for warning in report["warnings"]:
        print(f"- Warning: {warning}")
    print(f"- Compatibility report: {output_root / 'depth_model_compatibility_report.json'}")
    if report["gt_validation_report_written"]:
        print(f"- GT validation: {output_root / 'depth_model_pose_validation.json'}")
    if report["sample_manifest_written"]:
        print(f"- Sample manifest: {output_root / 'depth_model_samples.json'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--camera")
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--write-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-per-frame-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-sample-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    report, validation = export_depth_model_compatibility(
        args.dataset,
        calibration,
        args.calibration,
        args.output,
        evaluate_gt=args.evaluate_gt,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        camera_name=args.camera,
        write_jsonl=args.write_jsonl,
        write_per_frame_json=args.write_per_frame_json,
        write_sample_manifest=args.write_sample_manifest,
    )
    report_path = args.output / "depth_model_compatibility_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if validation is not None:
        validation_path = args.output / "depth_model_pose_validation.json"
        validation_path.write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
    print_summary(report, validation, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
