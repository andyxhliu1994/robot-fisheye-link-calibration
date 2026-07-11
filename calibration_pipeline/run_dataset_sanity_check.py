"""CLI for Milestone 2.5 camera-stream and transform sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset_sanity import run_dataset_sanity_check


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=10)
    return parser.parse_args(argv)


def print_summary(report: dict) -> None:
    identity = report["rgb_same_frame_identity"]
    print("Camera dataset sanity summary")
    print(f"- Status: {report['status']} ({report['diagnosis_code']})")
    print(
        f"- Sampled frames: {report['sampled_frame_count']}; cameras: "
        f"{len(report['camera_names'])}"
    )
    print(
        "- All-camera byte-identical frames: "
        f"{identity['all_cameras_identical_frame_count']}/"
        f"{report['sampled_frame_count']}"
    )
    for item in report["rgb_across_frame_changes"]["per_camera_summary"]:
        print(
            f"- {item['camera_name']}: unique_hashes={item['unique_sha256_count']}/"
            f"{item['sampled_frame_count']}, changed_adjacent="
            f"{item['changed_adjacent_count']}/{item['adjacent_comparison_count']}"
        )
    transforms = report["camera_transform_distinctness"]
    print(
        "- Camera transforms distinct across cameras: "
        f"{transforms['transforms_different_across_cameras']}"
    )
    detections = report["detection_output_consistency"]
    print(
        "- Detection JSONL identical ignoring camera_name: "
        f"{detections['contents_identical_ignoring_camera_name']}"
    )
    print(f"- Diagnosis: {report['diagnosis']}")
    print(f"- Recommendation: {report['recommendation']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_dataset_sanity_check(
        args.dataset,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        output_root_for_detections=args.output.parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"- Report: {args.output}")
    # A diagnosed dataset FAIL is a successful execution of this evaluation tool.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

