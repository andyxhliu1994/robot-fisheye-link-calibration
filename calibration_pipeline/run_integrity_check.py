"""CLI for the first-milestone dataset integrity report."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .dataset_loader import DatasetLoader, frame_number, normalize_frame_id
from .evaluation import evaluate_gt_consistency


def _selected_frames(
    all_frame_ids: list[str], start: int | None, stop: int | None, step: int
) -> list[str]:
    if step <= 0:
        raise ValueError("--frame-step must be positive")
    selected = [
        value
        for value in all_frame_ids
        if (start is None or frame_number(value) >= start)
        and (stop is None or frame_number(value) < stop)
        and ((frame_number(value) - (start or 0)) % step == 0)
    ]
    return selected


def _even_subset(frame_ids: list[str], count: int) -> list[str]:
    if count <= 0 or len(frame_ids) <= count:
        return frame_ids
    indices = {round(i * (len(frame_ids) - 1) / (count - 1)) for i in range(count)}
    return [frame_ids[index] for index in sorted(indices)]


def build_report(
    dataset_root: Path,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int = 1,
    gt_sample_count: int = 50,
) -> dict:
    loader = DatasetLoader(dataset_root)
    joint_ids, _ = loader.csv_frame_ids("joint_states.csv")
    selected = _selected_frames(joint_ids, frame_start, frame_stop, frame_step)
    alignment = loader.inspect_frame_alignment(selected)
    config = loader.inspect_camera_configuration()
    gt_frames = _even_subset(selected, gt_sample_count)
    gt = evaluate_gt_consistency(dataset_root, gt_frames)
    passed = alignment["passed"] and config["passed"] and gt["passed"]
    return {
        "schema_version": 1,
        "milestone": "first_dataset_integrity_check",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "passed": passed,
        "pipeline_metadata": {
            "kinematics_provider": "UnityLinkPoseProvider",
            "base_frame": "base",
            "candidate_links_source": "candidate_links.json",
            "ground_truth_used_for_estimation": False,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "selection": {
            "frame_start_inclusive": frame_start,
            "frame_stop_exclusive": frame_stop,
            "frame_step": frame_step,
            "selected_frame_count": len(selected),
            "gt_sample_count_requested": gt_sample_count,
            "gt_frame_ids": gt_frames,
        },
        "frame_alignment": alignment,
        "camera_configuration": config,
        "gt_transform_consistency": gt,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--gt-sample-count", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.dataset,
            args.frame_start,
            args.frame_stop,
            args.frame_step,
            args.gt_sample_count,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Integrity check could not run: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Integrity report: {args.output}")
    print(f"Frame alignment: {'PASS' if report['frame_alignment']['passed'] else 'FAIL'}")
    print(f"Camera configuration: {'PASS' if report['camera_configuration']['passed'] else 'FAIL'}")
    print(f"GT transform consistency: {'PASS' if report['gt_transform_consistency']['passed'] else 'FAIL'}")
    print(f"Overall: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

